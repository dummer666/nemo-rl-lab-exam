#!/usr/bin/env python
"""Build strictly verified one- and two-search QA trajectories for SFT."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Sequence

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.retrieval.evidence import text_keypoint_hits  # noqa: E402
from common.retrieval.markdown_bm25 import (  # noqa: E402
    MarkdownBM25Index,
    build_retrieval_query,
    format_search_results,
)
from common.retrieval.qa_sft import (  # noqa: E402
    assign_open_splits,
    build_objective_messages,
    build_search_messages,
    canonical_answer,
    format_agent_prompt,
    parse_query_candidate,
    query_rejection_reason,
    validate_messages,
    visible_retrieval_text,
)

DEFAULT_SELECTION_DIR = (
    "/shared/outputs/wanghaonan/qa_sft_data_select_wanghaonan/"
    "qa_sft_data_select_wanghaonan-wanghaonan-20260718-111114/"
    "sft_selection"
)
DEFAULT_CLEAN_TRAIN_PATH = (
    "/shared/outputs/wanghaonan/qa_training_clean_wanghaonan/"
    "qa_training_clean_wanghaonan-wanghaonan-20260718-092437/"
    "cleaned_data/clean_train.jsonl"
)
MODEL_NAME = "Qwen/Qwen3.5-9B"
REWRITE_SYSTEM_PROMPT = """
你是检索 Agent 的查询改写器。根据题目、首次查询和首次检索结果，
生成一个不同且更具体的第二次检索关键词，用于补齐首次结果缺少的信息。

规则：
- 只能使用题目或首次检索结果中已经出现的实体、设备、流程或属性；
- 不得猜测或直接写最终答案；
- 不得重复首次查询；
- 只输出一行检索关键词，不要解释，不要标签，不要编号。
""".strip()


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Build verified retrieval SFT data")
    parser.add_argument("--config")
    return parser.parse_known_args()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _output_dir(overrides: Sequence[str]) -> Path:
    for override in overrides:
        if override.startswith("logger.log_dir="):
            output = Path(override.split("=", 1)[1]).parent / "sft_trajectories"
            output.mkdir(parents=True, exist_ok=True)
            return output
    output = THIS_DIR / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _initial_runtime_prompt(tokenizer, messages: list[dict]) -> str:
    if [message.get("role") for message in messages[:2]] != ["system", "user"]:
        raise ValueError("trajectory must start with system and user messages")
    return format_agent_prompt(
        tokenizer,
        str(messages[1]["content"]),
        system_prompt=str(messages[0]["content"]),
    )


def _runtime_messages(tokenizer, messages: list[dict]) -> list[dict]:
    """Match GRPO: preformatted initial prompt, then raw generated/env chunks."""
    return [
        {
            "role": "user",
            "content": _initial_runtime_prompt(tokenizer, messages),
        },
        *[dict(message) for message in messages[2:]],
    ]


def _trajectory_token_audit(
    trajectory: dict,
    tokenizer,
    *,
    max_tokens: int = 6000,
) -> tuple[dict | None, str | None]:
    messages = trajectory["messages"]
    issues = validate_messages(messages)
    if issues:
        return None, "message_validation:" + ",".join(issues)

    runtime_messages = _runtime_messages(tokenizer, messages)
    expected_roles = [
        "user",
        *[
            "assistant" if index % 2 else "environment"
            for index in range(1, len(runtime_messages))
        ],
    ]
    if [message["role"] for message in runtime_messages] != expected_roles:
        return None, "runtime_role_order"

    rendered = "".join(message["content"] for message in runtime_messages)
    for message in runtime_messages:
        if (
            message["role"] == "environment"
            and message["content"].startswith("[检索结果]")
            and visible_retrieval_text(message["content"])[:80] not in rendered
        ):
            return None, "observation_not_rendered"
    if runtime_messages[-1]["content"] not in rendered:
        return None, "final_answer_not_rendered"

    token_ids = tokenizer(
        rendered,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]
    token_length = len(token_ids)
    if token_length > max_tokens:
        return None, "trajectory_too_long"

    audited = dict(trajectory)
    audited["messages"] = runtime_messages
    audited["_audit"] = {
        **dict(trajectory.get("_audit") or {}),
        "token_length": token_length,
        "assistant_message_count": sum(
            message["role"] == "assistant"
            for message in runtime_messages
        ),
        "observation_message_count": sum(
            message["role"] == "environment"
            and message["content"].startswith("[检索结果]")
            for message in runtime_messages
        ),
        "rendered_observations_visible": True,
        "runtime_raw_chunk_alignment": True,
    }
    return audited, None


def _rewrite_prompt(record: dict) -> list[dict[str, str]]:
    visible = visible_retrieval_text(record["first_retrieval_output"])
    user_prompt = (
        f"题目：\n{record['query']}\n\n"
        f"首次查询：\n{record['first_search_query']}\n\n"
        f"首次检索结果：\n{visible}\n\n"
        "首次结果不足以完整回答题目。请给出第二次检索关键词。"
    )
    return [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _generation_prompt(tokenizer, record: dict) -> str:
    messages = _rewrite_prompt(record)
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _generate_query_candidates(
    records: Sequence[dict],
    tokenizer,
    model,
    *,
    candidates_per_record: int = 8,
    batch_size: int = 4,
) -> dict[int, list[str]]:
    import torch

    generated: dict[int, list[str]] = {}
    for start in range(0, len(records), batch_size):
        batch = list(records[start : start + batch_size])
        prompts = [_generation_prompt(tokenizer, record) for record in batch]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
            add_special_tokens=False,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        input_width = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            outputs = model.generate(
                **encoded,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.05,
                num_return_sequences=candidates_per_record,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        continuations = tokenizer.batch_decode(
            outputs[:, input_width:],
            skip_special_tokens=True,
        )
        for offset, record in enumerate(batch):
            begin = offset * candidates_per_record
            end = begin + candidates_per_record
            generated[int(record["row_id"])] = continuations[begin:end]
        completed = min(start + len(batch), len(records))
        print(f"[sft-trajectory] teacher rewrites {completed}/{len(records)}")
    return generated


def _evaluate_rewrite_candidate(
    record: dict,
    sampled_text: str,
    index: MarkdownBM25Index,
) -> tuple[dict | None, str]:
    query = parse_query_candidate(sampled_text)
    visible_context = (
        str(record["query"])
        + "\n"
        + visible_retrieval_text(record["first_retrieval_output"])
    )
    reason = query_rejection_reason(
        query,
        first_query=str(record["first_search_query"]),
        visible_context=visible_context,
        keypoints=record["keypoints"],
    )
    if reason:
        return None, reason

    retrieval_query = build_retrieval_query(
        query,
        str(record["query"]),
        str(record.get("bank", "")),
    )
    results = index.search(
        retrieval_query,
        top_k=4,
        candidate_k=50,
        quality_rerank=True,
    )
    rendered = format_search_results(
        results,
        retrieval_query,
        max_chars=1800,
        per_result_chars=360,
    )
    second_hits = text_keypoint_hits(
        visible_retrieval_text(rendered),
        record["keypoints"],
    )
    first_hits = {int(hit) for hit in record["first_observation_hits"]}
    new_hits = second_hits - first_hits
    if not new_hits:
        return None, "no_evidence_gain"
    cumulative_hits = first_hits | second_hits
    if len(cumulative_hits) != len(record["keypoints"]):
        return None, "incomplete_cumulative_evidence"
    return (
        {
            "query": query,
            "retrieval_query": retrieval_query,
            "rendered_observation": rendered,
            "second_hits": sorted(second_hits),
            "new_hits": sorted(new_hits),
            "cumulative_hits": sorted(cumulative_hits),
        },
        "accepted",
    )


def _direct_trajectory(record: dict) -> dict:
    return {
        "row_id": int(record["row_id"]),
        "question_type": str(record["question_type"]),
        "query": str(record["query"]),
        "expected_answer": str(record["expected_answer"]),
        "bank": str(record.get("bank", "")),
        "search_turns": 1,
        "messages": build_search_messages(
            query=str(record["query"]),
            expected=str(record["expected_answer"]),
            first_query=str(record["first_search_query"]),
            first_observation=str(record["first_retrieval_output"]),
        ),
        "_audit": {
            "source_status": "ready_one_search",
            "first_query": str(record["first_search_query"]),
            "first_hits": list(record["first_observation_hits"]),
            "cumulative_coverage": 1.0,
        },
    }


def _rewrite_trajectory(record: dict, accepted: dict, sampled_count: int) -> dict:
    return {
        "row_id": int(record["row_id"]),
        "question_type": str(record["question_type"]),
        "query": str(record["query"]),
        "expected_answer": str(record["expected_answer"]),
        "bank": str(record.get("bank", "")),
        "search_turns": 2,
        "messages": build_search_messages(
            query=str(record["query"]),
            expected=str(record["expected_answer"]),
            first_query=str(record["first_search_query"]),
            first_observation=str(record["first_retrieval_output"]),
            second_query=accepted["query"],
            second_observation=accepted["rendered_observation"],
        ),
        "_audit": {
            "source_status": "needs_query_rewrite",
            "first_query": str(record["first_search_query"]),
            "second_query": accepted["query"],
            "first_hits": list(record["first_observation_hits"]),
            "second_hits": accepted["second_hits"],
            "new_hits": accepted["new_hits"],
            "cumulative_coverage": 1.0,
            "sampled_query_count": sampled_count,
        },
    }


def _select_objective_rows(
    clean_rows: Sequence[dict],
    *,
    train_per_type: int = 24,
    validation_per_type: int = 4,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in clean_rows:
        try:
            question_type, _answer = canonical_answer(
                str(row.get("expected_answer", ""))
            )
        except ValueError:
            continue
        if question_type in {"single", "multiple", "bool"}:
            groups[question_type].append(row)

    train, validation = [], []
    for question_type in ("single", "multiple", "bool"):
        ordered = sorted(
            groups[question_type],
            key=lambda row: hashlib.sha256(
                (
                    f"{seed}:{question_type}:"
                    f"{row.get('_clean', {}).get('row_id')}:{row.get('query', '')}"
                ).encode("utf-8")
            ).hexdigest(),
        )
        required = train_per_type + validation_per_type
        if len(ordered) < required:
            raise ValueError(f"not enough clean objective rows: {question_type}")
        validation.extend(ordered[:validation_per_type])
        train.extend(ordered[validation_per_type:required])
    return train, validation


def _objective_trajectory(row: dict, *, split: str) -> dict:
    clean = row.get("_clean") if isinstance(row.get("_clean"), dict) else {}
    question_type, _answer = canonical_answer(str(row["expected_answer"]))
    return {
        "row_id": int(clean["row_id"]),
        "question_type": question_type,
        "query": str(row["query"]),
        "expected_answer": str(row["expected_answer"]),
        "search_turns": 0,
        "split": split,
        "messages": build_objective_messages(
            query=str(row["query"]),
            expected=str(row["expected_answer"]),
        ),
        "_audit": {
            "source_status": "objective_retention",
            "cumulative_coverage": None,
        },
    }


def _load_teacher():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        .eval()
        .to("cuda")
    )
    torch.manual_seed(42)
    return tokenizer, model


def _assert_split_isolation(records: Sequence[dict]) -> None:
    split_ids: dict[str, set[int]] = defaultdict(set)
    for record in records:
        split_ids[str(record["split"])].add(int(record["row_id"]))
    names = sorted(split_ids)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = split_ids[left] & split_ids[right]
            if overlap:
                raise ValueError(f"row IDs overlap between {left} and {right}: {overlap}")


def main() -> None:
    _, overrides = _parse_args()
    selection_dir = Path(
        os.environ.get("QA_SFT_SELECTION_DIR", DEFAULT_SELECTION_DIR)
    )
    clean_train_path = Path(
        os.environ.get("QA_CLEAN_TRAIN_PATH", DEFAULT_CLEAN_TRAIN_PATH)
    )
    docs_dir = Path(os.environ.get("QA_DOCS_DIR", "/data/docs"))
    manifest_path = selection_dir / "selection_manifest.jsonl"
    for path in (manifest_path, clean_train_path):
        if not path.is_file():
            raise FileNotFoundError(f"required input does not exist: {path}")

    records = _read_jsonl(manifest_path)
    direct_records = [
        record
        for record in records
        if record.get("selection_status") == "ready_one_search"
    ]
    rewrite_records = [
        record
        for record in records
        if record.get("selection_status") == "needs_query_rewrite"
    ]
    if not direct_records or not rewrite_records:
        raise ValueError("selection manifest is missing primary trajectory groups")

    load_start = time.perf_counter()
    tokenizer, model = _load_teacher()
    model_load_seconds = time.perf_counter() - load_start
    prompts = _generate_query_candidates(
        rewrite_records,
        tokenizer,
        model,
        candidates_per_record=8,
        batch_size=4,
    )
    del model
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover - cluster always has torch
        pass

    index_start = time.perf_counter()
    index = MarkdownBM25Index.from_directory(
        docs_dir,
        chunk_chars=1200,
        overlap_chars=160,
        k1=1.5,
        b=0.75,
    )
    index_seconds = time.perf_counter() - index_start

    trajectories = [_direct_trajectory(record) for record in direct_records]
    rewrite_rejections: Counter[str] = Counter()
    rewrite_audit = []
    rewrite_start = time.perf_counter()
    for position, record in enumerate(rewrite_records, start=1):
        sampled = prompts[int(record["row_id"])]
        accepted_candidates = []
        candidate_audit = []
        seen_queries = set()
        for sampled_text in sampled:
            parsed = parse_query_candidate(sampled_text)
            normalized = " ".join(parsed.lower().split())
            if normalized in seen_queries:
                reason = "duplicate_sample"
                accepted = None
            else:
                seen_queries.add(normalized)
                accepted, reason = _evaluate_rewrite_candidate(
                    record,
                    sampled_text,
                    index,
                )
            rewrite_rejections[reason] += int(reason != "accepted")
            candidate_audit.append(
                {
                    "query": parsed,
                    "decision": reason,
                }
            )
            if accepted:
                accepted_candidates.append(accepted)

        selected = None
        if accepted_candidates:
            selected = max(
                accepted_candidates,
                key=lambda candidate: (
                    len(candidate["new_hits"]),
                    -len(candidate["query"]),
                    candidate["query"],
                ),
            )
            trajectories.append(
                _rewrite_trajectory(record, selected, sampled_count=len(sampled))
            )
        rewrite_audit.append(
            {
                "row_id": int(record["row_id"]),
                "question_type": record["question_type"],
                "accepted": selected is not None,
                "selected_query": selected["query"] if selected else None,
                "candidates": candidate_audit,
            }
        )
        if position % 25 == 0:
            print(
                f"[sft-trajectory] rewrite validation {position}/{len(rewrite_records)}; "
                f"accepted={sum(item['accepted'] for item in rewrite_audit)}"
            )
    rewrite_seconds = time.perf_counter() - rewrite_start

    audited_open = []
    trajectory_rejections: Counter[str] = Counter()
    for trajectory in trajectories:
        audited, reason = _trajectory_token_audit(trajectory, tokenizer)
        if audited:
            audited_open.append(audited)
        else:
            trajectory_rejections[str(reason)] += 1
    assigned_open = assign_open_splits(audited_open, seed=42)
    _assert_split_isolation(assigned_open)

    clean_rows = _read_jsonl(clean_train_path)
    objective_train_rows, objective_validation_rows = _select_objective_rows(clean_rows)
    objectives = [
        *(
            _objective_trajectory(row, split="train")
            for row in objective_train_rows
        ),
        *(
            _objective_trajectory(row, split="validation")
            for row in objective_validation_rows
        ),
    ]
    audited_objectives = []
    for trajectory in objectives:
        audited, reason = _trajectory_token_audit(trajectory, tokenizer)
        if not audited:
            raise ValueError(f"objective retention trajectory failed audit: {reason}")
        audited_objectives.append(audited)

    all_trajectories = [*assigned_open, *audited_objectives]
    random.Random(42).shuffle(all_trajectories)
    train = [
        {"messages": record["messages"]}
        for record in all_trajectories
        if record["split"] == "train"
    ]
    validation = [
        {"messages": record["messages"]}
        for record in all_trajectories
        if record["split"] == "validation"
    ]
    rl_holdout = [
        {
            "query": record["query"],
            "expected_answer": record["expected_answer"],
            "meta": {
                "source_row_id": record["row_id"],
                "question_type": record["question_type"],
                "search_turns": record["search_turns"],
                "bank": record["bank"],
            },
        }
        for record in assigned_open
        if record["split"] == "rl_holdout"
    ]
    if not train or not validation or not rl_holdout:
        raise ValueError("trajectory split unexpectedly produced an empty output")

    output_dir = _output_dir(overrides)
    paths = {
        "train": output_dir / "sft_train.jsonl",
        "validation": output_dir / "sft_validation.jsonl",
        "rl_holdout": output_dir / "rl_holdout.jsonl",
        "trajectory_manifest": output_dir / "trajectory_manifest.jsonl",
        "rewrite_audit": output_dir / "rewrite_audit.jsonl",
        "summary": output_dir / "summary.json",
    }
    _write_jsonl(paths["train"], train)
    _write_jsonl(paths["validation"], validation)
    _write_jsonl(paths["rl_holdout"], rl_holdout)
    _write_jsonl(paths["trajectory_manifest"], all_trajectories)
    _write_jsonl(paths["rewrite_audit"], rewrite_audit)

    open_counts = Counter(
        (
            record["split"],
            record["question_type"],
            record["search_turns"],
        )
        for record in assigned_open
    )
    token_lengths = [
        int(record["_audit"]["token_length"])
        for record in all_trajectories
    ]
    accepted_rewrites = sum(
        record["search_turns"] == 2
        for record in assigned_open
    )
    summary = {
        "source_selection": str(manifest_path),
        "source_clean_train": str(clean_train_path),
        "teacher_model": MODEL_NAME,
        "direct_candidates": len(direct_records),
        "rewrite_candidates": len(rewrite_records),
        "accepted_one_search": sum(
            record["search_turns"] == 1
            for record in assigned_open
        ),
        "accepted_two_search": accepted_rewrites,
        "rewrite_acceptance_rate": accepted_rewrites / len(rewrite_records),
        "rewrite_rejection_counts": dict(sorted(rewrite_rejections.items())),
        "trajectory_rejection_counts": dict(sorted(trajectory_rejections.items())),
        "open_split_counts": {
            f"{split}:{question_type}:{turns}": count
            for (split, question_type, turns), count in sorted(open_counts.items())
        },
        "objective_retention": {
            "train": len(objective_train_rows),
            "validation": len(objective_validation_rows),
        },
        "sft_rows": {
            "train": len(train),
            "validation": len(validation),
            "rl_holdout": len(rl_holdout),
        },
        "token_lengths": {
            "min": min(token_lengths),
            "mean": mean(token_lengths),
            "max": max(token_lengths),
        },
        "smoke_samples": [
            {
                "row_id": record["row_id"],
                "split": record["split"],
                "question_type": record["question_type"],
                "search_turns": record["search_turns"],
                "roles": [message["role"] for message in record["messages"]],
                "token_length": record["_audit"]["token_length"],
                "rendered_observations_visible": record["_audit"][
                    "rendered_observations_visible"
                ],
            }
            for record in all_trajectories[:5]
        ],
        "timing_seconds": {
            "model_load": model_load_seconds,
            "index_build": index_seconds,
            "rewrite_validation": rewrite_seconds,
        },
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[sft-trajectory] summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if accepted_rewrites < 10:
        raise RuntimeError(
            f"only {accepted_rewrites} verified two-search trajectories; "
            "at least 10 are required before SFT"
        )


if __name__ == "__main__":
    main()
