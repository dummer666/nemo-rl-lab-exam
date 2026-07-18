#!/usr/bin/env python
"""Evaluate merged retrieval-SFT checkpoints with the production multi-turn protocol."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Sequence

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.retrieval.markdown_bm25 import MarkdownBM25Index  # noqa: E402
from common.retrieval.qa_agent import QARetrievalRunner  # noqa: E402
from common.retrieval.qa_sft import format_agent_prompt  # noqa: E402
from common.rewards.qa_reward import qa_rule_reward_fn  # noqa: E402

SFT_ROOT = Path(
    "/shared/outputs/wanghaonan/sft_qwen3.5-9b_qa-retrieval-v1_wanghaonan/"
    "sft_qwen3.5-9b_qa-retrieval-v1_wanghaonan-wanghaonan-20260718-130828"
)
CHECKPOINTS = {
    25: SFT_ROOT / "hf_export" / "step_25",
    50: SFT_ROOT / "hf_export" / "step_50",
}
MAX_ROLLOUT_TURNS = 3
MAX_NEW_TOKENS = 384
MAX_INPUT_TOKENS = 5760
BATCH_SIZE = 16
_EXPECTED_TYPE = re.compile(r"^\s*\[(\w+)\]")
_SEARCH_CLOSE = re.compile(r"</search\s*>", re.IGNORECASE)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def _output_dir(overrides: Sequence[str]) -> Path:
    for override in overrides:
        if override.startswith("logger.log_dir="):
            output = Path(override.split("=", 1)[1]).parent / "sft_multiturn_eval"
            output.mkdir(parents=True, exist_ok=True)
            return output
    output = THIS_DIR / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            if not row.get("query") or not row.get("expected_answer"):
                raise ValueError(f"{path}:{line_number}: missing query or expected_answer")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _question_type(expected: str) -> str:
    match = _EXPECTED_TYPE.match(str(expected))
    return match.group(1).lower() if match else "unknown"


def _truncate_at_search_close(response: str) -> str:
    text = str(response).strip()
    match = _SEARCH_CLOSE.search(text)
    return text[: match.end()].strip() if match else text


def _initial_state(row: dict, row_index: int, tokenizer) -> dict:
    query = str(row["query"])
    expected = str(row["expected_answer"])
    row_meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    metadata = {
        "expected_answer": expected,
        "query": query,
        "bank": str(row_meta.get("bank", "")),
        "search_count": 0,
        "search_queries": [],
        "invalid_count": 0,
        "force_search": False,
        "evidence_hits": [],
        "evidence_coverage": 0.0,
        "curriculum_step": 0,
        "curriculum_phase": "validation",
    }
    return {
        "row_index": row_index,
        "question_type": _question_type(expected),
        "query": query,
        "expected_answer": expected,
        "bank": metadata["bank"],
        "transcript": format_agent_prompt(tokenizer, query),
        "assistant_responses": [],
        "environment_observations": [],
        "metadata": metadata,
        "terminated": False,
        "reward": 0.0,
        "protocol_error": False,
        "invalid_turns": 0,
        "termination_reason": None,
    }


def _generate_responses(model, tokenizer, prompts: Sequence[str], *, label: str) -> list[str]:
    import torch

    responses = []
    for start in range(0, len(prompts), BATCH_SIZE):
        batch = list(prompts[start : start + BATCH_SIZE])
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        lengths = encoded["attention_mask"].sum(dim=1).tolist()
        if max(lengths, default=0) > MAX_INPUT_TOKENS:
            raise RuntimeError(f"{label}: input length {max(lengths)} exceeds {MAX_INPUT_TOKENS}")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        input_width = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stop_strings=["</search>"],
                tokenizer=tokenizer,
            )
        decoded = tokenizer.batch_decode(
            generated[:, input_width:],
            skip_special_tokens=True,
        )
        responses.extend(_truncate_at_search_close(text) for text in decoded)
        completed = min(start + len(batch), len(prompts))
        print(f"[sft-eval] {label} generated {completed}/{len(prompts)}", flush=True)
    return responses


def _run_rollouts(model, tokenizer, rows: Sequence[dict], index: MarkdownBM25Index) -> list[dict]:
    runner = QARetrievalRunner(
        index,
        qa_rule_reward_fn,
        max_searches=2,
        max_invalid_actions=2,
        top_k=4,
        candidate_k=50,
        quality_rerank=True,
        max_result_chars=1800,
        per_result_chars=360,
    )
    states = [_initial_state(row, row_index, tokenizer) for row_index, row in enumerate(rows)]

    for turn_index in range(MAX_ROLLOUT_TURNS):
        active = [state for state in states if not state["terminated"]]
        if not active:
            break
        responses = _generate_responses(
            model,
            tokenizer,
            [state["transcript"] for state in active],
            label=f"turn={turn_index + 1}",
        )
        for state, response in zip(active, responses, strict=True):
            state["assistant_responses"].append(response)
            state["transcript"] += response
            turn = runner.process(response, state["metadata"])
            observation = turn.observation
            state["environment_observations"].append(observation)

            if observation.startswith(("[格式", "[检索次数已用完]")):
                state["protocol_error"] = True
                state["invalid_turns"] += 1
            if turn.reward < 0:
                state["protocol_error"] = True

            if turn.metadata is not None:
                state["metadata"] = turn.metadata
            if turn.terminated:
                state["terminated"] = True
                state["reward"] = float(turn.reward)
                state["termination_reason"] = (
                    "final_answer" if observation.startswith("[最终答案已提交]") else "protocol_termination"
                )
            else:
                state["transcript"] += observation

    results = []
    for state in states:
        if not state["terminated"]:
            state["protocol_error"] = True
            state["termination_reason"] = "max_rollout_turns"
        metadata = state.pop("metadata")
        state["search_count"] = int(metadata.get("search_count", 0))
        state["search_queries"] = list(metadata.get("search_queries", []))
        state["evidence_coverage"] = float(metadata.get("evidence_coverage", 0.0))
        state["evidence_hits"] = list(metadata.get("evidence_hits", []))
        state["assistant_turns"] = len(state["assistant_responses"])
        results.append(state)
    return results


def _metric_block(rows: Sequence[dict]) -> dict:
    if not rows:
        return {
            "count": 0,
            "accuracy": 0.0,
            "mean_reward": 0.0,
            "perfect_count": 0,
            "perfect_rate": 0.0,
        }
    clipped = [max(0.0, min(1.0, float(row["reward"]))) for row in rows]
    perfect_count = sum(reward >= 1.0 for reward in clipped)
    return {
        "count": len(rows),
        "accuracy": mean(clipped),
        "mean_reward": mean(float(row["reward"]) for row in rows),
        "perfect_count": perfect_count,
        "perfect_rate": perfect_count / len(rows),
    }


def _summarize(rows: Sequence[dict]) -> dict:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[str(row["question_type"])].append(row)

    search_counts = [int(row["search_count"]) for row in rows]
    protocol_errors = sum(bool(row["protocol_error"]) for row in rows)
    open_rows = [row for row in rows if row["question_type"] in {"fill", "short"}]
    summary = {
        **_metric_block(rows),
        "question_types": {
            question_type: {
                **_metric_block(group),
                "retrieval_count": sum(int(row["search_count"]) > 0 for row in group),
                "average_searches": mean(int(row["search_count"]) for row in group),
            }
            for question_type, group in sorted(by_type.items())
        },
        "retrieval": {
            "retrieval_count": sum(count > 0 for count in search_counts),
            "retrieval_rate": (sum(count > 0 for count in search_counts) / len(rows) if rows else 0.0),
            "one_search_count": sum(count == 1 for count in search_counts),
            "two_search_count": sum(count == 2 for count in search_counts),
            "average_searches": mean(search_counts) if search_counts else 0.0,
            "open_retrieval_rate": (
                sum(int(row["search_count"]) > 0 for row in open_rows) / len(open_rows) if open_rows else 0.0
            ),
        },
        "protocol": {
            "error_count": protocol_errors,
            "error_rate": protocol_errors / len(rows) if rows else 0.0,
            "unterminated_count": sum(not bool(row["terminated"]) for row in rows),
        },
        "open_questions": _metric_block(open_rows),
    }
    return summary


def _load_model(model_path: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        .eval()
        .to("cuda")
    )
    return tokenizer, model


def main() -> None:
    _, overrides = _parse_args()
    output_dir = _output_dir(overrides)
    data_dir = Path(os.environ.get("QA_RL_DATA_DIR", "/data/datasets/qa_rl"))
    docs_dir = Path(os.environ.get("QA_DOCS_DIR", "/data/docs"))
    val_path = data_dir / "val.jsonl"
    required = [val_path, docs_dir, *CHECKPOINTS.values()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing evaluation inputs: {missing}")

    rows = _read_jsonl(val_path)
    print(f"[sft-eval] validation rows={len(rows)}", flush=True)
    index_start = time.perf_counter()
    index = MarkdownBM25Index.from_directory(
        docs_dir,
        chunk_chars=1200,
        overlap_chars=160,
        k1=1.5,
        b=0.75,
    )
    index_seconds = time.perf_counter() - index_start
    print(
        f"[sft-eval] indexed {index.num_documents} chunks in {index_seconds:.1f}s; "
        f"quality={index.quality_category_counts}",
        flush=True,
    )

    summaries = {}
    for step, model_path in CHECKPOINTS.items():
        step_start = time.perf_counter()
        tokenizer, model = _load_model(model_path)
        load_seconds = time.perf_counter() - step_start
        print(f"[sft-eval] step={step} model loaded in {load_seconds:.1f}s", flush=True)
        results = _run_rollouts(model, tokenizer, rows, index)
        summary = {
            "step": step,
            "model_path": str(model_path),
            "validation_path": str(val_path),
            "docs_path": str(docs_dir),
            "generation": {
                "mode": "greedy",
                "batch_size": BATCH_SIZE,
                "max_new_tokens": MAX_NEW_TOKENS,
                "max_rollout_turns": MAX_ROLLOUT_TURNS,
            },
            "model_load_seconds": load_seconds,
            "evaluation_seconds": time.perf_counter() - step_start,
            **_summarize(results),
        }
        step_dir = output_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(step_dir / "trajectories.jsonl", results)
        (step_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summaries[str(step)] = summary
        print(
            f"[sft-eval] step={step} accuracy={summary['accuracy']:.4f} "
            f"open_perfect={summary['open_questions']['perfect_count']} "
            f"retrieval={summary['retrieval']['retrieval_rate']:.2%} "
            f"protocol_errors={summary['protocol']['error_count']}",
            flush=True,
        )

        del model, tokenizer
        gc.collect()
        import torch

        torch.cuda.empty_cache()

    combined = {
        "index_seconds": index_seconds,
        "num_documents": index.num_documents,
        "summaries": summaries,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[sft-eval] complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
