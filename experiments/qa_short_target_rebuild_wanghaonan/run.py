#!/usr/bin/env python
"""Rebuild complete short-answer targets from independently verified evidence."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.retrieval.evidence import (  # noqa: E402
    expected_keypoints,
    text_keypoint_hits,
)
from common.retrieval.markdown_bm25 import (  # noqa: E402
    MarkdownBM25Index,
    SearchResult,
    best_snippet,
    build_retrieval_query,
    format_search_results_with_visible_snippets,
    question_context,
)
from common.retrieval.qa_sft import (  # noqa: E402
    build_search_messages,
    format_agent_prompt,
    parse_query_candidate,
    query_rejection_reason,
    validate_messages,
    visible_retrieval_text,
)
from common.retrieval.qa_target_rebuild import (  # noqa: E402
    assign_group_splits,
    bind_visible_evidence,
    extract_json_object,
    non_text_task_reason,
    question_fingerprint,
    rebuilt_expected_answer,
    trusted_visible_quote_hits,
    validate_generated_target,
    verifier_accepts,
)

DEFAULT_CLEAN_TRAIN_PATH = (
    "/shared/outputs/wanghaonan/qa_training_clean_wanghaonan/"
    "qa_training_clean_wanghaonan-wanghaonan-20260718-092437/"
    "cleaned_data/clean_train.jsonl"
)
DEFAULT_AUDIT_DIR = (
    "/shared/outputs/wanghaonan/qa_short_gold_audit_wanghaonan/"
    "qa_short_gold_audit_wanghaonan-wanghaonan-20260718-174624/"
    "short_gold_audit"
)
MODEL_NAME = "Qwen/Qwen3.5-9B"
TARGET_SYSTEM_PROMPT = """
你是技术问答数据的严格证据编辑器。给定一道简答题和若干真实检索片段，
只在资料能够完整回答时，从资料中重建 2 至 6 个自包含、可核验的答案要点。

只能输出一个 JSON 对象，不要输出 Markdown、思考过程或额外文字。
可回答时格式：
{"decision":"answerable","answer_points":[
  {"statement":"完整事实句","evidence_id":"E01","quote":"片段中的连续原文"}
]}
不可回答时格式：
{"decision":"reject","reason":"简短原因"}

硬规则：
- 每个 statement 必须直接回答题目，不能只是“代码、设备、流程、步骤、略”等主题词；
- 每个 quote 必须是对应 evidence_id 文本中 12 至 220 字的连续原文，不得改写；
- statement 只能归纳 quote 中明确出现的事实，不得补充模型常识、数字或实体；
- 各要点必须互不重复，合起来完整回答题目；
- 证据不完整、只是题目/答案标签、要求截图上传画图写代码、或只能得到一个要点时必须 reject；
- 不要猜测旧标准答案；本任务没有向你提供旧标准答案。
""".strip()
VERIFY_SYSTEM_PROMPT = """
你是独立的证据审计员。检查候选简答的每个要点是否由其引用原文明确支持、
是否直接回答题目，以及全部要点是否构成完整答案。不得使用外部知识。

只输出一个 JSON 对象：
{"decision":"accept或reject","complete":true或false,
 "point_checks":[{"index":1,"supported":true或false,"relevant":true或false}],
 "reason":"简短原因"}
只有每个要点都 supported=true、relevant=true，且整体 complete=true 时才 accept。
""".strip()
REWRITE_SYSTEM_PROMPT = """
你是检索 Agent 的查询改写器。题目需要多个答案要点，但首次可见检索结果不足。
请生成一个不同且更具体的第二次检索关键词。

规则：
- 只能使用题目或首次检索结果中已经出现的实体、设备、流程或属性；
- 不得猜测、泄漏或直接写最终答案；
- 不得重复首次查询；
- 只输出一行检索关键词，不要解释、标签或编号。
""".strip()


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _output_dir(overrides: Sequence[str]) -> Path:
    for override in overrides:
        if override.startswith("logger.log_dir="):
            output = Path(override.split("=", 1)[1]).parent / "short_target_rebuild"
            output.mkdir(parents=True, exist_ok=True)
            return output
    output = THIS_DIR / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _chat_prompt(tokenizer, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "add_special_tokens": False,
    }
    try:
        return str(
            tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **kwargs,
            )
        )
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        return str(tokenizer.apply_chat_template(messages, **kwargs))


def _generate(
    prompts: Sequence[str],
    tokenizer,
    model,
    *,
    label: str,
    batch_size: int,
    max_new_tokens: int,
    num_return_sequences: int = 1,
    do_sample: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
) -> list[list[str]]:
    import torch

    all_outputs: list[list[str]] = []
    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start : start + batch_size])
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=5000,
            add_special_tokens=False,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        input_width = encoded["input_ids"].shape[1]
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.03,
        }
        if do_sample:
            generation_kwargs["temperature"] = float(temperature or 0.7)
            generation_kwargs["top_p"] = float(top_p or 0.9)
        with torch.inference_mode():
            generated = model.generate(**encoded, **generation_kwargs)
        continuations = tokenizer.batch_decode(
            generated[:, input_width:],
            skip_special_tokens=True,
        )
        for offset in range(len(batch)):
            begin = offset * num_return_sequences
            all_outputs.append(
                [
                    text.strip()
                    for text in continuations[
                        begin : begin + num_return_sequences
                    ]
                ]
            )
        completed = min(start + len(batch), len(prompts))
        print(f"[short-rebuild] {label} {completed}/{len(prompts)}", flush=True)
    return all_outputs


def _load_teacher():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.enable_cudnn_sdp(False)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
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
    random.seed(42)
    print("[short-rebuild] disabled cuDNN SDPA; using stable PyTorch attention backend", flush=True)
    return tokenizer, model


def _assert_gpu_capacity() -> None:
    import torch

    minimum_free_gib = float(
        os.environ.get("QA_REBUILD_MIN_FREE_GPU_GIB", "48")
    )
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except RuntimeError as exc:
        raise RuntimeError(
            "shared H200 cannot initialize a CUDA context because competing "
            "processes exhausted device memory; retry unchanged after release"
        ) from exc
    gib = 1024**3
    free_gib = free_bytes / gib
    total_gib = total_bytes / gib
    print(
        f"[short-rebuild] gpu_memory free={free_gib:.2f} GiB "
        f"total={total_gib:.2f} GiB required={minimum_free_gib:.2f} GiB",
        flush=True,
    )
    if free_gib < minimum_free_gib:
        raise RuntimeError(
            "shared H200 is already occupied: "
            f"free={free_gib:.2f} GiB < required={minimum_free_gib:.2f} GiB; "
            "retry unchanged configuration after the competing process exits"
        )


def _row_id(row: Mapping[str, Any]) -> int:
    clean = row.get("_clean")
    if not isinstance(clean, Mapping) or clean.get("row_id") is None:
        raise ValueError("clean training row is missing _clean.row_id")
    return int(clean["row_id"])


def _static_label_reasons(audit: Mapping[str, Any]) -> list[str]:
    return [
        str(reason)
        for reason in audit.get("label_defect_reasons", [])
        if reason != "no_answer_bearing_evidence_mapping"
    ]


def _prepare_source_rows(
    clean_rows: Sequence[dict[str, Any]],
    audits: Sequence[dict[str, Any]],
    official_rows: Sequence[dict[str, Any]],
    *,
    max_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    audit_by_id = {
        int(row["source_row_id"]): row
        for row in audits
        if row.get("dataset_split") == "train"
    }
    official_fingerprints = {
        question_fingerprint(str(row.get("query", "")))
        for row in official_rows
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prefilter_rejections: list[dict[str, Any]] = []
    short_count = 0
    for row in clean_rows:
        question_type, _keypoints = expected_keypoints(
            str(row.get("expected_answer", ""))
        )
        if question_type != "short":
            continue
        short_count += 1
        source_row_id = _row_id(row)
        audit = audit_by_id.get(source_row_id, {})
        fingerprint = question_fingerprint(str(row["query"]))
        reason = non_text_task_reason(str(row["query"]))
        if fingerprint in official_fingerprints:
            reason = "official_validation_overlap"
        if reason:
            prefilter_rejections.append(
                {
                    "source_row_id": source_row_id,
                    "question_fingerprint": fingerprint,
                    "query": str(row["query"]),
                    "stage": "prefilter",
                    "reason": reason,
                }
            )
            continue
        grouped[fingerprint].append(
            {
                "source_row_id": source_row_id,
                "question_fingerprint": fingerprint,
                "query": str(row["query"]),
                "legacy_expected_answer": str(row["expected_answer"]),
                "bank": str(
                    row.get("meta", {}).get("bank", "")
                    if isinstance(row.get("meta"), Mapping)
                    else ""
                ),
                "legacy_audit": {
                    "support_level": str(audit.get("support_level", "unknown")),
                    "static_label_defect_reasons": _static_label_reasons(audit),
                    "all_label_defect_reasons": list(
                        audit.get("label_defect_reasons", [])
                    ),
                    "primary_attribution": audit.get("primary_attribution"),
                    "selection_status": audit.get("selection_status"),
                },
            }
        )

    support_rank = {"full": 0, "partial": 1, "none": 2, "unknown": 3}
    deduplicated = []
    for fingerprint, group in grouped.items():
        selected = min(
            group,
            key=lambda row: (
                support_rank.get(
                    str(row["legacy_audit"]["support_level"]),
                    4,
                ),
                bool(row["legacy_audit"]["static_label_defect_reasons"]),
                int(row["source_row_id"]),
            ),
        )
        deduplicated.append(selected)
        for duplicate in group:
            if duplicate is selected:
                continue
            prefilter_rejections.append(
                {
                    "source_row_id": duplicate["source_row_id"],
                    "question_fingerprint": fingerprint,
                    "query": duplicate["query"],
                    "stage": "prefilter",
                    "reason": "duplicate_source_question",
                    "kept_source_row_id": selected["source_row_id"],
                }
            )

    deduplicated.sort(
        key=lambda row: (
            support_rank.get(str(row["legacy_audit"]["support_level"]), 4),
            bool(row["legacy_audit"]["static_label_defect_reasons"]),
            int(row["source_row_id"]),
        )
    )
    if max_rows:
        for row in deduplicated[max_rows:]:
            prefilter_rejections.append(
                {
                    "source_row_id": row["source_row_id"],
                    "question_fingerprint": row["question_fingerprint"],
                    "query": row["query"],
                    "stage": "prefilter",
                    "reason": "smoke_row_limit",
                }
            )
        deduplicated = deduplicated[:max_rows]
    stats = {
        "clean_short_rows": short_count,
        "unique_text_candidates": len(grouped),
        "selected_for_teacher": len(deduplicated),
        "official_fingerprint_count": len(official_fingerprints),
    }
    return deduplicated, prefilter_rejections, stats


def _search(
    index: MarkdownBM25Index,
    *,
    model_query: str,
    original_query: str,
    bank: str,
    top_k: int,
) -> tuple[str, list[SearchResult]]:
    retrieval_query = build_retrieval_query(
        model_query,
        original_query,
        bank,
    )
    return retrieval_query, index.search(
        retrieval_query,
        top_k=top_k,
        candidate_k=50,
        quality_rerank=True,
    )


def _evidence_records(
    results: Sequence[SearchResult],
    retrieval_query: str,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    records = []
    seen = set()
    for result in results:
        if result.quality_category in {"question-only", "noise"}:
            continue
        key = (result.source, result.heading)
        if key in seen:
            continue
        seen.add(key)
        snippet = best_snippet(result.text, retrieval_query, 640)
        if len(snippet.strip()) < 24:
            continue
        records.append(
            {
                "evidence_id": f"E{len(records) + 1:02d}",
                "source": result.source,
                "heading": result.heading,
                "quality_category": result.quality_category,
                "raw_score": result.raw_score,
                "text": snippet,
            }
        )
        if len(records) >= limit:
            break
    return records


def _target_prompt(row: Mapping[str, Any]) -> str:
    evidence = [
        {
            "evidence_id": item["evidence_id"],
            "source": item["source"],
            "heading": item["heading"],
            "text": item["text"],
        }
        for item in row["evidence"]
    ]
    return (
        f"题目：\n{row['query']}\n\n"
        f"知识库：{row['bank'] or '未指定'}\n\n"
        "检索证据（仅可使用这些文本）：\n"
        + json.dumps(evidence, ensure_ascii=False, indent=2)
    )


def _verify_prompt(row: Mapping[str, Any], points: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "index": point["index"],
            "statement": point["statement"],
            "source": point["source"],
            "heading": point["heading"],
            "quote": point["quote"],
        }
        for point in points
    ]
    return (
        f"题目：\n{row['query']}\n\n"
        "候选答案及逐点证据：\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _rewrite_prompt(row: Mapping[str, Any]) -> str:
    return (
        f"题目：\n{row['query']}\n\n"
        f"首次查询：\n{row['first_search_query']}\n\n"
        f"首次可见检索结果：\n"
        f"{visible_retrieval_text(str(row['first_observation']))}\n\n"
        "请给出不同的第二次检索关键词，以补齐题目要求的其余信息。"
    )


def _result_records(
    results: Sequence[SearchResult],
    visible_snippets: Sequence[str],
) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "source": result.source,
            "heading": result.heading,
            "quality_category": result.quality_category,
            "raw_score": result.raw_score,
            "text": snippet,
        }
        for rank, (result, snippet) in enumerate(
            zip(results, visible_snippets, strict=True),
            start=1,
        )
    ]


def _runtime_messages(tokenizer, messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    if [message.get("role") for message in messages[:2]] != ["system", "user"]:
        raise ValueError("trajectory must start with system and user")
    initial = format_agent_prompt(
        tokenizer,
        str(messages[1]["content"]),
        system_prompt=str(messages[0]["content"]),
    )
    return [
        {"role": "user", "content": initial},
        *[dict(message) for message in messages[2:]],
    ]


def _build_manifest_record(
    row: Mapping[str, Any],
    tokenizer,
    *,
    max_tokens: int,
) -> tuple[dict[str, Any] | None, str]:
    try:
        points = bind_visible_evidence(
            row["answer_points"],
            row["search_hops"],
        )
    except ValueError:
        return None, "missing_visible_source_binding"
    expected = rebuilt_expected_answer(points)
    messages = build_search_messages(
        query=str(row["query"]),
        expected=expected,
        first_query=str(row["first_search_query"]),
        first_observation=str(row["first_observation"]),
        second_query=row.get("second_search_query"),
        second_observation=row.get("second_observation"),
        answer_points=[str(point["statement"]) for point in points],
    )
    issues = validate_messages(messages)
    if issues:
        return None, "message_validation:" + ",".join(issues)
    runtime_messages = _runtime_messages(tokenizer, messages)
    rendered = "".join(message["content"] for message in runtime_messages)
    token_length = len(
        tokenizer(
            rendered,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
    )
    if token_length > max_tokens:
        return None, "trajectory_too_long"
    legacy_type, legacy_keypoints = expected_keypoints(
        str(row["legacy_expected_answer"])
    )
    if legacy_type != "short":
        raise ValueError("rebuilt source unexpectedly is not short")
    legacy_text = "\n".join(
        [
            *(str(point["statement"]) for point in points),
            *(str(point["quote"]) for point in points),
        ]
    )
    legacy_hits = sorted(text_keypoint_hits(legacy_text, legacy_keypoints))
    record = {
        "source_row_id": int(row["source_row_id"]),
        "question_fingerprint": str(row["question_fingerprint"]),
        "question_type": "short",
        "query": str(row["query"]),
        "bank": str(row["bank"]),
        "legacy_expected_answer": str(row["legacy_expected_answer"]),
        "expected_answer": expected,
        "answer_points": points,
        "search_turns": int(row["search_turns"]),
        "search_hops": list(row["search_hops"]),
        "messages": runtime_messages,
        "machine_verified": True,
        "human_reviewed": False,
        "sft_v2_ready": False,
        "_audit": {
            "teacher_candidate_index": int(row["teacher_candidate_index"]),
            "deterministic_point_checks": True,
            "independent_verifier_accept": True,
            "exact_quote_coverage": 1.0,
            "incremental_two_hop": bool(row.get("incremental_two_hop", False)),
            "query_leakage_check": True,
            "official_validation_fingerprint_overlap": False,
            "legacy_keypoint_count": len(legacy_keypoints),
            "legacy_keypoint_hits": legacy_hits,
            "legacy_keypoint_coverage": (
                len(legacy_hits) / len(legacy_keypoints)
                if legacy_keypoints
                else 0.0
            ),
            "legacy_label_audit": dict(row["legacy_audit"]),
            "token_length": token_length,
            "assistant_message_count": sum(
                message["role"] == "assistant"
                for message in runtime_messages
            ),
            "environment_message_count": sum(
                message["role"] == "environment"
                for message in runtime_messages
            ),
            "runtime_raw_chunk_alignment": True,
        },
    }
    return record, "accepted"


def main() -> None:
    _, overrides = _parse_args()
    output_dir = _output_dir(overrides)
    max_rows = _env_int("QA_REBUILD_MAX_ROWS", 0)
    max_tokens = _env_int("QA_REBUILD_MAX_TOKENS", 6000)
    minimum_accepted = _env_int("QA_REBUILD_MIN_ACCEPTED", 24)
    minimum_two_hop = _env_int("QA_REBUILD_MIN_TWO_HOP", 4)
    enforce_minimum = os.environ.get("QA_REBUILD_ENFORCE_MIN", "0") == "1"
    require_splits = os.environ.get("QA_REBUILD_REQUIRE_SPLITS", "1") == "1"

    clean_path = Path(
        os.environ.get("QA_CLEAN_TRAIN_PATH", DEFAULT_CLEAN_TRAIN_PATH)
    )
    audit_dir = Path(os.environ.get("QA_SHORT_AUDIT_DIR", DEFAULT_AUDIT_DIR))
    data_dir = Path(os.environ.get("QA_RL_DATA_DIR", "/data/datasets/qa_rl"))
    docs_dir = Path(os.environ.get("QA_DOCS_DIR", "/data/docs"))
    audit_path = audit_dir / "short_gold_audit.jsonl"
    val_path = data_dir / "val.jsonl"
    required = [clean_path, audit_path, val_path, docs_dir]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing rebuild inputs: {missing}")

    clean_rows = _read_jsonl(clean_path)
    audit_rows = _read_jsonl(audit_path)
    official_rows = _read_jsonl(val_path)
    source_rows, rejected, source_stats = _prepare_source_rows(
        clean_rows,
        audit_rows,
        official_rows,
        max_rows=max_rows,
    )
    print(
        f"[short-rebuild] source candidates={len(source_rows)} "
        f"prefilter_rejected={len(rejected)}",
        flush=True,
    )

    _assert_gpu_capacity()
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
        f"[short-rebuild] indexed={index.num_documents} "
        f"seconds={index_seconds:.1f}",
        flush=True,
    )

    prepared = []
    for row in source_rows:
        first_search_query = question_context(str(row["query"]))[:256]
        retrieval_query, results = _search(
            index,
            model_query=first_search_query,
            original_query=str(row["query"]),
            bank=str(row["bank"]),
            top_k=12,
        )
        evidence = _evidence_records(results, retrieval_query)
        if not evidence:
            rejected.append(
                {
                    **row,
                    "stage": "evidence_pool",
                    "reason": "no_trusted_evidence_snippets",
                }
            )
            continue
        prepared.append(
            {
                **row,
                "initial_retrieval_query": retrieval_query,
                "evidence": evidence,
            }
        )

    _assert_gpu_capacity()
    load_start = time.perf_counter()
    tokenizer, model = _load_teacher()
    model_load_seconds = time.perf_counter() - load_start
    target_prompts = [
        _chat_prompt(tokenizer, TARGET_SYSTEM_PROMPT, _target_prompt(row))
        for row in prepared
    ]
    target_outputs = _generate(
        target_prompts,
        tokenizer,
        model,
        label="target-generation",
        batch_size=4,
        max_new_tokens=512,
        num_return_sequences=2,
        do_sample=True,
        temperature=0.4,
        top_p=0.9,
    )

    generation_audit: list[dict[str, Any]] = []
    valid_attempts: list[dict[str, Any]] = []
    for row, outputs in zip(prepared, target_outputs, strict=True):
        evidence_by_id = {
            str(item["evidence_id"]): item
            for item in row["evidence"]
        }
        for candidate_index, raw in enumerate(outputs, start=1):
            payload = extract_json_object(raw)
            if payload is None:
                points, reason = None, "invalid_json"
            else:
                points, reason = validate_generated_target(
                    payload,
                    question=str(row["query"]),
                    evidence_by_id=evidence_by_id,
                )
            audit_position = len(generation_audit)
            generation_audit.append(
                {
                    "source_row_id": row["source_row_id"],
                    "question_fingerprint": row["question_fingerprint"],
                    "candidate_index": candidate_index,
                    "query": row["query"],
                    "raw_generation": raw,
                    "deterministic_decision": reason,
                    "points": points,
                    "verifier_raw": None,
                    "verifier_accept": False,
                }
            )
            if points:
                valid_attempts.append(
                    {
                        "row": row,
                        "candidate_index": candidate_index,
                        "points": points,
                        "generation_audit_position": audit_position,
                    }
                )

    verify_prompts = [
        _chat_prompt(
            tokenizer,
            VERIFY_SYSTEM_PROMPT,
            _verify_prompt(attempt["row"], attempt["points"]),
        )
        for attempt in valid_attempts
    ]
    print(
        f"[short-rebuild] deterministic valid attempts={len(valid_attempts)}",
        flush=True,
    )
    verify_outputs = _generate(
        verify_prompts,
        tokenizer,
        model,
        label="independent-verification",
        batch_size=4,
        max_new_tokens=256,
    )
    verified_by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for attempt, outputs in zip(valid_attempts, verify_outputs, strict=True):
        raw = outputs[0]
        payload = extract_json_object(raw)
        accepted = bool(
            payload
            and verifier_accepts(
                payload,
                len(attempt["points"]),
            )
        )
        audit = generation_audit[attempt["generation_audit_position"]]
        audit["verifier_raw"] = raw
        audit["verifier_accept"] = accepted
        if accepted:
            verified_by_row[int(attempt["row"]["source_row_id"])].append(
                attempt
            )

    verified_targets = []
    for row in prepared:
        attempts = verified_by_row.get(int(row["source_row_id"]), [])
        if not attempts:
            rejected.append(
                {
                    **row,
                    "stage": "target_verification",
                    "reason": "no_independently_verified_target",
                }
            )
            continue
        selected = min(
            attempts,
            key=lambda attempt: (
                len(
                    {
                        point["evidence_id"]
                        for point in attempt["points"]
                    }
                ),
                sum(
                    len(str(point["statement"]))
                    for point in attempt["points"]
                ),
                int(attempt["candidate_index"]),
            ),
        )
        verified_targets.append(
            {
                **row,
                "answer_points": selected["points"],
                "teacher_candidate_index": selected["candidate_index"],
            }
        )

    route_pending = []
    routed_targets = []
    route_audit: list[dict[str, Any]] = []
    for row in verified_targets:
        first_query = question_context(str(row["query"]))[:256]
        retrieval_query, results = _search(
            index,
            model_query=first_query,
            original_query=str(row["query"]),
            bank=str(row["bank"]),
            top_k=4,
        )
        observation, visible_snippets = format_search_results_with_visible_snippets(
            results,
            retrieval_query,
            max_chars=1800,
            per_result_chars=360,
        )
        first_hits = trusted_visible_quote_hits(
            results,
            visible_snippets,
            row["answer_points"],
        )
        first_hop = {
            "hop": 1,
            "model_search_query": first_query,
            "retrieval_query": retrieval_query,
            "top_k_results": _result_records(results, visible_snippets),
            "observation": observation,
            "answer_point_hit_indexes": sorted(first_hits),
            "new_answer_point_hit_indexes": sorted(first_hits),
        }
        routed = {
            **row,
            "first_search_query": first_query,
            "first_observation": observation,
            "first_hits": sorted(first_hits),
            "search_hops": [first_hop],
        }
        if len(first_hits) == len(row["answer_points"]):
            routed_targets.append(
                {
                    **routed,
                    "search_turns": 1,
                    "incremental_two_hop": False,
                }
            )
            route_audit.append(
                {
                    "source_row_id": row["source_row_id"],
                    "accepted": True,
                    "search_turns": 1,
                    "first_hits": sorted(first_hits),
                    "candidates": [],
                }
            )
        else:
            route_pending.append(routed)

    rewrite_prompts = [
        _chat_prompt(tokenizer, REWRITE_SYSTEM_PROMPT, _rewrite_prompt(row))
        for row in route_pending
    ]
    rewrite_outputs = _generate(
        rewrite_prompts,
        tokenizer,
        model,
        label="query-rewrite",
        batch_size=4,
        max_new_tokens=64,
        num_return_sequences=8,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
    )
    rewrite_rejections: Counter[str] = Counter()
    for row, outputs in zip(route_pending, rewrite_outputs, strict=True):
        accepted_candidates = []
        candidate_audit = []
        seen = set()
        for raw in outputs:
            query = parse_query_candidate(raw)
            normalized = " ".join(query.lower().split())
            if normalized in seen:
                reason = "duplicate_sample"
            else:
                seen.add(normalized)
                reason = query_rejection_reason(
                    query,
                    first_query=str(row["first_search_query"]),
                    visible_context=(
                        str(row["query"])
                        + "\n"
                        + visible_retrieval_text(
                            str(row["first_observation"])
                        )
                    ),
                    keypoints=[
                        [str(point["statement"])]
                        for point in row["answer_points"]
                    ],
                )
            candidate: dict[str, Any] | None = None
            if reason is None:
                retrieval_query, results = _search(
                    index,
                    model_query=query,
                    original_query=str(row["query"]),
                    bank=str(row["bank"]),
                    top_k=4,
                )
                observation, visible_snippets = format_search_results_with_visible_snippets(
                    results,
                    retrieval_query,
                    max_chars=1800,
                    per_result_chars=360,
                )
                second_hits = trusted_visible_quote_hits(
                    results,
                    visible_snippets,
                    row["answer_points"],
                )
                first_hits = set(int(hit) for hit in row["first_hits"])
                new_hits = second_hits - first_hits
                cumulative_hits = first_hits | second_hits
                if not new_hits:
                    reason = "no_evidence_gain"
                elif len(cumulative_hits) != len(row["answer_points"]):
                    reason = "incomplete_cumulative_evidence"
                else:
                    reason = "accepted"
                    candidate = {
                        "query": query,
                        "retrieval_query": retrieval_query,
                        "results": results,
                        "observation": observation,
                        "visible_snippets": visible_snippets,
                        "second_hits": sorted(second_hits),
                        "new_hits": sorted(new_hits),
                        "cumulative_hits": sorted(cumulative_hits),
                    }
            rewrite_rejections[str(reason)] += int(reason != "accepted")
            candidate_audit.append(
                {
                    "raw": raw,
                    "query": query,
                    "decision": reason,
                    "new_hits": candidate["new_hits"] if candidate else [],
                }
            )
            if candidate:
                accepted_candidates.append(candidate)

        selected = (
            min(
                accepted_candidates,
                key=lambda candidate: (
                    -len(candidate["new_hits"]),
                    len(candidate["query"]),
                    candidate["query"],
                ),
            )
            if accepted_candidates
            else None
        )
        route_audit.append(
            {
                "source_row_id": row["source_row_id"],
                "accepted": selected is not None,
                "search_turns": 2 if selected else None,
                "first_hits": row["first_hits"],
                "selected_query": selected["query"] if selected else None,
                "candidates": candidate_audit,
            }
        )
        if selected is None:
            rejected.append(
                {
                    **row,
                    "stage": "route_verification",
                    "reason": "no_full_visible_two_hop_route",
                }
            )
            continue
        second_hop = {
            "hop": 2,
            "model_search_query": selected["query"],
            "retrieval_query": selected["retrieval_query"],
            "top_k_results": _result_records(
                selected["results"],
                selected["visible_snippets"],
            ),
            "observation": selected["observation"],
            "answer_point_hit_indexes": selected["second_hits"],
            "new_answer_point_hit_indexes": selected["new_hits"],
        }
        routed_targets.append(
            {
                **row,
                "second_search_query": selected["query"],
                "second_observation": selected["observation"],
                "search_hops": [*row["search_hops"], second_hop],
                "search_turns": 2,
                "incremental_two_hop": True,
            }
        )

    manifest = []
    for row in routed_targets:
        record, reason = _build_manifest_record(
            row,
            tokenizer,
            max_tokens=max_tokens,
        )
        if record:
            manifest.append(record)
        else:
            rejected.append(
                {
                    **row,
                    "stage": "token_protocol_audit",
                    "reason": reason,
                }
            )

    assigned = assign_group_splits(manifest, seed=42) if manifest else []
    official_fingerprints = {
        question_fingerprint(str(row["query"]))
        for row in official_rows
    }
    assigned_fingerprints = {
        str(row["question_fingerprint"])
        for row in assigned
    }
    leaked = assigned_fingerprints & official_fingerprints
    if leaked:
        raise RuntimeError(
            f"rebuilt targets overlap official validation: {len(leaked)}"
        )

    paths = {
        "manifest": output_dir / "machine_verified_targets.jsonl",
        "generation_audit": output_dir / "generation_audit.jsonl",
        "route_audit": output_dir / "route_audit.jsonl",
        "rejected": output_dir / "rejected_candidates.jsonl",
        "representatives": output_dir / "representative_examples.jsonl",
        "train": output_dir / "machine_verified_sft_train.jsonl",
        "validation": output_dir / "machine_verified_sft_validation.jsonl",
        "rl_holdout": output_dir / "curated_short_rl_holdout.jsonl",
        "summary": output_dir / "summary.json",
    }
    _write_jsonl(paths["manifest"], assigned)
    _write_jsonl(paths["generation_audit"], generation_audit)
    _write_jsonl(paths["route_audit"], route_audit)
    _write_jsonl(paths["rejected"], rejected)
    _write_jsonl(
        paths["train"],
        [
            {"messages": row["messages"]}
            for row in assigned
            if row["split"] == "train"
        ],
    )
    _write_jsonl(
        paths["validation"],
        [
            {"messages": row["messages"]}
            for row in assigned
            if row["split"] == "validation"
        ],
    )
    _write_jsonl(
        paths["rl_holdout"],
        [
            {
                "query": row["query"],
                "expected_answer": row["expected_answer"],
                "meta": {
                    "source_row_id": row["source_row_id"],
                    "question_fingerprint": row["question_fingerprint"],
                    "bank": row["bank"],
                    "answer_points": row["answer_points"],
                    "search_turns": row["search_turns"],
                },
            }
            for row in assigned
            if row["split"] == "rl_holdout"
        ],
    )

    accepted_examples = [
        {
            "kind": "machine_verified",
            "source_row_id": row["source_row_id"],
            "split": row["split"],
            "query": row["query"],
            "legacy_expected_answer": row["legacy_expected_answer"],
            "expected_answer": row["expected_answer"],
            "answer_points": row["answer_points"],
            "search_hops": row["search_hops"],
            "final_completion": row["messages"][-1]["content"],
        }
        for row in assigned[:15]
    ]
    rejection_examples = [
        {
            "kind": "rejected",
            "source_row_id": row.get("source_row_id"),
            "query": row.get("query"),
            "stage": row.get("stage"),
            "reason": row.get("reason"),
        }
        for row in sorted(
            rejected,
            key=lambda row: (
                str(row.get("stage")),
                str(row.get("reason")),
                int(row.get("source_row_id", -1)),
            ),
        )[:5]
    ]
    representatives = [*accepted_examples, *rejection_examples][:20]
    _write_jsonl(paths["representatives"], representatives)

    split_counts = Counter(str(row["split"]) for row in assigned)
    point_counts = Counter(len(row["answer_points"]) for row in assigned)
    hop_counts = Counter(int(row["search_turns"]) for row in assigned)
    rejection_counts = Counter(
        f"{row.get('stage')}:{row.get('reason')}"
        for row in rejected
    )
    token_lengths = [
        int(row["_audit"]["token_length"])
        for row in assigned
    ]
    two_hop_count = hop_counts.get(2, 0)
    gate_passed = (
        len(assigned) >= minimum_accepted
        and two_hop_count >= minimum_two_hop
        and (
            not require_splits
            or (
                split_counts.get("validation", 0) > 0
                and split_counts.get("rl_holdout", 0) > 0
            )
        )
    )
    summary = {
        "mode": "smoke" if max_rows else "full",
        "teacher_model": MODEL_NAME,
        "inputs": {
            "clean_train": str(clean_path),
            "short_audit": str(audit_path),
            "official_validation": str(val_path),
            "docs": str(docs_dir),
        },
        "source": source_stats,
        "evidence_pool_ready": len(prepared),
        "deterministically_valid_attempts": len(valid_attempts),
        "independently_verified_unique_targets": len(verified_targets),
        "machine_verified_route_targets": len(assigned),
        "point_count_distribution": dict(sorted(point_counts.items())),
        "search_turn_counts": dict(sorted(hop_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "rewrite_rejection_counts": dict(sorted(rewrite_rejections.items())),
        "quality_category_counts": index.quality_category_counts,
        "official_validation_overlap_count": 0,
        "token_lengths": {
            "min": min(token_lengths) if token_lengths else None,
            "mean": mean(token_lengths) if token_lengths else None,
            "max": max(token_lengths) if token_lengths else None,
        },
        "training_gate": {
            "minimum_unique_targets": minimum_accepted,
            "minimum_two_hop_targets": minimum_two_hop,
            "requires_validation_and_rl_splits": require_splits,
            "passed_machine_gate": gate_passed,
            "human_review_required": True,
            "sft_v2_ready_count": 0,
        },
        "timing_seconds": {
            "index_build": index_seconds,
            "model_load": model_load_seconds,
        },
        "outputs": {
            name: str(path)
            for name, path in paths.items()
        },
    }
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[short-rebuild] summary", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("[short-rebuild] human-review-samples", flush=True)
    print(
        json.dumps(accepted_examples[:3], ensure_ascii=False, indent=2),
        flush=True,
    )

    del model, tokenizer
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass

    if enforce_minimum and not gate_passed:
        raise RuntimeError(
            "machine rebuild gate failed: "
            f"accepted={len(assigned)}, two_hop={two_hop_count}"
        )


if __name__ == "__main__":
    main()
