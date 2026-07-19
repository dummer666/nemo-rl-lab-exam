#!/usr/bin/env python
"""Audit no-answer second-hop retrieval variants before any new training."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
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
    fragile_keypoint_indexes,
    normalize_evidence_text,
    visible_evidence_keypoint_hits,
)
from common.retrieval.markdown_bm25 import (  # noqa: E402
    MarkdownBM25Index,
    build_retrieval_query,
    format_search_results_with_visible_snippets,
    question_context,
)

BASELINE_TRAJECTORIES = Path(
    "/shared/outputs/wanghaonan/qa_fill_sft_multiturn_eval_wanghaonan/"
    "qa_fill_sft_multiturn_eval_wanghaonan-wanghaonan-20260719-033956/"
    "sft_multiturn_eval/step_50/trajectories.jsonl"
)
DOCS_ROOT = Path("/data/docs")
EXPECTED_FILL_ROWS = 16
_BLANK = re.compile(r"【\s*(\d+)\s*】|_{2,}")
_BOUNDARY = re.compile(r"[\n。！？!?；;]")
_ACRONYM = re.compile(
    r"(?<![A-Za-z0-9])[A-Z][A-Z0-9+./-]{1,15}(?![A-Za-z0-9])"
)
_SPACE = re.compile(r"\s+")


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def _output_dir(overrides: Sequence[str]) -> Path:
    for override in overrides:
        if override.startswith("logger.log_dir="):
            output = Path(override.split("=", 1)[1]).parent / "retrieval_oracle"
            output.mkdir(parents=True, exist_ok=True)
            return output
    output = THIS_DIR / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _deduplicate(values: Sequence[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        compact = _SPACE.sub(" ", str(value)).strip(" ，,。；;：:")
        normalized = normalize_evidence_text(compact)
        if compact and normalized and normalized not in seen:
            seen.add(normalized)
            result.append(compact[:256])
    return result


def blank_context_queries(query: str) -> list[str]:
    """Create answer-free query variants from each numbered blank's local clause."""
    question = question_context(query)
    matches = list(_BLANK.finditer(question))
    variants = []
    for match in matches:
        left_boundaries = [
            boundary.end()
            for boundary in _BOUNDARY.finditer(question[: match.start()])
        ]
        right_match = _BOUNDARY.search(question, match.end())
        start = left_boundaries[-1] if left_boundaries else max(0, match.start() - 80)
        end = right_match.start() if right_match else min(len(question), match.end() + 100)
        clause = question[start:end].strip()
        masked = _BLANK.sub("待填", clause)
        variants.append(masked)

        before = question[max(0, match.start() - 80) : match.start()]
        acronyms = _ACRONYM.findall(before)
        if acronyms:
            variants.append(f"{acronyms[-1]} 定义 全称 含义")
        if any(token in clause for token in ("多少", "数量", "超过", "低于", "不少于", "不超过")):
            variants.append(f"{masked} 阈值 数量")

    variants.append(_BLANK.sub("待填", question))
    return _deduplicate(variants)


def _query_leaks_new_answer(
    variant: str,
    original_query: str,
    keypoints: Sequence[Sequence[str]],
) -> bool:
    normalized_variant = normalize_evidence_text(variant)
    normalized_original = normalize_evidence_text(original_query)
    return any(
        alternative
        and alternative in normalized_variant
        and alternative not in normalized_original
        for alternatives in keypoints
        for alternative in alternatives
    )


def _search(
    index: MarkdownBM25Index,
    *,
    model_query: str,
    original_query: str,
    bank: str,
    keypoints: Sequence[Sequence[str]],
    exclude_sources: set[str] | None = None,
) -> dict[str, Any]:
    retrieval_query = build_retrieval_query(model_query, original_query, bank)
    results = index.search(
        retrieval_query,
        top_k=4,
        candidate_k=50,
        quality_rerank=True,
        exclude_sources=exclude_sources,
    )
    rendered, snippets = format_search_results_with_visible_snippets(
        results,
        retrieval_query,
        max_chars=1800,
        per_result_chars=360,
    )
    raw_hits = visible_evidence_keypoint_hits(results, snippets, keypoints)
    fragile = fragile_keypoint_indexes(keypoints)
    return {
        "model_query": model_query,
        "retrieval_query": retrieval_query,
        "sources": [result.source for result in results],
        "quality_categories": [result.quality_category for result in results],
        "raw_hits": sorted(raw_hits),
        "robust_hits": sorted(set(raw_hits) - fragile),
        "rendered": rendered,
    }


def _coverage(hit_count: int, keypoint_count: int) -> float | None:
    return hit_count / keypoint_count if keypoint_count else None


def _audit_row(
    index: MarkdownBM25Index,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    query = str(row["query"])
    expected = str(row["expected_answer"])
    question_type, keypoints = expected_keypoints(expected)
    if question_type != "fill" or not keypoints:
        raise ValueError("oracle row must be a fill question with keypoints")
    search_queries = [str(value) for value in row.get("search_queries", [])]
    first_query = search_queries[0] if search_queries else question_context(query)
    bank = str(row.get("bank", ""))
    fragile = fragile_keypoint_indexes(keypoints)
    robust_count = len(keypoints) - len(fragile)
    first = _search(
        index,
        model_query=first_query,
        original_query=query,
        bank=bank,
        keypoints=keypoints,
    )
    first_sources = set(first["sources"])
    first_raw = set(first["raw_hits"])
    first_robust = set(first["robust_hits"])

    candidates = [
        ("source_diverse_same_query", first_query),
        *[
            (f"blank_context_{index + 1}", variant)
            for index, variant in enumerate(blank_context_queries(query))
        ],
    ]
    variants = []
    for strategy, variant in candidates:
        leaked = _query_leaks_new_answer(variant, query, keypoints)
        second = _search(
            index,
            model_query=variant,
            original_query=query,
            bank=bank,
            keypoints=keypoints,
            exclude_sources=first_sources,
        )
        raw_hits = set(second["raw_hits"])
        robust_hits = set(second["robust_hits"])
        variants.append(
            {
                "strategy": strategy,
                **second,
                "answer_leak": leaked,
                "raw_incremental_hits": sorted(raw_hits - first_raw),
                "robust_incremental_hits": sorted(robust_hits - first_robust),
                "raw_cumulative_hits": sorted(first_raw | raw_hits),
                "robust_cumulative_hits": sorted(first_robust | robust_hits),
            }
        )
    if any(variant["answer_leak"] for variant in variants):
        raise RuntimeError(f"row {row['row_index']}: generated query leaks an answer")

    best = max(
        variants,
        key=lambda variant: (
            len(variant["robust_incremental_hits"]),
            len(variant["raw_incremental_hits"]),
            len(variant["robust_cumulative_hits"]),
            -len(str(variant["model_query"])),
        ),
    )
    return {
        "row_index": int(row["row_index"]),
        "query": query,
        "expected_answer": expected,
        "keypoint_count": len(keypoints),
        "fragile_keypoint_indexes": sorted(fragile),
        "robust_keypoint_count": robust_count,
        "first": {
            **first,
            "raw_coverage": _coverage(len(first_raw), len(keypoints)),
            "robust_coverage": _coverage(len(first_robust), robust_count),
        },
        "variants": variants,
        "best": best,
        "gains": {
            "raw_incremental": len(best["raw_incremental_hits"]),
            "robust_incremental": len(best["robust_incremental_hits"]),
            "raw_full_after": len(best["raw_cumulative_hits"]) == len(keypoints),
            "robust_full_after": bool(
                robust_count
                and len(best["robust_cumulative_hits"]) == robust_count
            ),
            "raw_full_gain": (
                len(first_raw) < len(keypoints)
                and len(best["raw_cumulative_hits"]) == len(keypoints)
            ),
            "robust_full_gain": bool(
                robust_count
                and len(first_robust) < robust_count
                and len(best["robust_cumulative_hits"]) == robust_count
            ),
        },
    }


def _metric_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    robust_records = [
        record for record in records
        if int(record["robust_keypoint_count"]) > 0
    ]
    strategy_counts = Counter(str(record["best"]["strategy"]) for record in records)
    baseline_robust = [
        float(record["first"]["robust_coverage"])
        for record in robust_records
        if record["first"]["robust_coverage"] is not None
    ]
    proposed_robust = [
        len(record["best"]["robust_cumulative_hits"])
        / int(record["robust_keypoint_count"])
        for record in robust_records
    ]
    return {
        "fill_row_count": len(records),
        "robust_evaluable_count": len(robust_records),
        "fragile_keypoint_row_count": sum(
            bool(record["fragile_keypoint_indexes"]) for record in records
        ),
        "baseline_mean_robust_coverage": mean(baseline_robust) if baseline_robust else 0.0,
        "proposed_mean_robust_coverage": mean(proposed_robust) if proposed_robust else 0.0,
        "robust_incremental_row_count": sum(
            int(record["gains"]["robust_incremental"]) > 0
            for record in records
        ),
        "raw_incremental_row_count": sum(
            int(record["gains"]["raw_incremental"]) > 0
            for record in records
        ),
        "robust_full_gain_row_count": sum(
            bool(record["gains"]["robust_full_gain"]) for record in records
        ),
        "raw_full_gain_row_count": sum(
            bool(record["gains"]["raw_full_gain"]) for record in records
        ),
        "best_strategy_counts": dict(sorted(strategy_counts.items())),
    }


def main() -> None:
    _, overrides = _parse_args()
    output_dir = _output_dir(overrides)
    missing = [
        str(path)
        for path in (BASELINE_TRAJECTORIES, DOCS_ROOT)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing oracle inputs: {missing}")
    rows = [
        row
        for row in _read_jsonl(BASELINE_TRAJECTORIES)
        if str(row.get("question_type")) == "fill"
    ]
    if len(rows) != EXPECTED_FILL_ROWS:
        raise RuntimeError(f"expected {EXPECTED_FILL_ROWS} fill rows, got {len(rows)}")

    index_start = time.perf_counter()
    index = MarkdownBM25Index.from_directory(
        DOCS_ROOT,
        chunk_chars=1200,
        overlap_chars=160,
        k1=1.5,
        b=0.75,
    )
    index_seconds = time.perf_counter() - index_start
    records = [_audit_row(index, row) for row in rows]
    metrics = _metric_summary(records)
    gate = {
        "minimum_robust_incremental_rows": 2,
        "minimum_robust_full_gain_rows": 1,
        "passed": (
            metrics["robust_incremental_row_count"] >= 2
            and metrics["robust_full_gain_row_count"] >= 1
        ),
    }
    summary = {
        "mode": "read_only_answer_free_diverse_retrieval_oracle",
        "sources": {
            "baseline_trajectories": str(BASELINE_TRAJECTORIES),
            "docs_root": str(DOCS_ROOT),
        },
        "index": {
            "num_documents": index.num_documents,
            "build_seconds": index_seconds,
            "quality_category_counts": index.quality_category_counts,
        },
        "metrics": metrics,
        "machine_gate": gate,
        "training_submitted": False,
        "outputs": {
            "details": str(output_dir / "details.jsonl"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    _write_jsonl(output_dir / "details.jsonl", records)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("[fill-retrieval-oracle]", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
