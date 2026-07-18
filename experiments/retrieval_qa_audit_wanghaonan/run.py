#!/usr/bin/env python
"""Audit lexical, quality-aware, semantic, and hybrid retrieval on server data."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Sequence

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.retrieval.markdown_bm25 import (  # noqa: E402
    MarkdownBM25Index,
    SearchResult,
    build_retrieval_query,
    question_context,
)
from common.retrieval.semantic_reranker import (  # noqa: E402
    TransformerSemanticReranker,
    reciprocal_rank_fusion,
    rerank_by_semantic,
)

_EXPECTED = re.compile(r"^\s*\[(\w+)\]\s*(.*)", re.DOTALL)
_PUNCT = re.compile(r"[\s，,。．.、；;：:！!？?\"'`（）()【】\[\]{}<>]+")


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="QA retrieval audit")
    parser.add_argument("--config")
    return parser.parse_known_args()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    normalized = normalized.replace("μ", "u").replace("µ", "u")
    return _PUNCT.sub("", normalized)


def _gold_keypoints(expected: str) -> tuple[str, list[list[str]]]:
    match = _EXPECTED.match(str(expected))
    if not match:
        return "unknown", []
    question_type, answer = match.group(1).lower(), match.group(2)
    if question_type not in {"fill", "short"}:
        return question_type, []
    keypoints = []
    for raw_point in answer.split("|||"):
        parts = re.split(r"[/／]", raw_point)
        alternatives = {_normalize(part) for part in parts}
        if len(parts) == 1:
            alternatives.add(_normalize(raw_point))
        alternatives.discard("")
        if alternatives:
            keypoints.append(sorted(alternatives, key=len, reverse=True))
    return question_type, keypoints


def _evidence_coverage(
    results: Sequence[SearchResult],
    keypoints: Sequence[Sequence[str]],
    *,
    top_k: int,
) -> float:
    if not keypoints:
        return 0.0
    searchable = [
        result
        for result in results[:top_k]
        if result.quality_category not in {"question-only", "noise"}
    ]
    evidence = _normalize("\n".join(result.text for result in searchable))
    hits = sum(any(alternative in evidence for alternative in alternatives) for alternatives in keypoints)
    return hits / len(keypoints)


def _top1_question_only(results: Sequence[SearchResult]) -> int:
    return int(bool(results) and results[0].quality_category == "question-only")


def _output_path(overrides: Sequence[str]) -> Path:
    for override in overrides:
        if override.startswith("logger.log_dir="):
            directory = Path(override.split("=", 1)[1])
            directory.mkdir(parents=True, exist_ok=True)
            return directory / "retrieval_audit.json"
    return THIS_DIR / "retrieval_audit.json"


def _cached_huggingface_models() -> list[str]:
    roots = [Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"]
    roots.extend(
        Path(value)
        for name in ("HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")
        if (value := os.environ.get(name))
    )
    models = set()
    for root in roots:
        if not str(root) or not root.is_dir():
            continue
        for path in root.glob("models--*"):
            models.add(path.name.removeprefix("models--").replace("--", "/"))
    return sorted(models)


def main() -> None:
    _, overrides = _parse_args()
    docs_dir = Path(os.environ.get("QA_DOCS_DIR", "/data/docs"))
    data_dir = Path(os.environ.get("QA_RL_DATA_DIR", "/data/datasets/qa_rl"))
    candidate_k = int(os.environ.get("QA_AUDIT_CANDIDATE_K", "50"))
    max_rows = int(os.environ.get("QA_AUDIT_MAX_ROWS", "0"))
    semantic_enabled = os.environ.get("QA_AUDIT_SEMANTIC", "1") != "0"
    semantic_local_only = os.environ.get("QA_SEMANTIC_LOCAL_ONLY", "1") != "0"
    semantic_model = os.environ.get(
        "QA_SEMANTIC_MODEL",
        "intfloat/multilingual-e5-small",
    )
    semantic_batch_size = int(os.environ.get("QA_SEMANTIC_BATCH_SIZE", "64"))
    semantic_max_length = int(os.environ.get("QA_SEMANTIC_MAX_LENGTH", "512"))
    semantic_query_prefix = os.environ.get("QA_SEMANTIC_QUERY_PREFIX", "query: ")
    semantic_passage_prefix = os.environ.get("QA_SEMANTIC_PASSAGE_PREFIX", "passage: ")

    index_start = time.perf_counter()
    index = MarkdownBM25Index.from_directory(
        docs_dir,
        chunk_chars=1200,
        overlap_chars=160,
        k1=1.5,
        b=0.75,
    )
    index_seconds = time.perf_counter() - index_start
    rows = _read_jsonl(data_dir / "val.jsonl")
    if max_rows > 0:
        rows = rows[:max_rows]

    model_load_seconds = 0.0
    reranker = None
    semantic_error = None
    cached_models = _cached_huggingface_models()
    if semantic_enabled:
        model_start = time.perf_counter()
        try:
            reranker = TransformerSemanticReranker(
                semantic_model,
                device="auto",
                batch_size=semantic_batch_size,
                max_length=semantic_max_length,
                query_prefix=semantic_query_prefix,
                passage_prefix=semantic_passage_prefix,
                local_files_only=semantic_local_only,
            )
        except OSError as exc:
            semantic_error = f"{type(exc).__name__}: {exc}"
            print(f"[retrieval-audit] semantic model unavailable: {semantic_error}")
        model_load_seconds = time.perf_counter() - model_start

    stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    type_counts: dict[str, int] = defaultdict(int)
    improvements = []
    semantic_seconds = 0.0
    open_rows = 0

    for row_index, row in enumerate(rows, start=1):
        query = str(row["query"])
        expected = str(row["expected_answer"])
        metadata = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        retrieval_query = build_retrieval_query(
            question_context(query),
            query,
            str(metadata.get("bank", "")),
        )
        candidates = index.search(retrieval_query, top_k=candidate_k)
        quality = index.search(
            retrieval_query,
            top_k=candidate_k,
            candidate_k=candidate_k,
            quality_rerank=True,
        )
        methods: dict[str, Sequence[SearchResult]] = {
            "bm25": candidates,
            "quality": quality,
        }
        for method_name, results in methods.items():
            stats[method_name]["evaluated_rows"] += 1
            stats[method_name]["top1_question_only"] += _top1_question_only(results)

        question_type, keypoints = _gold_keypoints(expected)
        type_counts[question_type] += 1
        if keypoints:
            open_rows += 1
            if reranker is not None:
                semantic_start = time.perf_counter()
                semantic_scores = reranker.score(retrieval_query, candidates)
                semantic_seconds += time.perf_counter() - semantic_start
                methods["semantic"] = rerank_by_semantic(candidates, semantic_scores)
                methods["hybrid"] = reciprocal_rank_fusion(candidates, semantic_scores)
                for method_name in ("semantic", "hybrid"):
                    stats[method_name]["evaluated_rows"] += 1
                    stats[method_name]["top1_question_only"] += _top1_question_only(methods[method_name])

            coverages = {}
            for method_name, results in methods.items():
                stats[method_name]["open_rows"] += 1
                stats[method_name]["top1_question_only_open"] += _top1_question_only(results)
                for top_k in (3, 20):
                    coverage = _evidence_coverage(results, keypoints, top_k=top_k)
                    stats[method_name][f"coverage_at_{top_k}"] += coverage
                    stats[method_name][f"full_coverage_at_{top_k}"] += int(coverage >= 1.0)
                    if top_k == 3:
                        coverages[method_name] = coverage
            if coverages.get("hybrid", 0.0) > coverages["bm25"] and len(improvements) < 20:
                improvements.append(
                    {
                        "row": row_index,
                        "type": question_type,
                        "question": question_context(query)[:240],
                        "bm25_coverage_at_3": coverages["bm25"],
                        "hybrid_coverage_at_3": coverages["hybrid"],
                        "bm25_sources": [result.source for result in candidates[:3]],
                        "hybrid_sources": [result.source for result in methods["hybrid"][:3]],
                    }
                )

        if row_index % 25 == 0 or row_index == len(rows):
            print(f"[retrieval-audit] processed {row_index}/{len(rows)} rows")

    method_summary = {}
    for method_name, values in stats.items():
        evaluated_rows = int(values.get("evaluated_rows", 0))
        method_open_rows = int(values.get("open_rows", 0))
        method_summary[method_name] = {
            "evaluated_rows": evaluated_rows,
            "top1_question_only_rate_evaluated": values["top1_question_only"] / max(1, evaluated_rows),
            "top1_question_only_rate_open": values["top1_question_only_open"] / max(1, method_open_rows),
            "evidence_coverage_at_3": values["coverage_at_3"] / max(1, method_open_rows),
            "evidence_coverage_at_20": values["coverage_at_20"] / max(1, method_open_rows),
            "full_evidence_rate_at_3": values["full_coverage_at_3"] / max(1, method_open_rows),
            "full_evidence_rate_at_20": values["full_coverage_at_20"] / max(1, method_open_rows),
        }

    report = {
        "docs_dir": str(docs_dir),
        "data_dir": str(data_dir),
        "num_chunks": index.num_documents,
        "quality_category_counts": index.quality_category_counts,
        "validation_rows": len(rows),
        "open_answer_rows": open_rows,
        "question_type_counts": dict(sorted(type_counts.items())),
        "candidate_k": candidate_k,
        "semantic_requested": semantic_enabled,
        "semantic_available": reranker is not None,
        "semantic_model": semantic_model if semantic_enabled else None,
        "semantic_batch_size": semantic_batch_size,
        "semantic_max_length": semantic_max_length,
        "semantic_error": semantic_error,
        "cached_huggingface_models": cached_models,
        "timing_seconds": {
            "index_build": index_seconds,
            "semantic_model_load": model_load_seconds,
            "semantic_queries": semantic_seconds,
            "semantic_per_open_query": semantic_seconds / max(1, open_rows),
        },
        "methods": method_summary,
        "hybrid_improvements": improvements,
    }
    output_path = _output_path(overrides)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[retrieval-audit] report")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[retrieval-audit] saved to {output_path}")


if __name__ == "__main__":
    main()
