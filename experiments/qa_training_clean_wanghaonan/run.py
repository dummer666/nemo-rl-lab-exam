#!/usr/bin/env python
"""Create a non-destructive cleaned QA training manifest on the server."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.retrieval.evidence import (  # noqa: E402
    evidence_coverage,
    expected_keypoints,
    normalize_evidence_text,
)
from common.retrieval.markdown_bm25 import (  # noqa: E402
    MarkdownBM25Index,
    build_retrieval_query,
    question_context,
)

_EXPECTED = re.compile(r"^\s*\[(\w+)\]\s*(.*)", re.DOTALL)
_OPTION = re.compile(r"^\s*([A-Z])\.\s*(.+?)\s*$", re.MULTILINE)
_BLANK = re.compile(r"【(\d+)】")
_VALID_TYPES = {"single", "multiple", "bool", "fill", "short"}


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Clean QA training data")
    parser.add_argument("--config")
    return parser.parse_known_args()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def _question_key(query: str, bank: str = "") -> str:
    options = _OPTION.findall(query)
    option_text = "|".join(f"{letter}:{text}" for letter, text in options)
    return normalize_evidence_text(f"{bank}|{question_context(query)}|{option_text}")


def _analyze_structure(row: dict, row_id: int) -> dict:
    query = str(row.get("query", "")).strip()
    expected = str(row.get("expected_answer", "")).strip()
    issues: list[str] = []
    fatal_issues: list[str] = []
    match = _EXPECTED.match(expected)
    question_type = match.group(1).lower() if match else "unknown"
    answer = match.group(2).strip() if match else ""
    options = dict(_OPTION.findall(query))
    metadata = row.get("meta") if isinstance(row.get("meta"), dict) else {}

    if not query:
        fatal_issues.append("missing_query")
    if not expected:
        fatal_issues.append("missing_expected_answer")
    if question_type not in _VALID_TYPES:
        fatal_issues.append("invalid_answer_type")

    if question_type in {"single", "bool", "multiple"}:
        letters = set(re.findall(r"[A-Z]", answer.upper()))
        if not options:
            fatal_issues.append("missing_options")
        if not letters:
            fatal_issues.append("missing_answer_letters")
        if question_type in {"single", "bool"} and len(letters) != 1:
            fatal_issues.append("non_unique_objective_answer")
        if options and not letters.issubset(options):
            fatal_issues.append("answer_letter_out_of_range")

    keypoint_type, keypoints = expected_keypoints(expected)
    if question_type == "fill":
        if not keypoints:
            fatal_issues.append("empty_fill_answer")
        blank_count = len(set(_BLANK.findall(query)))
        if blank_count and blank_count != len(keypoints):
            issues.append("fill_blank_count_mismatch")
    elif question_type == "short" and not keypoints:
        fatal_issues.append("empty_short_keypoints")

    return {
        "row_id": row_id,
        "type": question_type,
        "question_key": _question_key(query, str(metadata.get("bank", ""))),
        "expected_key": normalize_evidence_text(expected),
        "issues": issues,
        "fatal_issues": fatal_issues,
        "duplicate_group_size": 1,
        "duplicate_of": None,
        "evidence_coverage": None,
        "support_level": "not_applicable",
        "evidence_sources": [],
        "sample_weight": 0.0,
        "_keypoints": keypoints if keypoint_type in {"fill", "short"} else [],
    }


def _apply_duplicate_policy(records: list[dict]) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["question_key"]:
            groups[record["question_key"]].append(record)

    duplicate_group = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        duplicate_group += 1
        expected_keys = {record["expected_key"] for record in group}
        for record in group:
            record["duplicate_group"] = duplicate_group
            record["duplicate_group_size"] = len(group)
        if len(expected_keys) > 1:
            for record in group:
                record["fatal_issues"].append("duplicate_answer_conflict")
            continue
        canonical = min(group, key=lambda record: record["row_id"])
        for record in group:
            if record is not canonical:
                record["duplicate_of"] = canonical["row_id"]
                record["issues"].append("exact_duplicate")


def _sample_weight(record: dict) -> float:
    if record["fatal_issues"] or record["duplicate_of"] is not None:
        return 0.0
    if record["type"] in {"fill", "short"}:
        return {
            "full": 3.0,
            "partial": 2.0,
            "none": 0.25,
        }.get(record["support_level"], 0.0)
    return 1.0


def _output_dir(overrides: Sequence[str]) -> Path:
    for override in overrides:
        if override.startswith("logger.log_dir="):
            output = Path(override.split("=", 1)[1]).parent / "cleaned_data"
            output.mkdir(parents=True, exist_ok=True)
            return output
    output = THIS_DIR / "cleaned_data"
    output.mkdir(parents=True, exist_ok=True)
    return output


def main() -> None:
    _, overrides = _parse_args()
    data_dir = Path(os.environ.get("QA_RL_DATA_DIR", "/data/datasets/qa_rl"))
    docs_dir = Path(os.environ.get("QA_DOCS_DIR", "/data/docs"))
    rows = _read_jsonl(data_dir / "train.jsonl")
    records = [_analyze_structure(row, row_id) for row_id, row in enumerate(rows)]
    _apply_duplicate_policy(records)

    index_start = time.perf_counter()
    index = MarkdownBM25Index.from_directory(
        docs_dir,
        chunk_chars=1200,
        overlap_chars=160,
        k1=1.5,
        b=0.75,
    )
    index_seconds = time.perf_counter() - index_start

    evidence_cache: dict[str, tuple[float, list[str]]] = {}
    evidence_queries = 0
    evidence_start = time.perf_counter()
    for record, row in zip(records, rows, strict=True):
        if record["type"] not in {"fill", "short"} or record["fatal_issues"]:
            continue
        key = record["question_key"]
        if key not in evidence_cache:
            query = str(row["query"])
            metadata = row.get("meta") if isinstance(row.get("meta"), dict) else {}
            retrieval_query = build_retrieval_query(
                question_context(query),
                query,
                str(metadata.get("bank", "")),
            )
            results = index.search(
                retrieval_query,
                top_k=20,
                candidate_k=50,
                quality_rerank=True,
            )
            coverage = evidence_coverage(results, record["_keypoints"], top_k=20)
            sources = [result.source for result in results[:5]]
            evidence_cache[key] = (coverage, sources)
            evidence_queries += 1
            if evidence_queries % 100 == 0:
                print(f"[training-clean] evidence queries {evidence_queries}")
        coverage, sources = evidence_cache[key]
        record["evidence_coverage"] = coverage
        record["support_level"] = "full" if coverage >= 1.0 else "partial" if coverage > 0.0 else "none"
        record["evidence_sources"] = sources

    evidence_seconds = time.perf_counter() - evidence_start
    for record in records:
        record["sample_weight"] = _sample_weight(record)

    output_dir = _output_dir(overrides)
    manifest_path = output_dir / "clean_manifest.jsonl"
    clean_path = output_dir / "clean_train.jsonl"
    summary_path = output_dir / "summary.json"

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            public_record = {key: value for key, value in record.items() if not key.startswith("_")}
            handle.write(json.dumps(public_record, ensure_ascii=False) + "\n")

    clean_rows = 0
    with clean_path.open("w", encoding="utf-8") as handle:
        for row, record in zip(rows, records, strict=True):
            if record["sample_weight"] <= 0:
                continue
            clean_row = dict(row)
            clean_row["_clean"] = {
                "row_id": record["row_id"],
                "support_level": record["support_level"],
                "evidence_coverage": record["evidence_coverage"],
                "sample_weight": record["sample_weight"],
            }
            handle.write(json.dumps(clean_row, ensure_ascii=False) + "\n")
            clean_rows += 1

    issue_counts = Counter(
        issue
        for record in records
        for issue in [*record["issues"], *record["fatal_issues"]]
    )
    type_input = Counter(record["type"] for record in records)
    type_kept = Counter(record["type"] for record in records if record["sample_weight"] > 0)
    support_counts = Counter(
        f"{record['type']}:{record['support_level']}"
        for record in records
        if record["type"] in {"fill", "short"} and not record["fatal_issues"]
    )
    summary = {
        "input_rows": len(rows),
        "clean_rows": clean_rows,
        "excluded_rows": len(rows) - clean_rows,
        "effective_weight_sum": sum(record["sample_weight"] for record in records),
        "type_counts_input": dict(sorted(type_input.items())),
        "type_counts_kept": dict(sorted(type_kept.items())),
        "support_counts": dict(sorted(support_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "exact_duplicate_rows": sum(record["duplicate_of"] is not None for record in records),
        "conflicting_rows": sum("duplicate_answer_conflict" in record["fatal_issues"] for record in records),
        "fatal_rows": sum(bool(record["fatal_issues"]) for record in records),
        "evidence_queries": evidence_queries,
        "quality_category_counts": index.quality_category_counts,
        "timing_seconds": {
            "index_build": index_seconds,
            "evidence_search": evidence_seconds,
            "evidence_per_query": evidence_seconds / max(1, evidence_queries),
        },
        "outputs": {
            "manifest": str(manifest_path),
            "clean_train": str(clean_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[training-clean] summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
