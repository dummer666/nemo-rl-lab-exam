#!/usr/bin/env python
"""Print and machine-audit the fixed objective SFT review sample."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PACK_ROOT = Path(
    "/shared/outputs/wanghaonan/qa_objective_sft_data_wanghaonan/"
    "qa_objective_sft_data_wanghaonan-wanghaonan-20260719-103532/"
    "objective_sft_data"
)
REVIEW_PATH = PACK_ROOT / "review_sample.jsonl"
MANIFEST_PATHS = (
    PACK_ROOT / "objective_train_manifest.jsonl",
    PACK_ROOT / "objective_validation_manifest.jsonl",
)
_OPTION = re.compile(r"^\s*([A-Z])\.\s*(.+?)\s*$", re.MULTILINE)
_ANSWER = re.compile(r"^\s*\[(single|multiple|bool)\]\s*(.+?)\s*$")
_LETTER = re.compile(r"[A-Z]")


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def _output_dir(overrides: Sequence[str]) -> Path:
    for override in overrides:
        if override.startswith("logger.log_dir="):
            output = (
                Path(override.split("=", 1)[1]).parent
                / "objective_sft_review"
            )
            output.mkdir(parents=True, exist_ok=True)
            return output
    output = Path(__file__).resolve().parent / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("review row must be an object")
                rows.append(value)
    return rows


def review_issues(row: Mapping[str, Any]) -> list[str]:
    issues = []
    query = str(row.get("query", ""))
    expected = str(row.get("expected_answer", ""))
    match = _ANSWER.match(expected)
    if not match:
        return ["invalid_expected_answer"]
    question_type, answer = match.groups()
    options = dict(_OPTION.findall(query))
    letters = set(_LETTER.findall(answer.upper()))
    if not options:
        issues.append("missing_options")
    if not letters:
        issues.append("missing_answer_letters")
    if options and not letters.issubset(options):
        issues.append("answer_out_of_range")
    if question_type in {"single", "bool"} and len(letters) != 1:
        issues.append("non_unique_answer")

    messages = row.get("messages")
    if not isinstance(messages, list):
        issues.append("missing_messages")
        return issues
    roles = [message.get("role") for message in messages]
    if roles != ["user", "assistant"]:
        issues.append("invalid_runtime_roles")
    final = str(messages[-1].get("content", "")) if messages else ""
    if "\\boxed{" not in final:
        issues.append("missing_boxed_answer")
    if answer.strip() not in final:
        issues.append("boxed_answer_mismatch")
    if row.get("_audit", {}).get("official_validation_overlap") is not False:
        issues.append("official_overlap_not_cleared")
    return issues


def target_categories(row: Mapping[str, Any]) -> list[str]:
    match = _ANSWER.match(str(row.get("expected_answer", "")))
    if not match:
        return []
    question_type, answer = match.groups()
    answer_letters = set(_LETTER.findall(answer.upper()))
    option_letters = set(_OPTION.findall(str(row.get("query", ""))))
    option_keys = {letter for letter, _text in option_letters}
    categories = []
    if answer_letters.intersection("EFGH"):
        categories.append("rare_answer_letter")
    if question_type == "multiple" and len(answer_letters) == 1:
        categories.append("multiple_single_answer")
    if len(option_keys) >= 6:
        categories.append("six_or_more_options")
    if (
        question_type == "multiple"
        and len(option_keys) >= 4
        and answer_letters == option_keys
    ):
        categories.append("all_options_selected")
    return categories


def targeted_review_rows(
    rows: Sequence[dict[str, Any]], per_category: int = 8
) -> tuple[list[dict[str, Any]], Counter[str]]:
    category_counts = Counter(
        category for row in rows for category in target_categories(row)
    )
    selected: dict[str, dict[str, Any]] = {}
    for category in (
        "multiple_single_answer",
        "rare_answer_letter",
        "six_or_more_options",
        "all_options_selected",
    ):
        candidates = sorted(
            (row for row in rows if category in target_categories(row)),
            key=lambda row: str(row["question_fingerprint"]),
        )
        for row in candidates[:per_category]:
            fingerprint = str(row["question_fingerprint"])
            selected.setdefault(fingerprint, row)
    return list(selected.values()), category_counts


def main() -> None:
    _, overrides = _parse_args()
    if not REVIEW_PATH.is_file():
        raise FileNotFoundError(REVIEW_PATH)
    rows = _read_jsonl(REVIEW_PATH)
    if len(rows) != 24:
        raise RuntimeError(f"expected 24 review rows, found {len(rows)}")
    manifest_rows = [
        row for path in MANIFEST_PATHS for row in _read_jsonl(path)
    ]
    targeted_rows, target_population = targeted_review_rows(manifest_rows)

    issue_counts = Counter(
        issue
        for row in [*rows, *targeted_rows]
        for issue in review_issues(row)
    )
    if issue_counts:
        raise RuntimeError(f"objective review machine gate failed: {issue_counts}")
    packet = {
        "pack_root": str(PACK_ROOT),
        "sample_count": len(rows),
        "targeted_sample_count": len(targeted_rows),
        "target_population": dict(target_population),
        "question_types": dict(
            Counter(str(row["question_type"]) for row in rows)
        ),
        "machine_issue_counts": {},
        "human_review_required": True,
        "rows": [
            {
                "question_type": row["question_type"],
                "query": row["query"],
                "expected_answer": row["expected_answer"],
                "source_row_id": row["source_row_id"],
                "question_fingerprint": row["question_fingerprint"],
                "issues": [],
            }
            for row in rows
        ],
        "targeted_rows": [
            {
                "target_categories": target_categories(row),
                "question_type": row["question_type"],
                "query": row["query"],
                "expected_answer": row["expected_answer"],
                "source_row_id": row["source_row_id"],
                "question_fingerprint": row["question_fingerprint"],
                "issues": [],
            }
            for row in targeted_rows
        ],
    }
    output_dir = _output_dir(overrides)
    output_path = output_dir / "review_packet.json"
    output_path.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("[objective-sft-review]", flush=True)
    print(json.dumps(packet, ensure_ascii=False, indent=2), flush=True)
    print(f"[objective-sft-review] saved={output_path}", flush=True)


if __name__ == "__main__":
    main()
