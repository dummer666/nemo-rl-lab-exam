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
    "qa_objective_sft_data_wanghaonan-wanghaonan-20260719-102742/"
    "objective_sft_data"
)
REVIEW_PATH = PACK_ROOT / "review_sample.jsonl"
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


def main() -> None:
    _, overrides = _parse_args()
    if not REVIEW_PATH.is_file():
        raise FileNotFoundError(REVIEW_PATH)
    rows = _read_jsonl(REVIEW_PATH)
    if len(rows) != 24:
        raise RuntimeError(f"expected 24 review rows, found {len(rows)}")

    issue_counts = Counter(
        issue for row in rows for issue in review_issues(row)
    )
    if issue_counts:
        raise RuntimeError(f"objective review machine gate failed: {issue_counts}")
    packet = {
        "pack_root": str(PACK_ROOT),
        "sample_count": len(rows),
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
