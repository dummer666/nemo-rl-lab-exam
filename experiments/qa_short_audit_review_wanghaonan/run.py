#!/usr/bin/env python
"""Print compact human-review views of the completed short-answer audit."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_AUDIT_DIR = Path(
    "/shared/outputs/wanghaonan/qa_short_gold_audit_wanghaonan/"
    "qa_short_gold_audit_wanghaonan-wanghaonan-20260718-164635/"
    "short_gold_audit"
)


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
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def _output_dir(overrides: Sequence[str]) -> Path:
    for override in overrides:
        if override.startswith("logger.log_dir="):
            output = Path(override.split("=", 1)[1]).parent / "short_audit_review"
            output.mkdir(parents=True, exist_ok=True)
            return output
    output = THIS_DIR / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _evidence_excerpts(row: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    excerpts = []
    for hop in row.get("search_hops") or []:
        for result in hop.get("top_k_results") or []:
            if not result.get("keypoint_matches"):
                continue
            excerpts.append(
                {
                    "hop": hop.get("hop"),
                    "rank": result.get("rank"),
                    "source": result.get("source"),
                    "heading": result.get("heading"),
                    "quality_category": result.get("quality_category"),
                    "keypoint_matches": result.get("keypoint_matches"),
                    "text_excerpt": str(result.get("text", ""))[:500],
                }
            )
            if len(excerpts) >= limit:
                return excerpts
    return excerpts


def _compact_gold(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_split": row.get("dataset_split"),
        "source_row_id": row.get("source_row_id"),
        "primary_attribution": row.get("primary_attribution"),
        "label_defect_reasons": row.get("label_defect_reasons"),
        "support_level": row.get("support_level"),
        "selection_status": row.get("selection_status"),
        "verified_trajectory_search_turns": row.get("verified_trajectory_search_turns"),
        "strict_rebuild_candidate": row.get("strict_rebuild_candidate"),
        "query": row.get("query"),
        "expected_answer": row.get("expected_answer"),
        "full_gold_keypoints": row.get("full_gold_keypoints"),
        "evidence_excerpts": _evidence_excerpts(row),
    }


def _category_examples(
    gold_rows: Sequence[dict[str, Any]],
    *,
    per_category: int = 2,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gold_rows:
        category = str(row.get("primary_attribution", "unknown"))
        if len(groups[category]) < per_category:
            groups[category].append(_compact_gold(row))
    return dict(sorted(groups.items()))


def main() -> None:
    _, overrides = _parse_args()
    audit_dir = Path(os.environ.get("QA_SHORT_AUDIT_DIR", str(DEFAULT_AUDIT_DIR)))
    paths = {
        "summary": audit_dir / "summary.json",
        "gold": audit_dir / "short_gold_audit.jsonl",
        "candidates": audit_dir / "rebuild_candidates.jsonl",
        "representatives": audit_dir / "representative_examples.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing short-audit review inputs: {missing}")

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    gold_rows = _read_jsonl(paths["gold"])
    candidates = _read_jsonl(paths["candidates"])
    representatives = _read_jsonl(paths["representatives"])
    clean_full = [
        _compact_gold(row)
        for row in gold_rows
        if row.get("primary_attribution") == "clean_full_evidence"
    ]
    official_validation = [
        _compact_gold(row)
        for row in gold_rows
        if row.get("dataset_split") == "official_validation"
    ]
    report = {
        "source_audit_dir": str(audit_dir),
        "summary_counts": {
            "gold_labels": summary["gold_labels"],
            "trajectory_audit": summary["trajectory_audit"],
            "cleaning_decision": summary["cleaning_decision"],
            "reward_hacking_regression_step20_sample180": summary[
                "reward_hacking_regression_step20_sample180"
            ],
        },
        "representative_examples": representatives,
        "strict_rebuild_candidates": [
            _compact_gold(row) for row in candidates
        ],
        "clean_full_evidence_labels": clean_full,
        "official_validation_short_labels": official_validation,
        "category_examples": _category_examples(gold_rows),
    }
    if len(representatives) != 15:
        raise RuntimeError(
            f"expected 15 representative examples, got {len(representatives)}"
        )
    if len(candidates) != int(
        summary["cleaning_decision"]["strict_rebuild_candidate_count"]
    ):
        raise RuntimeError("strict candidate count does not match audit summary")
    output_dir = _output_dir(overrides)
    output_path = output_dir / "review_report.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[short-audit-review] report")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[short-audit-review] saved: {output_path}")


if __name__ == "__main__":
    main()
