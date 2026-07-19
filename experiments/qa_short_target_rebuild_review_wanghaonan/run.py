#!/usr/bin/env python
"""Render complete human-review records from a successful rebuild smoke."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.retrieval.qa_target_rebuild import extract_json_object  # noqa: E402

DEFAULT_REBUILD_DIR = Path(
    "/shared/outputs/wanghaonan/qa_short_target_rebuild_wanghaonan/"
    "qa_short_target_rebuild_wanghaonan-wanghaonan-20260718-184512/"
    "short_target_rebuild"
)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


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
            output = Path(override.split("=", 1)[1]).parent / "short_target_review"
            output.mkdir(parents=True, exist_ok=True)
            return output
    output = Path(__file__).resolve().parent / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _assert_visible_quotes(target: Mapping[str, Any]) -> None:
    observations = "\n".join(
        str(hop.get("observation", ""))
        for hop in target.get("search_hops", [])
    )
    for point in target.get("answer_points", []):
        quote = str(point.get("quote", ""))
        if not quote or quote not in observations:
            raise ValueError(
                f"row {target.get('source_row_id')}: quote is not visible: {quote!r}"
            )


def _selected_generation_attempts(
    target: Mapping[str, Any],
    generation_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    source_row_id = int(target["source_row_id"])
    return [
        dict(row)
        for row in generation_rows
        if int(row["source_row_id"]) == source_row_id
    ]


def _verifier_failure_categories(
    payload: Mapping[str, Any] | None,
    point_count: int,
) -> list[str]:
    if payload is None:
        return ["invalid_json"]

    categories = []
    if payload.get("decision") != "reject":
        categories.append("invalid_decision")
    if payload.get("complete") is not True:
        categories.append("incomplete")

    checks = payload.get("point_checks")
    if not isinstance(checks, list) or len(checks) != point_count:
        categories.append("point_check_count")
        return categories
    if any(
        not isinstance(check, Mapping) or check.get("supported") is not True
        for check in checks
    ):
        categories.append("unsupported_point")
    if any(
        not isinstance(check, Mapping) or check.get("relevant") is not True
        for check in checks
    ):
        categories.append("irrelevant_point")
    return categories or ["invalid_reject_schema"]


def _build_report(
    summary: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    generation_rows: Sequence[Mapping[str, Any]],
    route_rows: Sequence[Mapping[str, Any]],
    rejected_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if int(summary.get("official_validation_overlap_count", -1)) != 0:
        raise ValueError("smoke summary reports official-validation leakage")
    if len(targets) != int(summary.get("machine_verified_route_targets", -1)):
        raise ValueError("target count does not match smoke summary")

    accepted = []
    for target in targets:
        _assert_visible_quotes(target)
        final_completion = str(target["messages"][-1]["content"])
        missing_statements = [
            str(point["statement"])
            for point in target["answer_points"]
            if str(point["statement"]) not in final_completion
        ]
        if missing_statements:
            raise ValueError(
                f"row {target['source_row_id']}: final answer omits {missing_statements}"
            )
        accepted.append(
            {
                "source_row_id": int(target["source_row_id"]),
                "question_fingerprint": str(target["question_fingerprint"]),
                "split": str(target["split"]),
                "query": str(target["query"]),
                "legacy_expected_answer": str(target["legacy_expected_answer"]),
                "rebuilt_expected_answer": str(target["expected_answer"]),
                "answer_points": list(target["answer_points"]),
                "search_turns": int(target["search_turns"]),
                "search_hops": list(target["search_hops"]),
                "final_completion": final_completion,
                "token_protocol_audit": dict(target["_audit"]),
                "teacher_and_verifier_attempts": _selected_generation_attempts(
                    target,
                    generation_rows,
                ),
                "route_audit": [
                    dict(row)
                    for row in route_rows
                    if int(row["source_row_id"]) == int(target["source_row_id"])
                ],
                "human_review_checklist": {
                    "all_points_answer_question": None,
                    "all_quotes_literal_and_sufficient": None,
                    "no_unsupported_claims": None,
                    "route_queries_do_not_leak_answer": None,
                    "complete_readable_answer": None,
                    "decision": "pending_human_review",
                },
            }
        )

    rejection_counts = Counter(
        f"{row.get('stage')}:{row.get('reason')}"
        for row in rejected_rows
    )
    rejection_examples = []
    seen_reasons = set()
    for row in rejected_rows:
        reason = f"{row.get('stage')}:{row.get('reason')}"
        if reason in seen_reasons:
            continue
        seen_reasons.add(reason)
        rejection_examples.append(
            {
                "source_row_id": row.get("source_row_id"),
                "query": row.get("query"),
                "stage": row.get("stage"),
                "reason": row.get("reason"),
            }
        )
        if len(rejection_examples) >= 10:
            break
    generation_counts = Counter(
        str(row.get("deterministic_decision"))
        for row in generation_rows
    )
    verifier_counts = Counter(
        "accepted" if row.get("verifier_accept") is True else "rejected"
        for row in generation_rows
        if row.get("deterministic_decision") == "accepted"
    )
    verifier_failure_counts: Counter[str] = Counter()
    verifier_rejections = []
    for row in generation_rows:
        if (
            row.get("deterministic_decision") != "accepted"
            or row.get("verifier_accept") is True
        ):
            continue
        points = row.get("points")
        point_count = len(points) if isinstance(points, list) else 0
        verifier_raw = str(row.get("verifier_raw", ""))
        verifier_payload = extract_json_object(verifier_raw)
        categories = _verifier_failure_categories(
            verifier_payload,
            point_count,
        )
        verifier_failure_counts.update(categories)
        verifier_rejections.append(
            {
                "source_row_id": row.get("source_row_id"),
                "candidate_index": row.get("candidate_index"),
                "query": row.get("query"),
                "points": points,
                "failure_categories": categories,
                "verifier_payload": verifier_payload,
                "verifier_raw": verifier_raw[:2000],
            }
        )
    generation_examples = []
    seen_generation_decisions = set()
    for row in generation_rows:
        decision = str(row.get("deterministic_decision"))
        verifier_rejected = (
            decision == "accepted"
            and row.get("verifier_accept") is not True
        )
        example_key = (
            "verifier_rejected"
            if verifier_rejected
            else decision
        )
        if example_key in seen_generation_decisions:
            continue
        seen_generation_decisions.add(example_key)
        generation_examples.append(
            {
                "source_row_id": row.get("source_row_id"),
                "candidate_index": row.get("candidate_index"),
                "query": row.get("query"),
                "deterministic_decision": decision,
                "points": row.get("points"),
                "verifier_accept": row.get("verifier_accept"),
                "raw_generation": str(row.get("raw_generation", ""))[:2000],
                "verifier_raw": str(row.get("verifier_raw", ""))[:2000],
            }
        )
        if len(generation_examples) >= 15:
            break
    return {
        "source_summary": dict(summary),
        "accepted_target_count": len(accepted),
        "accepted_targets": accepted,
        "generation_decision_counts": dict(sorted(generation_counts.items())),
        "independent_verifier_counts": dict(sorted(verifier_counts.items())),
        "independent_verifier_failure_counts": dict(
            sorted(verifier_failure_counts.items())
        ),
        "independent_verifier_rejections": verifier_rejections,
        "representative_generation_failures": generation_examples,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "representative_rejections": rejection_examples,
    }


def main() -> None:
    _, overrides = _parse_args()
    rebuild_dir = Path(
        os.environ.get("QA_REBUILD_REVIEW_DIR", str(DEFAULT_REBUILD_DIR))
    )
    paths = {
        "summary": rebuild_dir / "summary.json",
        "targets": rebuild_dir / "machine_verified_targets.jsonl",
        "generation": rebuild_dir / "generation_audit.jsonl",
        "routes": rebuild_dir / "route_audit.jsonl",
        "rejected": rebuild_dir / "rejected_candidates.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing rebuild review inputs: {missing}")

    report = _build_report(
        _read_json(paths["summary"]),
        _read_jsonl(paths["targets"]),
        _read_jsonl(paths["generation"]),
        _read_jsonl(paths["routes"]),
        _read_jsonl(paths["rejected"]),
    )
    output_dir = _output_dir(overrides)
    report_path = output_dir / "review_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[short-target-review] report", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"[short-target-review] saved: {report_path}", flush=True)


if __name__ == "__main__":
    main()
