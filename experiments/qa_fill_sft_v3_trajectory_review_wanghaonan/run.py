#!/usr/bin/env python
"""Build a compact human-review packet from the stronger fill trajectory audit."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

AUDIT_ROOT = Path(
    "/shared/outputs/wanghaonan/qa_fill_sft_v3_trajectory_audit_wanghaonan/"
    "qa_fill_sft_v3_trajectory_audit_wanghaonan-wanghaonan-20260719-045411/"
    "trajectory_audit"
)
CRITICAL_DIAGNOSES = {
    "reward_regression",
    "synthesis_failure_after_full_evidence",
    "incremental_evidence_not_used",
}


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def _output_dir(overrides: Sequence[str]) -> Path:
    for override in overrides:
        if override.startswith("logger.log_dir="):
            output = Path(override.split("=", 1)[1]).parent / "trajectory_review"
            output.mkdir(parents=True, exist_ok=True)
            return output
    output = Path(__file__).resolve().parent / "outputs"
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


def select_review_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[list[Mapping[str, Any]]]:
    by_row: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_row[int(row["row_index"])].append(row)
    for versions in by_row.values():
        versions.sort(key=lambda row: int(row["step"]))

    selected: list[int] = []
    for row_index, versions in sorted(by_row.items()):
        diagnoses = {str(row["primary_diagnosis"]) for row in versions}
        if diagnoses & CRITICAL_DIAGNOSES:
            selected.append(row_index)

    quotas = {
        "off_target_second_search": 4,
        "redundant_second_search": 4,
    }
    for diagnosis, quota in quotas.items():
        candidates = [
            row_index
            for row_index, versions in sorted(by_row.items())
            if row_index not in selected
            and any(
                str(row["primary_diagnosis"]) == diagnosis
                for row in versions
            )
        ]
        selected.extend(candidates[:quota])
    return [by_row[row_index] for row_index in selected]


def _compact(versions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = versions[0]
    return {
        "row_index": int(first["row_index"]),
        "question_type": str(first["question_type"]),
        "query": str(first["query"]),
        "expected_answer": str(first["expected_answer"]),
        "baseline": first["baseline"],
        "versions": [
            {
                "step": int(row["step"]),
                "diagnosis": str(row["primary_diagnosis"]),
                "reward": float(row["candidate"]["reward"]),
                "reward_delta": float(row["reward_delta"]),
                "boxed_answer": row["candidate"]["boxed_answer"],
                "final_response": str(
                    row["candidate"]["assistant_responses"][-1]
                )[:1600],
                "search_queries": row["search"]["queries"],
                "exact_duplicate_query": row["search"]["exact_duplicate"],
                "query_similarity": row["search"]["query_similarity"],
                "first_sources": row["search"]["first_sources"],
                "second_sources": row["search"]["second_sources"],
                "new_sources": row["search"]["new_sources"],
                "first_keypoint_hits": row["evidence"]["first_keypoint_hits"],
                "second_keypoint_hits": row["evidence"]["second_keypoint_hits"],
                "incremental_keypoint_hits": row["evidence"][
                    "incremental_keypoint_hits"
                ],
                "cumulative_keypoint_hits": row["evidence"][
                    "cumulative_keypoint_hits"
                ],
                "keypoint_count": row["evidence"]["keypoint_count"],
            }
            for row in versions
        ],
    }


def main() -> None:
    _, overrides = _parse_args()
    output_dir = _output_dir(overrides)
    rows_path = AUDIT_ROOT / "two_hop_rows.jsonl"
    if not rows_path.is_file():
        raise FileNotFoundError(f"missing trajectory audit rows: {rows_path}")
    rows = _read_jsonl(rows_path)
    if len(rows) != 47:
        raise RuntimeError(f"expected 47 step-specific two-hop rows, got {len(rows)}")

    selected = select_review_rows(rows)
    packet = {
        "source": str(rows_path),
        "selection": {
            "critical_diagnoses": sorted(CRITICAL_DIAGNOSES),
            "off_target_sample": 4,
            "redundant_sample": 4,
        },
        "unique_rows_in_audit": len(
            {int(row["row_index"]) for row in rows}
        ),
        "selected_unique_rows": len(selected),
        "human_reviewed": False,
        "rows": [_compact(versions) for versions in selected],
    }
    output_path = output_dir / "review_packet.json"
    output_path.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("[fill-sft-v3-trajectory-review]", flush=True)
    print(json.dumps(packet, ensure_ascii=False, indent=2), flush=True)
    print(f"[fill-sft-v3-trajectory-review] saved={output_path}", flush=True)


if __name__ == "__main__":
    main()
