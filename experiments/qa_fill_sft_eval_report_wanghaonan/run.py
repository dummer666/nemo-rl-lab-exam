#!/usr/bin/env python
"""Report paired baseline-versus-fill-SFT changes from persisted trajectories."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

EVAL_ROOT = Path(
    "/shared/outputs/wanghaonan/qa_fill_sft_multiturn_eval_wanghaonan/"
    "qa_fill_sft_multiturn_eval_wanghaonan-wanghaonan-20260719-033956/"
    "sft_multiturn_eval"
)
BASELINE_STEP = 50
CANDIDATE_STEP = 13
OBJECTIVE_TYPES = {"single", "multiple", "bool"}


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def _output_dir(overrides: Sequence[str]) -> Path:
    for override in overrides:
        if override.startswith("logger.log_dir="):
            output = Path(override.split("=", 1)[1]).parent / "fill_sft_eval_report"
            output.mkdir(parents=True, exist_ok=True)
            return output
    output = Path(__file__).resolve().parent / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    return output


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


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rewards = [
        max(0.0, min(1.0, float(row.get("reward", 0.0))))
        for row in rows
    ]
    return {
        "count": len(rows),
        "accuracy": mean(rewards) if rewards else 0.0,
        "perfect_count": sum(reward >= 1.0 for reward in rewards),
    }


def build_report(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    baseline_summary: Mapping[str, Any],
    candidate_summary: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = {int(row["row_index"]): row for row in baseline_rows}
    candidate = {int(row["row_index"]): row for row in candidate_rows}
    if set(baseline) != set(candidate):
        raise ValueError("evaluations do not contain the same validation rows")

    by_type: dict[str, list[int]] = defaultdict(list)
    for row_index, row in baseline.items():
        by_type[str(row["question_type"])].append(row_index)

    paired_by_type = {}
    for question_type, row_indexes in sorted(by_type.items()):
        deltas = [
            float(candidate[row_index]["reward"])
            - float(baseline[row_index]["reward"])
            for row_index in row_indexes
        ]
        paired_by_type[question_type] = {
            "mean_reward_delta": mean(deltas),
            "gain_count": sum(delta > 1e-9 for delta in deltas),
            "loss_count": sum(delta < -1e-9 for delta in deltas),
            "unchanged_count": sum(abs(delta) <= 1e-9 for delta in deltas),
            "perfect_gain_count": sum(
                float(baseline[row_index]["reward"]) < 1.0
                and float(candidate[row_index]["reward"]) >= 1.0
                for row_index in row_indexes
            ),
            "perfect_loss_count": sum(
                float(baseline[row_index]["reward"]) >= 1.0
                and float(candidate[row_index]["reward"]) < 1.0
                for row_index in row_indexes
            ),
        }

    objective_indexes = [
        row_index
        for row_index, row in baseline.items()
        if row["question_type"] in OBJECTIVE_TYPES
    ]
    open_indexes = [
        row_index
        for row_index, row in baseline.items()
        if row["question_type"] in {"fill", "short"}
    ]
    changed_open = []
    for row_index in open_indexes:
        before = baseline[row_index]
        after = candidate[row_index]
        if (
            abs(float(after["reward"]) - float(before["reward"])) <= 1e-9
            and int(after["search_count"]) == int(before["search_count"])
            and after["assistant_responses"] == before["assistant_responses"]
        ):
            continue
        changed_open.append(
            {
                "row_index": row_index,
                "question_type": before["question_type"],
                "query": before["query"],
                "expected_answer": before["expected_answer"],
                "baseline": {
                    "reward": before["reward"],
                    "search_count": before["search_count"],
                    "assistant_responses": before["assistant_responses"],
                },
                "candidate": {
                    "reward": after["reward"],
                    "search_count": after["search_count"],
                    "assistant_responses": after["assistant_responses"],
                },
            }
        )

    baseline_objective = _metrics(
        [baseline[row_index] for row_index in objective_indexes]
    )
    candidate_objective = _metrics(
        [candidate[row_index] for row_index in objective_indexes]
    )
    baseline_open = _metrics([baseline[row_index] for row_index in open_indexes])
    candidate_open = _metrics([candidate[row_index] for row_index in open_indexes])
    return {
        "baseline": {
            "step": BASELINE_STEP,
            "accuracy": baseline_summary["accuracy"],
            "question_types": baseline_summary["question_types"],
            "retrieval": baseline_summary["retrieval"],
            "protocol": baseline_summary["protocol"],
            "objective": baseline_objective,
            "open": baseline_open,
        },
        "candidate": {
            "step": CANDIDATE_STEP,
            "accuracy": candidate_summary["accuracy"],
            "question_types": candidate_summary["question_types"],
            "retrieval": candidate_summary["retrieval"],
            "protocol": candidate_summary["protocol"],
            "objective": candidate_objective,
            "open": candidate_open,
        },
        "deltas": {
            "overall_accuracy": (
                float(candidate_summary["accuracy"])
                - float(baseline_summary["accuracy"])
            ),
            "objective_accuracy": (
                candidate_objective["accuracy"]
                - baseline_objective["accuracy"]
            ),
            "open_accuracy": (
                candidate_open["accuracy"]
                - baseline_open["accuracy"]
            ),
            "paired_by_type": paired_by_type,
        },
        "changed_open_count": len(changed_open),
        "changed_open": changed_open,
    }


def main() -> None:
    _, overrides = _parse_args()
    output_dir = _output_dir(overrides)
    required = [
        EVAL_ROOT / f"step_{step}" / filename
        for step in (BASELINE_STEP, CANDIDATE_STEP)
        for filename in ("summary.json", "trajectories.jsonl")
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing fill SFT evaluation artifacts: {missing}")
    summaries = {
        step: _read_json(EVAL_ROOT / f"step_{step}" / "summary.json")
        for step in (BASELINE_STEP, CANDIDATE_STEP)
    }
    rows = {
        step: _read_jsonl(EVAL_ROOT / f"step_{step}" / "trajectories.jsonl")
        for step in (BASELINE_STEP, CANDIDATE_STEP)
    }
    report = build_report(
        rows[BASELINE_STEP],
        rows[CANDIDATE_STEP],
        summaries[BASELINE_STEP],
        summaries[CANDIDATE_STEP],
    )
    path = output_dir / "report.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("[fill-sft-eval-report]", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"[fill-sft-eval-report] saved={path}", flush=True)


if __name__ == "__main__":
    main()
