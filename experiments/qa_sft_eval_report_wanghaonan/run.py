#!/usr/bin/env python
"""Report detailed metrics and paired trajectory changes from the SFT evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Sequence

EVAL_ROOT = Path(
    "/shared/outputs/wanghaonan/qa_sft_multiturn_eval_wanghaonan/"
    "qa_sft_multiturn_eval_wanghaonan-wanghaonan-20260718-141126/"
    "sft_multiturn_eval"
)
STEPS = (25, 50)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def _output_dir(overrides: Sequence[str]) -> Path:
    for override in overrides:
        if override.startswith("logger.log_dir="):
            output = Path(override.split("=", 1)[1]).parent / "sft_eval_report"
            output.mkdir(parents=True, exist_ok=True)
            return output
    output = Path(__file__).resolve().parent / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def _trajectory_excerpt(row: dict) -> dict:
    return {
        "row_index": int(row["row_index"]),
        "question_type": str(row["question_type"]),
        "query": str(row["query"]),
        "expected_answer": str(row["expected_answer"]),
        "reward": float(row["reward"]),
        "search_count": int(row["search_count"]),
        "search_queries": list(row["search_queries"]),
        "evidence_coverage": float(row["evidence_coverage"]),
        "protocol_error": bool(row["protocol_error"]),
        "termination_reason": row["termination_reason"],
        "assistant_responses": list(row["assistant_responses"]),
    }


def _summary_excerpt(summary: dict) -> dict:
    return {
        "accuracy": float(summary["accuracy"]),
        "mean_reward": float(summary["mean_reward"]),
        "perfect_count": int(summary["perfect_count"]),
        "perfect_rate": float(summary["perfect_rate"]),
        "open_questions": summary["open_questions"],
        "retrieval": summary["retrieval"],
        "protocol": summary["protocol"],
        "question_types": summary["question_types"],
        "evaluation_seconds": float(summary["evaluation_seconds"]),
    }


def _build_report(
    summaries: dict[int, dict],
    trajectories: dict[int, Sequence[dict]],
) -> dict:
    indexed = {step: {int(row["row_index"]): row for row in trajectories[step]} for step in STEPS}
    row_ids = set(indexed[STEPS[0]])
    if row_ids != set(indexed[STEPS[1]]):
        raise ValueError("checkpoint evaluations do not contain the same validation rows")

    deltas = {row_id: float(indexed[50][row_id]["reward"]) - float(indexed[25][row_id]["reward"]) for row_id in row_ids}
    open_perfect = {
        step: {
            row_id
            for row_id, row in indexed[step].items()
            if row["question_type"] in {"fill", "short"} and float(row["reward"]) >= 1.0
        }
        for step in STEPS
    }
    notable_open_ids = sorted(open_perfect[25] | open_perfect[50])
    protocol_ids = {
        step: sorted(row_id for row_id, row in indexed[step].items() if bool(row["protocol_error"])) for step in STEPS
    }

    return {
        "summaries": {str(step): _summary_excerpt(summaries[step]) for step in STEPS},
        "paired_changes": {
            "mean_reward_delta_step50_minus_step25": mean(deltas.values()),
            "gain_count": sum(delta > 1e-9 for delta in deltas.values()),
            "loss_count": sum(delta < -1e-9 for delta in deltas.values()),
            "unchanged_count": sum(abs(delta) <= 1e-9 for delta in deltas.values()),
            "perfect_gain_count": sum(
                float(indexed[25][row_id]["reward"]) < 1.0 and float(indexed[50][row_id]["reward"]) >= 1.0
                for row_id in row_ids
            ),
            "perfect_loss_count": sum(
                float(indexed[25][row_id]["reward"]) >= 1.0 and float(indexed[50][row_id]["reward"]) < 1.0
                for row_id in row_ids
            ),
        },
        "open_perfect": {
            "step25_ids": sorted(open_perfect[25]),
            "step50_ids": sorted(open_perfect[50]),
            "shared_ids": sorted(open_perfect[25] & open_perfect[50]),
            "step25_only_ids": sorted(open_perfect[25] - open_perfect[50]),
            "step50_only_ids": sorted(open_perfect[50] - open_perfect[25]),
            "examples": [
                {
                    "row_index": row_id,
                    "step25": _trajectory_excerpt(indexed[25][row_id]),
                    "step50": _trajectory_excerpt(indexed[50][row_id]),
                }
                for row_id in notable_open_ids
            ],
        },
        "protocol_errors": {
            "step25_ids": protocol_ids[25],
            "step50_ids": protocol_ids[50],
            "examples": {
                str(step): [_trajectory_excerpt(indexed[step][row_id]) for row_id in protocol_ids[step]]
                for step in STEPS
            },
        },
    }


def main() -> None:
    _, overrides = _parse_args()
    output_dir = _output_dir(overrides)
    required = [
        EVAL_ROOT / f"step_{step}" / filename for step in STEPS for filename in ("summary.json", "trajectories.jsonl")
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing SFT evaluation artifacts: {missing}")

    summaries = {step: _read_json(EVAL_ROOT / f"step_{step}" / "summary.json") for step in STEPS}
    trajectories = {step: _read_jsonl(EVAL_ROOT / f"step_{step}" / "trajectories.jsonl") for step in STEPS}
    report = _build_report(summaries, trajectories)
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[sft-eval-report] " + json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[sft-eval-report] saved: {report_path}")


if __name__ == "__main__":
    main()
