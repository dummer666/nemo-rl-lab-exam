#!/usr/bin/env python
"""Evaluate stronger fill-SFT checkpoints and compare them with step 50."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.qa_sft_multiturn_eval_wanghaonan import run as evaluator  # noqa: E402

RETRIEVAL_SFT_STEP50 = Path(
    "/shared/outputs/wanghaonan/sft_qwen3.5-9b_qa-retrieval-v1_wanghaonan/"
    "sft_qwen3.5-9b_qa-retrieval-v1_wanghaonan-wanghaonan-20260718-130828/"
    "hf_export/step_50"
)
FILL_SFT_ROOT = Path(
    "/shared/outputs/wanghaonan/sft_qwen3.5-9b_qa-fill-v3_wanghaonan/"
    "sft_qwen3.5-9b_qa-fill-v3_wanghaonan-wanghaonan-20260719-043000"
)
BASELINE_EVAL_ROOT = Path(
    "/shared/outputs/wanghaonan/qa_fill_sft_multiturn_eval_wanghaonan/"
    "qa_fill_sft_multiturn_eval_wanghaonan-wanghaonan-20260719-033956/"
    "sft_multiturn_eval"
)
BASELINE_STEP = 50
CANDIDATE_STEPS = (20, 40, 60)
OBJECTIVE_TYPES = {"single", "multiple", "bool"}


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


def _slice_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        "objective": _metrics(
            [row for row in rows if row["question_type"] in OBJECTIVE_TYPES]
        ),
        "open": _metrics(
            [row for row in rows if row["question_type"] in {"fill", "short"}]
        ),
        "fill": _metrics([row for row in rows if row["question_type"] == "fill"]),
        "short": _metrics([row for row in rows if row["question_type"] == "short"]),
    }


def _paired_changes(
    baseline: Mapping[int, Mapping[str, Any]],
    candidate: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    by_type: dict[str, list[int]] = defaultdict(list)
    for row_index, row in baseline.items():
        by_type[str(row["question_type"])].append(row_index)

    result = {}
    for question_type, indexes in sorted(by_type.items()):
        deltas = [
            float(candidate[index]["reward"]) - float(baseline[index]["reward"])
            for index in indexes
        ]
        result[question_type] = {
            "mean_reward_delta": mean(deltas),
            "gain_count": sum(delta > 1e-9 for delta in deltas),
            "loss_count": sum(delta < -1e-9 for delta in deltas),
            "perfect_gain_count": sum(
                float(baseline[index]["reward"]) < 1.0
                and float(candidate[index]["reward"]) >= 1.0
                for index in indexes
            ),
            "perfect_loss_count": sum(
                float(baseline[index]["reward"]) >= 1.0
                and float(candidate[index]["reward"]) < 1.0
                for index in indexes
            ),
            "response_change_count": sum(
                candidate[index]["assistant_responses"]
                != baseline[index]["assistant_responses"]
                for index in indexes
            ),
        }
    return result


def build_comparison(
    baseline_summary: Mapping[str, Any],
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_summaries: Mapping[int, Mapping[str, Any]],
    candidate_rows: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    baseline = {int(row["row_index"]): row for row in baseline_rows}
    baseline_slices = _slice_metrics(baseline_rows)
    candidates = {}

    for step in CANDIDATE_STEPS:
        rows = candidate_rows[step]
        indexed = {int(row["row_index"]): row for row in rows}
        if set(indexed) != set(baseline):
            raise ValueError(f"step {step} does not contain the baseline validation rows")
        slices = _slice_metrics(rows)
        objective_delta = (
            slices["objective"]["accuracy"]
            - baseline_slices["objective"]["accuracy"]
        )
        fill_delta = slices["fill"]["accuracy"] - baseline_slices["fill"]["accuracy"]
        two_search_delta = (
            int(candidate_summaries[step]["retrieval"]["two_search_count"])
            - int(baseline_summary["retrieval"]["two_search_count"])
        )
        eligible = fill_delta > 1e-9 and objective_delta >= -0.02
        candidates[str(step)] = {
            "accuracy": candidate_summaries[step]["accuracy"],
            "question_types": candidate_summaries[step]["question_types"],
            "retrieval": candidate_summaries[step]["retrieval"],
            "protocol": candidate_summaries[step]["protocol"],
            "slices": slices,
            "deltas": {
                "overall_accuracy": (
                    float(candidate_summaries[step]["accuracy"])
                    - float(baseline_summary["accuracy"])
                ),
                "objective_accuracy": objective_delta,
                "open_accuracy": (
                    slices["open"]["accuracy"]
                    - baseline_slices["open"]["accuracy"]
                ),
                "fill_accuracy": fill_delta,
                "short_accuracy": (
                    slices["short"]["accuracy"]
                    - baseline_slices["short"]["accuracy"]
                ),
                "two_search_count": two_search_delta,
            },
            "paired_by_type": _paired_changes(baseline, indexed),
            "grpo_gate": {
                "fill_improved": fill_delta > 1e-9,
                "objective_regression_within_2pp": objective_delta >= -0.02,
                "eligible": eligible,
            },
        }

    eligible_steps = [
        step
        for step in CANDIDATE_STEPS
        if candidates[str(step)]["grpo_gate"]["eligible"]
    ]
    recommended = (
        max(
            eligible_steps,
            key=lambda step: (
                candidates[str(step)]["slices"]["fill"]["accuracy"],
                candidates[str(step)]["accuracy"],
                -step,
            ),
        )
        if eligible_steps
        else None
    )
    return {
        "baseline": {
            "step": BASELINE_STEP,
            "accuracy": baseline_summary["accuracy"],
            "question_types": baseline_summary["question_types"],
            "retrieval": baseline_summary["retrieval"],
            "protocol": baseline_summary["protocol"],
            "slices": baseline_slices,
        },
        "candidates": candidates,
        "decision": {
            "recommended_step_for_short_grpo": recommended,
            "eligible_steps": eligible_steps,
            "rule": "fill accuracy must improve and objective regression must be <= 2pp",
            "judge_available": False,
            "short_scores_semantic": False,
        },
    }


def _load_evaluation(root: Path, step: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = root / f"step_{step}" / "summary.json"
    trajectories_path = root / f"step_{step}" / "trajectories.jsonl"
    missing = [
        str(path)
        for path in (summary_path, trajectories_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing evaluation artifacts: {missing}")
    return _read_json(summary_path), _read_jsonl(trajectories_path)


def main(overrides: Sequence[str]) -> None:
    evaluator.CHECKPOINTS = {
        step: FILL_SFT_ROOT / "hf_export" / f"step_{step}"
        for step in CANDIDATE_STEPS
    }
    evaluator.main()

    candidate_root = evaluator._output_dir(overrides)
    baseline_summary, baseline_rows = _load_evaluation(
        BASELINE_EVAL_ROOT,
        BASELINE_STEP,
    )
    loaded_candidates = {
        step: _load_evaluation(candidate_root, step)
        for step in CANDIDATE_STEPS
    }
    report = build_comparison(
        baseline_summary,
        baseline_rows,
        {step: loaded_candidates[step][0] for step in CANDIDATE_STEPS},
        {step: loaded_candidates[step][1] for step in CANDIDATE_STEPS},
    )
    report_path = candidate_root / "baseline_comparison.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("[fill-sft-v3-eval]", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"[fill-sft-v3-eval] saved={report_path}", flush=True)


if __name__ == "__main__":
    _args, unknown = _parse_args()
    main(unknown)
