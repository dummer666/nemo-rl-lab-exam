#!/usr/bin/env python
"""Review every reward-changing objective SFT pilot trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

GATE_ROOT = Path(
    "/shared/outputs/wanghaonan/qa_objective_sft_gate_eval_wanghaonan/"
    "qa_objective_sft_gate_eval_wanghaonan-wanghaonan-20260719-110357/"
    "objective_sft_gate_eval"
)
STEPS = (10, 20)
DATASETS = ("internal", "official")


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def _output_dir(overrides: Sequence[str]) -> Path:
    for override in overrides:
        if override.startswith("logger.log_dir="):
            output = (
                Path(override.split("=", 1)[1]).parent
                / "objective_sft_delta_review"
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
                    raise ValueError(f"{path}: expected JSON objects")
                rows.append(value)
    return rows


def changed_rows(
    baseline: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(baseline) != len(candidate):
        raise ValueError("baseline and candidate populations differ")
    changes = []
    for base, trial in zip(baseline, candidate, strict=True):
        if (
            base["row_index"] != trial["row_index"]
            or base["query"] != trial["query"]
            or base["expected_answer"] != trial["expected_answer"]
        ):
            raise ValueError("baseline and candidate rows are misaligned")
        base_reward = float(base["reward"])
        trial_reward = float(trial["reward"])
        if abs(trial_reward - base_reward) <= 1e-12:
            continue
        changes.append(
            {
                "row_index": base["row_index"],
                "question_type": base["question_type"],
                "query": base["query"],
                "expected_answer": base["expected_answer"],
                "baseline": {
                    "reward": base_reward,
                    "responses": base["assistant_responses"],
                    "search_count": base["search_count"],
                },
                "candidate": {
                    "reward": trial_reward,
                    "responses": trial["assistant_responses"],
                    "search_count": trial["search_count"],
                },
                "reward_delta": trial_reward - base_reward,
            }
        )
    return changes


def main(overrides: Sequence[str]) -> None:
    required = [
        GATE_ROOT / f"step_{step}" / f"{dataset}_trajectories.jsonl"
        for step in (0, *STEPS)
        for dataset in DATASETS
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing gate trajectories: {missing}")
    packet: dict[str, Any] = {
        "gate_root": str(GATE_ROOT),
        "comparisons": {},
        "human_review_required": True,
    }
    for dataset in DATASETS:
        baseline = _read_jsonl(
            GATE_ROOT / "step_0" / f"{dataset}_trajectories.jsonl"
        )
        for step in STEPS:
            candidate = _read_jsonl(
                GATE_ROOT
                / f"step_{step}"
                / f"{dataset}_trajectories.jsonl"
            )
            changes = changed_rows(baseline, candidate)
            key = f"{dataset}:step_{step}"
            packet["comparisons"][key] = {
                "changed_count": len(changes),
                "improved_count": sum(
                    change["reward_delta"] > 0 for change in changes
                ),
                "regressed_count": sum(
                    change["reward_delta"] < 0 for change in changes
                ),
                "reward_delta": sum(
                    change["reward_delta"] for change in changes
                ),
                "changes": changes,
            }
    output_dir = _output_dir(overrides)
    output_path = output_dir / "delta_review.json"
    output_path.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("[objective-delta-review]", flush=True)
    print(json.dumps(packet, ensure_ascii=False, indent=2), flush=True)
    print(f"[objective-delta-review] saved={output_path}", flush=True)


if __name__ == "__main__":
    _args, unknown = _parse_args()
    main(unknown)
