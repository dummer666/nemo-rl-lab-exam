#!/usr/bin/env python
"""Deterministically gate selected checkpoints from direct open GRPO."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.qa_objective_sft_gate_eval_wanghaonan import run as gate  # noqa: E402

SAFE_STEP20 = Path(
    "/shared/outputs/wanghaonan/sft_qwen3.5-9b_qa-objective-pilot_wanghaonan/"
    "sft_qwen3.5-9b_qa-objective-pilot_wanghaonan-wanghaonan-20260719-105248/"
    "hf_export/step_20"
)
OBJECTIVE_STEP80 = Path(
    "/shared/outputs/wanghaonan/grpo_qwen3.5-9b_qa-objective-100_wanghaonan/"
    "grpo_qwen3.5-9b_qa-objective-100_wanghaonan-wanghaonan-20260719-141205/"
    "hf_export/step_80"
)
GRPO_ROOT = Path(
    "/shared/outputs/wanghaonan/grpo_qwen3.5-9b_qa-open-direct-200_wanghaonan/"
    "grpo_qwen3.5-9b_qa-open-direct-200_wanghaonan-wanghaonan-20260719-160346"
)
STEPS = (100, 110, 120, 200)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def main(overrides: Sequence[str]) -> None:
    gate.CHECKPOINTS = {
        0: SAFE_STEP20,
        80: OBJECTIVE_STEP80,
        **{
            step: GRPO_ROOT / "hf_export" / f"step_{step}"
            for step in STEPS
        },
    }
    gate.main(overrides)


if __name__ == "__main__":
    _args, unknown = _parse_args()
    main(unknown)
