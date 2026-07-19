#!/usr/bin/env python
"""Merge historical GRPO checkpoints into Hugging Face models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.qa_sft_checkpoint_export_wanghaonan import run as exporter  # noqa: E402

MODEL_NAME = "Qwen/Qwen3.5-9B-Base"
BASE_CHECKPOINT = Path(
    "/data/huggingface/nemo_rl/Qwen/Qwen3.5-9B-Base/iter_0000000"
)
GRPO_ROOT = Path(
    "/shared/outputs/wanghaonan/grpo_qwen3.5-9b_qa-rl-agent_wanghaonan/"
    "grpo_qwen3.5-9b_qa-rl-agent_wanghaonan-wanghaonan-20260718-043801"
)
STEPS = (40, 100, 120)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def main(overrides: list[str]) -> None:
    exporter.MODEL_NAME = MODEL_NAME
    exporter.BASE_CHECKPOINT = BASE_CHECKPOINT
    exporter.SFT_ROOT = GRPO_ROOT
    exporter.STEPS = STEPS
    exporter.main(overrides)


if __name__ == "__main__":
    _args, unknown = _parse_args()
    main(unknown)
