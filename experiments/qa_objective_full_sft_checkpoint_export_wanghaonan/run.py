#!/usr/bin/env python
"""Merge full objective SFT checkpoints into Hugging Face models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.qa_sft_checkpoint_export_wanghaonan import run as exporter  # noqa: E402

PILOT_STEP20 = Path(
    "/shared/outputs/wanghaonan/sft_qwen3.5-9b_qa-objective-pilot_wanghaonan/"
    "sft_qwen3.5-9b_qa-objective-pilot_wanghaonan-wanghaonan-20260719-105248/"
    "hf_export/step_20"
)
CONVERTED_PILOT_STEP20 = Path(
    "/data/huggingface/nemo_rl/"
    "model__shared_outputs_wanghaonan_sft_qwen3.5-9b_qa-objective-pilot_"
    "wanghaonan_sft_qwen3.5-9b_qa-objective-pilot_wanghaonan-wanghaonan-"
    "20260719-105248_hf_export_step_20/iter_0000000"
)
SFT_ROOT = Path(
    "/shared/outputs/wanghaonan/sft_qwen3.5-9b_qa-objective-full_wanghaonan/"
    "sft_qwen3.5-9b_qa-objective-full_wanghaonan-wanghaonan-20260719-123157"
)
STEPS = (50, 100, 150, 200, 250)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def main(overrides: list[str]) -> None:
    exporter.MODEL_NAME = str(PILOT_STEP20)
    exporter.BASE_CHECKPOINT = CONVERTED_PILOT_STEP20
    exporter.SFT_ROOT = SFT_ROOT
    exporter.STEPS = STEPS
    exporter.main(overrides)


if __name__ == "__main__":
    _args, unknown = _parse_args()
    main(unknown)
