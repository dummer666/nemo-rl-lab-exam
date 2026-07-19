#!/usr/bin/env python
"""Export selected checkpoints from the 200-step direct open GRPO run."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.qa_sft_checkpoint_export_wanghaonan import run as exporter  # noqa: E402

BASE_MODEL = Path(
    "/shared/outputs/wanghaonan/grpo_qwen3.5-9b_qa-objective-100_wanghaonan/"
    "grpo_qwen3.5-9b_qa-objective-100_wanghaonan-wanghaonan-20260719-141205/"
    "hf_export/step_80"
)
CONVERTED_BASE = Path(
    "/shared/outputs/wanghaonan/nemo_rl_megatron_cache/"
    "model__shared_outputs_wanghaonan_grpo_qwen3.5-9b_qa-objective-100_"
    "wanghaonan_grpo_qwen3.5-9b_qa-objective-100_wanghaonan-wanghaonan-"
    "20260719-141205_hf_export_step_80/iter_0000000"
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


def main(overrides: list[str]) -> None:
    partial_step100 = GRPO_ROOT / "hf_export" / "step_100"
    step100_weights = list(partial_step100.glob("*.safetensors"))
    step100_weight_bytes = sum(path.stat().st_size for path in step100_weights)
    step100_complete = (
        (partial_step100 / "config.json").is_file()
        and len(step100_weights) >= 4
        and step100_weight_bytes >= 15_000_000_000
    )
    if partial_step100.exists() and not step100_complete:
        shutil.rmtree(partial_step100)
    exporter.MODEL_NAME = str(BASE_MODEL)
    exporter.BASE_CHECKPOINT = CONVERTED_BASE
    exporter.SFT_ROOT = GRPO_ROOT
    exporter.STEPS = STEPS
    exporter.main(overrides)


if __name__ == "__main__":
    _args, unknown = _parse_args()
    main(unknown)
