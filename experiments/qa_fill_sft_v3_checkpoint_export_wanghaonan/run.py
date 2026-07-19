#!/usr/bin/env python
"""Merge the stronger fill-SFT checkpoints into Hugging Face models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.qa_sft_checkpoint_export_wanghaonan import run as exporter  # noqa: E402

MERGED_RETRIEVAL_SFT = Path(
    "/shared/outputs/wanghaonan/sft_qwen3.5-9b_qa-retrieval-v1_wanghaonan/"
    "sft_qwen3.5-9b_qa-retrieval-v1_wanghaonan-wanghaonan-20260718-130828/"
    "hf_export/step_50"
)
CONVERTED_RETRIEVAL_SFT = Path(
    "/data/huggingface/nemo_rl/"
    "model__shared_outputs_wanghaonan_sft_qwen3.5-9b_qa-retrieval-v1_wanghaonan_"
    "sft_qwen3.5-9b_qa-retrieval-v1_wanghaonan-wanghaonan-20260718-130828_"
    "hf_export_step_50/iter_0000000"
)
FILL_SFT_ROOT = Path(
    "/shared/outputs/wanghaonan/sft_qwen3.5-9b_qa-fill-v3_wanghaonan/"
    "sft_qwen3.5-9b_qa-fill-v3_wanghaonan-wanghaonan-20260719-043000"
)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def main(overrides: list[str]) -> None:
    exporter.MODEL_NAME = str(MERGED_RETRIEVAL_SFT)
    exporter.BASE_CHECKPOINT = CONVERTED_RETRIEVAL_SFT
    exporter.SFT_ROOT = FILL_SFT_ROOT
    exporter.STEPS = (20, 40, 60)
    exporter.main(overrides)


if __name__ == "__main__":
    _args, unknown = _parse_args()
    main(unknown)
