#!/usr/bin/env python
"""Run the three-epoch, two-hop-oversampled fill SFT."""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path

TRAIN_PATH = Path(
    "/shared/outputs/wanghaonan/qa_fill_sft_v3_data_wanghaonan/"
    "qa_fill_sft_v3_data_wanghaonan-wanghaonan-20260719-042650/"
    "fill_sft_v3_data/fill_sft_v3_train.jsonl"
)
VALIDATION_PATH = TRAIN_PATH.with_name("fill_sft_v3_validation.jsonl")
FULL_RUN = (
    Path(__file__).resolve().parents[1]
    / "sft_qwen3.5-9b_qa-fill-v2_wanghaonan"
    / "run.py"
)

os.environ["QA_FILL_SFT_TRAIN_PATH"] = str(TRAIN_PATH)
os.environ["QA_FILL_SFT_VALIDATION_PATH"] = str(VALIDATION_PATH)
os.environ["QA_FILL_SFT_EXPECTED_PROFILES"] = json.dumps(
    {
        "train": {"0": 24, "1": 31, "2": 28},
        "validation": {"0": 6, "1": 4, "2": 1},
    }
)
runpy.run_path(str(FULL_RUN), run_name="__main__")
