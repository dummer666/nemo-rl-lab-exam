#!/usr/bin/env python
"""Run the 20-step objective-heavy SFT behavior pilot."""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path

PACK_ROOT = Path(
    "/shared/outputs/wanghaonan/qa_objective_sft_data_wanghaonan/"
    "qa_objective_sft_data_wanghaonan-wanghaonan-20260719-103532/"
    "objective_sft_data"
)
FULL_RUN = (
    Path(__file__).resolve().parents[1]
    / "sft_qwen3.5-9b_qa-fill-v2_wanghaonan"
    / "run.py"
)

os.environ["QA_FILL_SFT_TRAIN_PATH"] = str(PACK_ROOT / "train.jsonl")
os.environ["QA_FILL_SFT_VALIDATION_PATH"] = str(
    PACK_ROOT / "validation.jsonl"
)
os.environ["QA_FILL_SFT_EXPECTED_PROFILES"] = json.dumps(
    {
        "train": {"0": 1600, "1": 323, "2": 77},
        "validation": {"0": 275, "1": 14, "2": 3},
    }
)
runpy.run_path(str(FULL_RUN), run_name="__main__")
