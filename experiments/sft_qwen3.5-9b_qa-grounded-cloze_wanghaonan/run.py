#!/usr/bin/env python
"""Train one epoch of grounded cloze SFT from retrieval-SFT step 50."""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path

PACK_ROOT = Path(
    "/shared/outputs/wanghaonan/qa_grounded_cloze_semantic_refine_wanghaonan/"
    "qa_grounded_cloze_semantic_refine_wanghaonan-wanghaonan-20260719-085554/"
    "grounded_cloze_semantic"
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
        "train": {"0": 96, "1": 160, "2": 120},
        "validation": {"0": 6, "1": 20, "2": 20},
    }
)
runpy.run_path(str(FULL_RUN), run_name="__main__")
