#!/usr/bin/env python
from __future__ import annotations

import runpy
from pathlib import Path

FULL_RUN = (
    Path(__file__).resolve().parents[1]
    / "sft_qwen3.5-9b_qa-fill-v3_wanghaonan"
    / "run.py"
)
runpy.run_path(str(FULL_RUN), run_name="__main__")
