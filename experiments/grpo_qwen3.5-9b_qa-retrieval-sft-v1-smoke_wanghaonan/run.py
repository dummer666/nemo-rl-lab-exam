#!/usr/bin/env python
"""Run the one-step preflight for post-SFT retrieval GRPO."""

from __future__ import annotations

import runpy
from pathlib import Path

FULL_EXPERIMENT = (
    Path(__file__).resolve().parents[1] / "grpo_qwen3.5-9b_qa-retrieval-sft-v1-short_wanghaonan" / "run.py"
)

if __name__ == "__main__":
    runpy.run_path(str(FULL_EXPERIMENT), run_name="__main__")
