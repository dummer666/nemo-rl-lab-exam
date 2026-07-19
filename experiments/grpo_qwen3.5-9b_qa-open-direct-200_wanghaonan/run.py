#!/usr/bin/env python
"""Run the final 200-step high-volume fill/short GRPO gamble."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    source = (
        Path(__file__).resolve().parent.parent
        / "grpo_qwen3.5-9b_qa-retrieval-sft-v1-short_wanghaonan"
        / "run.py"
    )
    runpy.run_path(str(source), run_name="__main__")
