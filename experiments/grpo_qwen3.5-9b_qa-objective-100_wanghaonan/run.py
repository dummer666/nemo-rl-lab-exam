#!/usr/bin/env python
"""Run conservative 100-step objective GRPO from the selected pilot model."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    source = (
        Path(__file__).resolve().parent.parent
        / "grpo_qwen3.5-9b_qa-objective-short_wanghaonan"
        / "run.py"
    )
    runpy.run_path(str(source), run_name="__main__")
