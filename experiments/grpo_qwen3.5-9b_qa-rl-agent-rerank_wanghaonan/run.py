#!/usr/bin/env python
"""Run the shared QA Agent entrypoint with the quality-reranked v2 config."""

from __future__ import annotations

import runpy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_ENTRY = REPO_ROOT / "experiments" / "grpo_qwen3.5-9b_qa-rl-agent_wanghaonan" / "run.py"

runpy.run_path(str(BASE_ENTRY), run_name="__main__")
