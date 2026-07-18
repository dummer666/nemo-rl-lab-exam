#!/usr/bin/env python
"""Run the retrieval audit with the cached Qwen3.5-9B Base as an encoder."""

from __future__ import annotations

import runpy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_ENTRY = REPO_ROOT / "experiments" / "retrieval_qa_audit_wanghaonan" / "run.py"

runpy.run_path(str(AUDIT_ENTRY), run_name="__main__")
