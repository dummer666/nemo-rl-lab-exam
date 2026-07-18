#!/usr/bin/env python
"""Run a small positive-path smoke of the short target reconstruction."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

os.environ.setdefault("QA_REBUILD_MAX_ROWS", "12")
os.environ.setdefault("QA_REBUILD_MIN_ACCEPTED", "1")
os.environ.setdefault("QA_REBUILD_MIN_TWO_HOP", "0")
os.environ.setdefault("QA_REBUILD_ENFORCE_MIN", "1")
os.environ.setdefault("QA_REBUILD_REQUIRE_SPLITS", "0")

ENTRY = (
    Path(__file__).resolve().parents[1]
    / "qa_short_target_rebuild_wanghaonan"
    / "run.py"
)
runpy.run_path(str(ENTRY), run_name="__main__")
