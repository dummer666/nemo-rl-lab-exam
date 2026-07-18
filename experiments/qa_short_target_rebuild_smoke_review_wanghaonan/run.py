#!/usr/bin/env python
"""Review the latest strict short-target rebuild smoke artifacts."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

os.environ.setdefault(
    "QA_REBUILD_REVIEW_DIR",
    (
        "/shared/outputs/wanghaonan/"
        "qa_short_target_rebuild_smoke_wanghaonan/"
        "qa_short_target_rebuild_smoke_wanghaonan-wanghaonan-20260718-191219/"
        "short_target_rebuild"
    ),
)

ENTRY = (
    Path(__file__).resolve().parents[1]
    / "qa_short_target_rebuild_review_wanghaonan"
    / "run.py"
)
runpy.run_path(str(ENTRY), run_name="__main__")
