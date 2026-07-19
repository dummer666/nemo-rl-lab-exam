#!/usr/bin/env python
"""Review reward changes from pilot step 20 to full SFT step 200."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.qa_objective_sft_delta_review_wanghaonan import run as review  # noqa: E402

GATE_ROOT = Path(
    "/shared/outputs/wanghaonan/qa_objective_full_sft_gate_eval_wanghaonan/"
    "qa_objective_full_sft_gate_eval_wanghaonan-wanghaonan-20260719-125930/"
    "objective_sft_gate_eval"
)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def main(overrides: Sequence[str]) -> None:
    review.GATE_ROOT = GATE_ROOT
    review.STEPS = (200,)
    review.main(overrides)


if __name__ == "__main__":
    _args, unknown = _parse_args()
    main(unknown)
