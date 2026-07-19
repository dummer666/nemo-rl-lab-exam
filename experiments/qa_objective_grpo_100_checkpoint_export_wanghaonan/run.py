#!/usr/bin/env python
"""Export the leading checkpoints from the 100-step objective GRPO run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.qa_objective_grpo_checkpoint_export_wanghaonan import (  # noqa: E402
    run as exporter,
)

GRPO_ROOT = Path(
    "/shared/outputs/wanghaonan/grpo_qwen3.5-9b_qa-objective-100_wanghaonan/"
    "grpo_qwen3.5-9b_qa-objective-100_wanghaonan-wanghaonan-20260719-141205"
)
STEPS = (70, 80, 100)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def main(overrides: list[str]) -> None:
    exporter.GRPO_ROOT = GRPO_ROOT
    exporter.STEPS = STEPS
    exporter.main(overrides)


if __name__ == "__main__":
    _args, unknown = _parse_args()
    main(unknown)
