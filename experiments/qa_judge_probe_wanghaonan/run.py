#!/usr/bin/env python
"""Probe the platform Judge endpoint without loading a policy model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.rewards.qa_judge_reward import probe_judge_endpoint  # noqa: E402


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def _output_dir(overrides: Sequence[str]) -> Path:
    for override in overrides:
        if override.startswith("logger.log_dir="):
            output = Path(override.split("=", 1)[1]).parent / "judge_probe"
            output.mkdir(parents=True, exist_ok=True)
            return output
    output = THIS_DIR / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _assert_report_is_sanitized(report: dict) -> None:
    serialized = json.dumps(report, ensure_ascii=False)
    for name in ("JUDGE_BASE_URL", "JUDGE_API_KEY"):
        secret = os.environ.get(name, "")
        if secret and secret in serialized:
            raise RuntimeError(f"judge probe report exposed {name}")


def main() -> None:
    _, overrides = _parse_args()
    output_dir = _output_dir(overrides)
    report = probe_judge_endpoint()
    _assert_report_is_sanitized(report)
    report_path = output_dir / "judge_probe.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "[judge-probe] "
        + json.dumps(report, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    print(f"[judge-probe] report={report_path}", flush=True)


if __name__ == "__main__":
    main()
