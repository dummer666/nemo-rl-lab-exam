#!/usr/bin/env python
"""Evaluate baseline and grounded cloze SFT checkpoints under one protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.qa_fill_sft_v3_multiturn_eval_wanghaonan import (  # noqa: E402
    run as comparison,
)
from experiments.qa_sft_multiturn_eval_wanghaonan import run as evaluator  # noqa: E402

RETRIEVAL_SFT_STEP50 = Path(
    "/shared/outputs/wanghaonan/sft_qwen3.5-9b_qa-retrieval-v1_wanghaonan/"
    "sft_qwen3.5-9b_qa-retrieval-v1_wanghaonan-wanghaonan-20260718-130828/"
    "hf_export/step_50"
)
SFT_ROOT = Path(
    "/shared/outputs/wanghaonan/sft_qwen3.5-9b_qa-grounded-cloze_wanghaonan/"
    "sft_qwen3.5-9b_qa-grounded-cloze_wanghaonan-wanghaonan-20260719-092744"
)
BASELINE_OUTPUT_STEP = 0
BASELINE_MODEL_STEP = 50
CANDIDATE_STEPS = (31, 62, 94)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def main(overrides: Sequence[str]) -> None:
    evaluator.CHECKPOINTS = {
        BASELINE_OUTPUT_STEP: RETRIEVAL_SFT_STEP50,
        **{
            step: SFT_ROOT / "hf_export" / f"step_{step}"
            for step in CANDIDATE_STEPS
        },
    }
    evaluator.main()

    evaluation_root = evaluator._output_dir(overrides)
    baseline_summary, baseline_rows = comparison._load_evaluation(
        evaluation_root,
        BASELINE_OUTPUT_STEP,
    )
    loaded_candidates = {
        step: comparison._load_evaluation(evaluation_root, step)
        for step in CANDIDATE_STEPS
    }
    comparison.BASELINE_STEP = BASELINE_MODEL_STEP
    comparison.CANDIDATE_STEPS = CANDIDATE_STEPS
    report = comparison.build_comparison(
        baseline_summary,
        baseline_rows,
        {step: loaded_candidates[step][0] for step in CANDIDATE_STEPS},
        {step: loaded_candidates[step][1] for step in CANDIDATE_STEPS},
    )
    report["baseline"]["evaluation_output_step"] = BASELINE_OUTPUT_STEP
    report["decision"]["protocol"] = (
        "baseline and candidates evaluated together with current source exclusion"
    )
    report_path = evaluation_root / "baseline_comparison.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("[grounded-cloze-sft-eval]", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"[grounded-cloze-sft-eval] saved={report_path}", flush=True)


if __name__ == "__main__":
    _args, unknown = _parse_args()
    main(unknown)
