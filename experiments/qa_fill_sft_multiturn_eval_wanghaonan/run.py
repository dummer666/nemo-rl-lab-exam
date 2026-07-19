#!/usr/bin/env python
"""Compare retrieval-SFT step 50 and fill-SFT step 13 on all 313 questions."""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.qa_sft_multiturn_eval_wanghaonan import run as evaluator  # noqa: E402

RETRIEVAL_SFT_STEP50 = Path(
    "/shared/outputs/wanghaonan/sft_qwen3.5-9b_qa-retrieval-v1_wanghaonan/"
    "sft_qwen3.5-9b_qa-retrieval-v1_wanghaonan-wanghaonan-20260718-130828/"
    "hf_export/step_50"
)
FILL_SFT_STEP13 = Path(
    "/shared/outputs/wanghaonan/sft_qwen3.5-9b_qa-fill-v2_wanghaonan/"
    "sft_qwen3.5-9b_qa-fill-v2_wanghaonan-wanghaonan-20260719-033321/"
    "hf_export/step_13"
)


if __name__ == "__main__":
    evaluator.CHECKPOINTS = {
        50: RETRIEVAL_SFT_STEP50,
        13: FILL_SFT_STEP13,
    }
    evaluator.main()
