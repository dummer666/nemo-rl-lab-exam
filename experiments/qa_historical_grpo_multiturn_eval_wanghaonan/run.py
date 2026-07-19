#!/usr/bin/env python
"""Evaluate historical GRPO checkpoints under the current QA protocol."""

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
GRPO_ROOT = Path(
    "/shared/outputs/wanghaonan/grpo_qwen3.5-9b_qa-rl-agent_wanghaonan/"
    "grpo_qwen3.5-9b_qa-rl-agent_wanghaonan-wanghaonan-20260718-043801"
)


def main() -> None:
    evaluator.CHECKPOINTS = {
        0: RETRIEVAL_SFT_STEP50,
        **{
            step: GRPO_ROOT / "hf_export" / f"step_{step}"
            for step in (40, 100, 120)
        },
    }
    evaluator.main()


if __name__ == "__main__":
    main()
