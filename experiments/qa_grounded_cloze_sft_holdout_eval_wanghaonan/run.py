#!/usr/bin/env python
"""Evaluate grounded cloze checkpoints on the source-isolated holdout."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.retrieval.markdown_bm25 import MarkdownBM25Index  # noqa: E402
from experiments.qa_sft_multiturn_eval_wanghaonan import run as evaluator  # noqa: E402

HOLDOUT_PATH = Path(
    "/shared/outputs/wanghaonan/qa_grounded_cloze_semantic_refine_wanghaonan/"
    "qa_grounded_cloze_semantic_refine_wanghaonan-wanghaonan-20260719-085554/"
    "grounded_cloze_semantic/validation.jsonl"
)
RETRIEVAL_SFT_STEP50 = Path(
    "/shared/outputs/wanghaonan/sft_qwen3.5-9b_qa-retrieval-v1_wanghaonan/"
    "sft_qwen3.5-9b_qa-retrieval-v1_wanghaonan-wanghaonan-20260718-130828/"
    "hf_export/step_50"
)
SFT_ROOT = Path(
    "/shared/outputs/wanghaonan/sft_qwen3.5-9b_qa-grounded-cloze_wanghaonan/"
    "sft_qwen3.5-9b_qa-grounded-cloze_wanghaonan-wanghaonan-20260719-092744"
)
CHECKPOINTS = {
    0: RETRIEVAL_SFT_STEP50,
    **{
        step: SFT_ROOT / "hf_export" / f"step_{step}"
        for step in (31, 62, 94)
    },
}


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def _output_dir(overrides: Sequence[str]) -> Path:
    for override in overrides:
        if override.startswith("logger.log_dir="):
            output = (
                Path(override.split("=", 1)[1]).parent
                / "grounded_cloze_holdout_eval"
            )
            output.mkdir(parents=True, exist_ok=True)
            return output
    output = THIS_DIR / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    return output


def expected_searches(row: Mapping[str, Any]) -> int:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("holdout row is missing messages")
    return sum(
        isinstance(message, Mapping) and message.get("role") == "environment"
        for message in messages
    )


def profile_summary(
    rows: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_profile: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        row_index = int(result["row_index"])
        by_profile[expected_searches(rows[row_index])].append(result)
    return {
        str(searches): {
            **evaluator._metric_block(group),
            "average_actual_searches": mean(
                int(result["search_count"]) for result in group
            ),
            "matched_search_count": sum(
                int(result["search_count"]) == searches for result in group
            ),
        }
        for searches, group in sorted(by_profile.items())
    }


def main(overrides: Sequence[str]) -> None:
    import torch

    output_dir = _output_dir(overrides)
    docs_dir = Path("/data/docs")
    required = [HOLDOUT_PATH, docs_dir, *CHECKPOINTS.values()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing holdout evaluation inputs: {missing}")

    rows = evaluator._read_jsonl(HOLDOUT_PATH)
    profiles = {
        searches: sum(expected_searches(row) == searches for row in rows)
        for searches in (0, 1, 2)
    }
    if profiles != {0: 6, 1: 20, 2: 20}:
        raise RuntimeError(f"holdout profile changed: {profiles}")
    print(f"[cloze-holdout] rows={len(rows)} profiles={profiles}", flush=True)

    index_start = time.perf_counter()
    index = MarkdownBM25Index.from_directory(
        docs_dir,
        chunk_chars=1200,
        overlap_chars=160,
        k1=1.5,
        b=0.75,
    )
    print(
        f"[cloze-holdout] indexed={index.num_documents} "
        f"seconds={time.perf_counter() - index_start:.1f}",
        flush=True,
    )

    summaries = {}
    for step, model_path in CHECKPOINTS.items():
        step_start = time.perf_counter()
        tokenizer, model = evaluator._load_model(model_path)
        results = evaluator._run_rollouts(model, tokenizer, rows, index)
        summary = {
            "step": step,
            "model_path": str(model_path),
            "validation_path": str(HOLDOUT_PATH),
            "profiles": profile_summary(rows, results),
            "evaluation_seconds": time.perf_counter() - step_start,
            **evaluator._summarize(results),
        }
        step_dir = output_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        evaluator._write_jsonl(step_dir / "trajectories.jsonl", results)
        (step_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summaries[str(step)] = summary
        print(
            f"[cloze-holdout] step={step} accuracy={summary['accuracy']:.4f} "
            f"two_searches={summary['retrieval']['two_search_count']} "
            f"profiles={summary['profiles']}",
            flush=True,
        )
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    comparison_path = output_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[cloze-holdout] saved={comparison_path}", flush=True)


if __name__ == "__main__":
    _args, unknown = _parse_args()
    main(unknown)
