#!/usr/bin/env python
"""Gate objective SFT pilots on internal holdout and official validation."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.retrieval.markdown_bm25 import MarkdownBM25Index  # noqa: E402
from experiments.qa_sft_multiturn_eval_wanghaonan import run as evaluator  # noqa: E402

RETRIEVAL_SFT_STEP50 = Path(
    "/shared/outputs/wanghaonan/sft_qwen3.5-9b_qa-retrieval-v1_wanghaonan/"
    "sft_qwen3.5-9b_qa-retrieval-v1_wanghaonan-wanghaonan-20260718-130828/"
    "hf_export/step_50"
)
SFT_ROOT = Path(
    "/shared/outputs/wanghaonan/sft_qwen3.5-9b_qa-objective-pilot_wanghaonan/"
    "sft_qwen3.5-9b_qa-objective-pilot_wanghaonan-wanghaonan-20260719-105248"
)
PACK_ROOT = Path(
    "/shared/outputs/wanghaonan/qa_objective_sft_data_wanghaonan/"
    "qa_objective_sft_data_wanghaonan-wanghaonan-20260719-103532/"
    "objective_sft_data"
)
CHECKPOINTS = {
    0: RETRIEVAL_SFT_STEP50,
    10: SFT_ROOT / "hf_export" / "step_10",
    20: SFT_ROOT / "hf_export" / "step_20",
}
INTERNAL_PATH = PACK_ROOT / "objective_validation_manifest.jsonl"
OFFICIAL_PATH = Path("/data/datasets/qa_rl/val.jsonl")
DOCS_PATH = Path("/data/docs")
MIN_INTERNAL_GAIN = 3
MIN_OFFICIAL_OBJECTIVE_GAIN = 1
MAX_OPEN_REWARD_LOSS = 0.5


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def _output_dir(overrides: Sequence[str]) -> Path:
    for override in overrides:
        if override.startswith("logger.log_dir="):
            output = (
                Path(override.split("=", 1)[1]).parent
                / "objective_sft_gate_eval"
            )
            output.mkdir(parents=True, exist_ok=True)
            return output
    output = THIS_DIR / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _objective_block(summary: dict[str, Any]) -> dict[str, float | int]:
    groups = [
        summary["question_types"].get(question_type, {})
        for question_type in ("single", "multiple", "bool")
    ]
    count = sum(int(group.get("count", 0)) for group in groups)
    reward_sum = sum(
        float(group.get("accuracy", 0.0)) * int(group.get("count", 0))
        for group in groups
    )
    perfect_count = sum(int(group.get("perfect_count", 0)) for group in groups)
    return {
        "count": count,
        "accuracy": reward_sum / count if count else 0.0,
        "reward_sum": reward_sum,
        "perfect_count": perfect_count,
    }


def build_gate_report(
    summaries: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    baseline = summaries[0]
    baseline_internal = _objective_block(baseline["internal"])
    baseline_official_objective = _objective_block(baseline["official"])
    baseline_open = baseline["official"]["open_questions"]
    candidates = {}
    passing_steps = []
    for step in sorted(set(summaries) - {0}):
        candidate = summaries[step]
        internal = _objective_block(candidate["internal"])
        official_objective = _objective_block(candidate["official"])
        official_open = candidate["official"]["open_questions"]
        internal_gain = (
            int(internal["perfect_count"])
            - int(baseline_internal["perfect_count"])
        )
        official_objective_gain = (
            int(official_objective["perfect_count"])
            - int(baseline_official_objective["perfect_count"])
        )
        open_reward_loss = (
            float(baseline_open["accuracy"]) * int(baseline_open["count"])
            - float(official_open["accuracy"]) * int(official_open["count"])
        )
        criteria = {
            "internal_gain_at_least_3": internal_gain >= MIN_INTERNAL_GAIN,
            "official_objective_gain_at_least_1": (
                official_objective_gain >= MIN_OFFICIAL_OBJECTIVE_GAIN
            ),
            "official_overall_non_decrease": (
                float(candidate["official"]["accuracy"])
                >= float(baseline["official"]["accuracy"])
            ),
            "open_reward_loss_at_most_0_5": (
                open_reward_loss <= MAX_OPEN_REWARD_LOSS
            ),
            "protocol_errors_not_increased": (
                int(candidate["official"]["protocol"]["error_count"])
                <= int(baseline["official"]["protocol"]["error_count"])
            ),
        }
        passed = all(criteria.values())
        if passed:
            passing_steps.append(step)
        candidates[str(step)] = {
            "internal": internal,
            "official_objective": official_objective,
            "official_overall": candidate["official"]["accuracy"],
            "official_open": official_open,
            "internal_perfect_gain": internal_gain,
            "official_objective_perfect_gain": official_objective_gain,
            "open_reward_loss": open_reward_loss,
            "criteria": criteria,
            "passed": passed,
        }
    selected = (
        max(
            passing_steps,
            key=lambda step: (
                float(summaries[step]["official"]["accuracy"]),
                int(
                    _objective_block(
                        summaries[step]["internal"]
                    )["perfect_count"]
                ),
            ),
        )
        if passing_steps
        else None
    )
    return {
        "thresholds": {
            "minimum_internal_perfect_gain": MIN_INTERNAL_GAIN,
            "minimum_official_objective_perfect_gain": (
                MIN_OFFICIAL_OBJECTIVE_GAIN
            ),
            "maximum_open_reward_loss": MAX_OPEN_REWARD_LOSS,
        },
        "baseline": {
            "internal": baseline_internal,
            "official_objective": baseline_official_objective,
            "official_overall": baseline["official"]["accuracy"],
            "official_open": baseline_open,
            "protocol": baseline["official"]["protocol"],
        },
        "candidates": candidates,
        "decision": {
            "promote_to_full_sft": selected is not None,
            "selected_step": selected,
            "reason": (
                "candidate passed every predeclared behavior gate"
                if selected is not None
                else "no candidate passed every predeclared behavior gate"
            ),
        },
    }


def _evaluate(
    model,
    tokenizer,
    rows: list[dict],
    index: MarkdownBM25Index,
    *,
    label: str,
) -> tuple[list[dict], dict[str, Any]]:
    started = time.perf_counter()
    results = evaluator._run_rollouts(model, tokenizer, rows, index)
    return results, {
        "dataset": label,
        "evaluation_seconds": time.perf_counter() - started,
        **evaluator._summarize(results),
    }


def main(overrides: Sequence[str]) -> None:
    output_dir = _output_dir(overrides)
    required = [
        INTERNAL_PATH,
        OFFICIAL_PATH,
        DOCS_PATH,
        *CHECKPOINTS.values(),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing objective gate inputs: {missing}")
    datasets = {
        "internal": evaluator._read_jsonl(INTERNAL_PATH),
        "official": evaluator._read_jsonl(OFFICIAL_PATH),
    }
    if len(datasets["internal"]) != 275 or len(datasets["official"]) != 313:
        raise RuntimeError(
            "evaluation population changed: "
            f"internal={len(datasets['internal'])} "
            f"official={len(datasets['official'])}"
        )
    index_started = time.perf_counter()
    index = MarkdownBM25Index.from_directory(
        DOCS_PATH,
        chunk_chars=1200,
        overlap_chars=160,
        k1=1.5,
        b=0.75,
    )
    print(
        f"[objective-gate] indexed {index.num_documents} chunks in "
        f"{time.perf_counter() - index_started:.1f}s",
        flush=True,
    )

    summaries: dict[int, dict[str, dict[str, Any]]] = {}
    for step, model_path in CHECKPOINTS.items():
        tokenizer, model = evaluator._load_model(model_path)
        summaries[step] = {}
        step_dir = output_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        for label, rows in datasets.items():
            results, summary = _evaluate(
                model,
                tokenizer,
                rows,
                index,
                label=label,
            )
            summary["step"] = step
            summary["model_path"] = str(model_path)
            summaries[step][label] = summary
            evaluator._write_jsonl(
                step_dir / f"{label}_trajectories.jsonl",
                results,
            )
            (step_dir / f"{label}_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            objective = _objective_block(summary)
            print(
                f"[objective-gate] step={step} dataset={label} "
                f"accuracy={summary['accuracy']:.4f} "
                f"objective_perfect={objective['perfect_count']}/"
                f"{objective['count']} "
                f"protocol_errors={summary['protocol']['error_count']}",
                flush=True,
            )
        del model, tokenizer
        gc.collect()
        import torch

        torch.cuda.empty_cache()

    report = build_gate_report(summaries)
    report["paths"] = {
        "internal": str(INTERNAL_PATH),
        "official": str(OFFICIAL_PATH),
        "docs": str(DOCS_PATH),
    }
    report_path = output_dir / "gate_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("[objective-gate-report]", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"[objective-gate] saved={report_path}", flush=True)


if __name__ == "__main__":
    _args, unknown = _parse_args()
    main(unknown)
