#!/usr/bin/env python
"""Audit whether stronger fill SFT learned useful or merely habitual second hops."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.retrieval.evidence import (  # noqa: E402
    expected_keypoints,
    normalize_evidence_text,
    visible_evidence_keypoint_hits,
)
from common.retrieval.markdown_bm25 import (  # noqa: E402
    MarkdownBM25Index,
    build_retrieval_query,
    format_search_results_with_visible_snippets,
)
from common.rewards.qa_reward import extract_boxed, qa_rule_reward_fn  # noqa: E402

BASELINE_ROOT = Path(
    "/shared/outputs/wanghaonan/qa_fill_sft_multiturn_eval_wanghaonan/"
    "qa_fill_sft_multiturn_eval_wanghaonan-wanghaonan-20260719-033956/"
    "sft_multiturn_eval"
)
CANDIDATE_ROOT = Path(
    "/shared/outputs/wanghaonan/qa_fill_sft_v3_multiturn_eval_wanghaonan/"
    "qa_fill_sft_v3_multiturn_eval_wanghaonan-wanghaonan-20260719-043947/"
    "sft_multiturn_eval"
)
DOCS_ROOT = Path("/data/docs")
BASELINE_STEP = 50
CANDIDATE_STEPS = (20, 40, 60)
EXPECTED_TWO_SEARCH_COUNTS = {20: 4, 40: 22, 60: 21}
_SOURCE = re.compile(r"^\d+\.\s+来源：([^\n]+)", re.MULTILINE)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def _output_dir(overrides: Sequence[str]) -> Path:
    for override in overrides:
        if override.startswith("logger.log_dir="):
            output = Path(override.split("=", 1)[1]).parent / "trajectory_audit"
            output.mkdir(parents=True, exist_ok=True)
            return output
    output = THIS_DIR / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_step(root: Path, step: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    step_root = root / f"step_{step}"
    summary_path = step_root / "summary.json"
    rows_path = step_root / "trajectories.jsonl"
    missing = [
        str(path)
        for path in (summary_path, rows_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing evaluation artifacts: {missing}")
    return _read_json(summary_path), _read_jsonl(rows_path)


def _indexed(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    result = {int(row["row_index"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate row_index in trajectories")
    return result


def _final_response(row: Mapping[str, Any]) -> str:
    responses = [str(response) for response in row.get("assistant_responses", [])]
    for response in reversed(responses):
        if extract_boxed(response) is not None:
            return response
    return responses[-1] if responses else ""


def _sources(rendered: str) -> list[str]:
    sources = []
    for match in _SOURCE.finditer(rendered):
        source = match.group(1).strip().split(" · ", 1)[0].strip()
        if source and source not in sources:
            sources.append(source)
    return sources


def _ngrams(text: str, width: int = 2) -> set[str]:
    normalized = normalize_evidence_text(text)
    if len(normalized) <= width:
        return {normalized} if normalized else set()
    return {
        normalized[index : index + width]
        for index in range(len(normalized) - width + 1)
    }


def query_similarity(first: str, second: str) -> float:
    left, right = _ngrams(first), _ngrams(second)
    return len(left & right) / len(left | right) if left or right else 1.0


def fragile_keypoint_indexes(
    keypoints: Sequence[Sequence[str]],
) -> set[int]:
    """Flag one-character or one-digit keypoints that are unsafe substring evidence."""
    return {
        index
        for index, alternatives in enumerate(keypoints)
        if max((len(alternative) for alternative in alternatives), default=0) <= 1
    }


def classify_two_hop(record: Mapping[str, Any]) -> str:
    if float(record["reward_delta"]) > 1e-9:
        return "useful_score_gain"
    if float(record["reward_delta"]) < -1e-9:
        return "reward_regression"
    evidence = record["evidence"]
    if evidence["full_after_two_hops"] and not record["candidate"]["perfect"]:
        return "synthesis_failure_after_full_evidence"
    if evidence["first_hop_full"]:
        return "unnecessary_second_search"
    if evidence["incremental_keypoint_hits"]:
        return "incremental_evidence_not_used"
    if evidence["new_sources"]:
        return "off_target_second_search"
    return "redundant_second_search"


def _replay_hop(
    index: MarkdownBM25Index,
    row: Mapping[str, Any],
    search_query: str,
) -> dict[str, Any]:
    retrieval_query = build_retrieval_query(
        search_query,
        str(row["query"]),
        str(row.get("bank", "")),
    )
    results = index.search(
        retrieval_query,
        top_k=4,
        candidate_k=50,
        quality_rerank=True,
    )
    rendered, visible_snippets = format_search_results_with_visible_snippets(
        results,
        retrieval_query,
        max_chars=1800,
        per_result_chars=360,
    )
    _question_type, keypoints = expected_keypoints(str(row["expected_answer"]))
    trusted_hits = visible_evidence_keypoint_hits(
        results,
        visible_snippets,
        keypoints,
    )
    return {
        "rendered": rendered,
        "sources": _sources(rendered),
        "quality_categories": [result.quality_category for result in results],
        "trusted_keypoint_hits": sorted(trusted_hits),
    }


def _verify_reward(row: Mapping[str, Any]) -> None:
    recomputed = float(
        qa_rule_reward_fn(
            [str(row["query"])],
            [_final_response(row)],
            [str(row["expected_answer"])],
        )[0]
    )
    stored = float(row["reward"])
    if abs(recomputed - stored) > 1e-9:
        raise RuntimeError(
            f"row {row['row_index']}: stored reward {stored} != recomputed {recomputed}"
        )


def _audit_two_hop(
    index: MarkdownBM25Index,
    step: int,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    queries = [str(query) for query in candidate.get("search_queries", [])]
    observations = [
        str(observation)
        for observation in candidate.get("environment_observations", [])
        if str(observation).startswith("[检索结果]")
    ]
    if len(queries) != 2 or len(observations) != 2:
        raise RuntimeError(
            f"step {step} row {candidate['row_index']}: expected two queries and observations"
        )

    hops = [
        _replay_hop(index, candidate, search_query)
        for search_query in queries
    ]
    replay_matches = [
        observation.startswith(hop["rendered"])
        for observation, hop in zip(observations, hops, strict=True)
    ]
    if not all(replay_matches):
        raise RuntimeError(
            f"step {step} row {candidate['row_index']}: BM25 replay mismatch"
        )

    _question_type, keypoints = expected_keypoints(
        str(candidate["expected_answer"])
    )
    fragile_hits = fragile_keypoint_indexes(keypoints)
    raw_first_hits = set(hops[0]["trusted_keypoint_hits"])
    raw_second_hits = set(hops[1]["trusted_keypoint_hits"])
    first_hits = raw_first_hits - fragile_hits
    second_hits = raw_second_hits - fragile_hits
    cumulative_hits = first_hits | second_hits
    keypoint_count = len(keypoints)
    robust_keypoint_count = keypoint_count - len(fragile_hits)
    first_sources = set(hops[0]["sources"])
    second_sources = set(hops[1]["sources"])
    candidate_response = _final_response(candidate)
    baseline_response = _final_response(baseline)
    reward_delta = float(candidate["reward"]) - float(baseline["reward"])
    record = {
        "step": step,
        "row_index": int(candidate["row_index"]),
        "question_type": str(candidate["question_type"]),
        "query": str(candidate["query"]),
        "expected_answer": str(candidate["expected_answer"]),
        "baseline": {
            "reward": float(baseline["reward"]),
            "search_count": int(baseline["search_count"]),
            "search_queries": list(baseline.get("search_queries", [])),
            "boxed_answer": extract_boxed(baseline_response),
            "assistant_responses": list(baseline.get("assistant_responses", [])),
        },
        "candidate": {
            "reward": float(candidate["reward"]),
            "perfect": float(candidate["reward"]) >= 1.0,
            "boxed_answer": extract_boxed(candidate_response),
            "assistant_responses": list(candidate.get("assistant_responses", [])),
        },
        "reward_delta": reward_delta,
        "answer_changed": (
            extract_boxed(candidate_response) != extract_boxed(baseline_response)
        ),
        "search": {
            "queries": queries,
            "exact_duplicate": (
                normalize_evidence_text(queries[0])
                == normalize_evidence_text(queries[1])
            ),
            "query_similarity": query_similarity(*queries),
            "first_sources": hops[0]["sources"],
            "second_sources": hops[1]["sources"],
            "new_sources": sorted(second_sources - first_sources),
            "source_overlap": sorted(first_sources & second_sources),
            "quality_categories": [
                hops[0]["quality_categories"],
                hops[1]["quality_categories"],
            ],
            "deterministic_replay_matches": replay_matches,
        },
        "evidence": {
            "keypoint_count": keypoint_count,
            "robust_keypoint_count": robust_keypoint_count,
            "fragile_keypoint_indexes": sorted(fragile_hits),
            "raw_first_keypoint_hits": sorted(raw_first_hits),
            "raw_second_keypoint_hits": sorted(raw_second_hits),
            "raw_incremental_keypoint_hits": sorted(
                raw_second_hits - raw_first_hits
            ),
            "first_keypoint_hits": sorted(first_hits),
            "second_keypoint_hits": sorted(second_hits),
            "incremental_keypoint_hits": sorted(second_hits - first_hits),
            "cumulative_keypoint_hits": sorted(cumulative_hits),
            "first_hop_full": bool(
                robust_keypoint_count
                and len(first_hits) == robust_keypoint_count
            ),
            "second_made_full": bool(
                robust_keypoint_count
                and len(first_hits) < robust_keypoint_count
                and len(cumulative_hits) == robust_keypoint_count
            ),
            "full_after_two_hops": bool(
                robust_keypoint_count
                and len(cumulative_hits) == robust_keypoint_count
            ),
            "new_sources": sorted(second_sources - first_sources),
        },
    }
    record["primary_diagnosis"] = classify_two_hop(record)
    return record


def _step_summary(
    step: int,
    summary: Mapping[str, Any],
    baseline: Mapping[int, Mapping[str, Any]],
    candidate: Mapping[int, Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reward_changes = []
    change_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row_index, row in candidate.items():
        before = baseline[row_index]
        delta = float(row["reward"]) - float(before["reward"])
        if abs(delta) <= 1e-9:
            continue
        direction = "gain" if delta > 0 else "loss"
        question_type = str(row["question_type"])
        change_counts[question_type][direction] += 1
        reward_changes.append(
            {
                "step": step,
                "row_index": row_index,
                "question_type": question_type,
                "query": str(row["query"]),
                "expected_answer": str(row["expected_answer"]),
                "baseline_reward": float(before["reward"]),
                "candidate_reward": float(row["reward"]),
                "reward_delta": delta,
                "baseline_search_count": int(before["search_count"]),
                "candidate_search_count": int(row["search_count"]),
                "baseline_boxed_answer": extract_boxed(_final_response(before)),
                "candidate_boxed_answer": extract_boxed(_final_response(row)),
                "candidate_search_queries": list(row.get("search_queries", [])),
            }
        )

    diagnoses = Counter(str(record["primary_diagnosis"]) for record in records)
    by_type = Counter(str(record["question_type"]) for record in records)
    baseline_rewards = [float(record["baseline"]["reward"]) for record in records]
    candidate_rewards = [float(record["candidate"]["reward"]) for record in records]
    audit = {
        "two_hop_count": len(records),
        "two_hop_by_type": dict(sorted(by_type.items())),
        "exact_duplicate_query_count": sum(
            bool(record["search"]["exact_duplicate"]) for record in records
        ),
        "high_query_similarity_count": sum(
            float(record["search"]["query_similarity"]) >= 0.8
            for record in records
        ),
        "second_hop_new_source_count": sum(
            bool(record["search"]["new_sources"]) for record in records
        ),
        "second_hop_incremental_keypoint_count": sum(
            bool(record["evidence"]["incremental_keypoint_hits"])
            for record in records
        ),
        "second_hop_raw_incremental_keypoint_count": sum(
            bool(record["evidence"]["raw_incremental_keypoint_hits"])
            for record in records
        ),
        "fragile_keypoint_record_count": sum(
            bool(record["evidence"]["fragile_keypoint_indexes"])
            for record in records
        ),
        "first_hop_already_full_count": sum(
            bool(record["evidence"]["first_hop_full"]) for record in records
        ),
        "second_hop_made_full_count": sum(
            bool(record["evidence"]["second_made_full"]) for record in records
        ),
        "full_evidence_after_two_hops_count": sum(
            bool(record["evidence"]["full_after_two_hops"])
            for record in records
        ),
        "full_evidence_but_not_perfect_count": sum(
            bool(record["evidence"]["full_after_two_hops"])
            and not bool(record["candidate"]["perfect"])
            for record in records
        ),
        "answer_changed_count": sum(
            bool(record["answer_changed"]) for record in records
        ),
        "reward_gain_count": sum(
            float(record["reward_delta"]) > 1e-9 for record in records
        ),
        "reward_loss_count": sum(
            float(record["reward_delta"]) < -1e-9 for record in records
        ),
        "reward_unchanged_count": sum(
            abs(float(record["reward_delta"])) <= 1e-9 for record in records
        ),
        "mean_baseline_reward": mean(baseline_rewards) if baseline_rewards else 0.0,
        "mean_candidate_reward": mean(candidate_rewards) if candidate_rewards else 0.0,
        "primary_diagnoses": dict(sorted(diagnoses.items())),
        "all_reward_changes_by_type": {
            question_type: dict(sorted(counts.items()))
            for question_type, counts in sorted(change_counts.items())
        },
        "protocol_error_count": int(summary["protocol"]["error_count"]),
    }
    return audit, reward_changes


def main() -> None:
    _, overrides = _parse_args()
    output_dir = _output_dir(overrides)
    baseline_summary, baseline_rows = _load_step(BASELINE_ROOT, BASELINE_STEP)
    candidate_data = {
        step: _load_step(CANDIDATE_ROOT, step)
        for step in CANDIDATE_STEPS
    }
    baseline = _indexed(baseline_rows)
    if len(baseline) != 313:
        raise RuntimeError(f"expected 313 baseline rows, got {len(baseline)}")
    for row in baseline.values():
        _verify_reward(row)

    index_start = time.perf_counter()
    index = MarkdownBM25Index.from_directory(
        DOCS_ROOT,
        chunk_chars=1200,
        overlap_chars=160,
        k1=1.5,
        b=0.75,
    )
    index_seconds = time.perf_counter() - index_start

    all_records = []
    all_reward_changes = []
    step_summaries = {}
    two_hop_sets = {}
    for step in CANDIDATE_STEPS:
        summary, rows = candidate_data[step]
        candidate = _indexed(rows)
        if set(candidate) != set(baseline):
            raise RuntimeError(f"step {step}: validation row set differs from baseline")
        if int(summary["protocol"]["error_count"]) != 0:
            raise RuntimeError(f"step {step}: protocol errors present")
        for row in candidate.values():
            _verify_reward(row)

        two_hop_rows = [
            row for row in candidate.values()
            if int(row["search_count"]) == 2
        ]
        expected_count = EXPECTED_TWO_SEARCH_COUNTS[step]
        if len(two_hop_rows) != expected_count:
            raise RuntimeError(
                f"step {step}: expected {expected_count} two-hop rows, got {len(two_hop_rows)}"
            )
        records = [
            _audit_two_hop(index, step, baseline[int(row["row_index"])], row)
            for row in two_hop_rows
        ]
        audit, reward_changes = _step_summary(
            step,
            summary,
            baseline,
            candidate,
            records,
        )
        step_summaries[str(step)] = audit
        two_hop_sets[step] = {int(row["row_index"]) for row in two_hop_rows}
        all_records.extend(records)
        all_reward_changes.extend(reward_changes)

    union = set().union(*two_hop_sets.values())
    intersection = set.intersection(*two_hop_sets.values())
    useful_gains = [
        record for record in all_records
        if record["primary_diagnosis"] == "useful_score_gain"
    ]
    evidence_not_used = [
        record for record in all_records
        if record["primary_diagnosis"]
        in {
            "synthesis_failure_after_full_evidence",
            "incremental_evidence_not_used",
        }
    ]
    summary = {
        "mode": "read_only_stronger_fill_two_hop_trajectory_audit",
        "sources": {
            "baseline_root": str(BASELINE_ROOT),
            "candidate_root": str(CANDIDATE_ROOT),
            "docs_root": str(DOCS_ROOT),
        },
        "baseline": {
            "step": BASELINE_STEP,
            "accuracy": baseline_summary["accuracy"],
            "fill": baseline_summary["question_types"]["fill"],
            "short": baseline_summary["question_types"]["short"],
            "two_search_count": baseline_summary["retrieval"]["two_search_count"],
        },
        "index": {
            "num_documents": index.num_documents,
            "build_seconds": index_seconds,
            "quality_category_counts": index.quality_category_counts,
        },
        "steps": step_summaries,
        "cross_step": {
            "unique_two_hop_row_count": len(union),
            "two_hop_in_all_steps_count": len(intersection),
            "two_hop_row_indexes_by_step": {
                str(step): sorted(indexes)
                for step, indexes in two_hop_sets.items()
            },
            "two_hop_in_all_steps_row_indexes": sorted(intersection),
        },
        "decision": {
            "additional_plain_epochs_authorized": False,
            "grpo_authorized": False,
            "useful_score_gain_record_count": len(useful_gains),
            "incremental_or_full_evidence_not_used_record_count": len(
                evidence_not_used
            ),
            "next_intervention": (
                "Use only audited incremental two-hop rows, add a stop-after-sufficient-"
                "evidence contrast, and supervise evidence-to-boxed synthesis; do not "
                "reward second-search count itself."
            ),
            "short_semantic_conclusions_allowed": False,
            "reason": (
                "The platform Judge is not injected; short rewards remain lexical. "
                "Single-character and single-digit substring hits are marked fragile "
                "instead of being treated as complete evidence. No candidate improved "
                "fill accuracy, so GRPO promotion remains closed."
            ),
        },
        "human_reviewed": False,
        "outputs": {
            "two_hop_rows": str(output_dir / "two_hop_rows.jsonl"),
            "reward_changes": str(output_dir / "reward_changes.jsonl"),
            "review_queue": str(output_dir / "review_queue.jsonl"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    review_queue = sorted(
        all_records,
        key=lambda record: (
            0 if record["primary_diagnosis"] == "reward_regression" else 1,
            0 if record["evidence"]["full_after_two_hops"] else 1,
            int(record["step"]),
            int(record["row_index"]),
        ),
    )
    _write_jsonl(output_dir / "two_hop_rows.jsonl", all_records)
    _write_jsonl(output_dir / "reward_changes.jsonl", all_reward_changes)
    _write_jsonl(output_dir / "review_queue.jsonl", review_queue)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("[fill-sft-v3-trajectory-audit]", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
