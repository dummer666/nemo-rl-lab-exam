#!/usr/bin/env python
"""Run a read-only, evidence-preserving audit of short labels and trajectories."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any, Sequence

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.retrieval.markdown_bm25 import (  # noqa: E402
    MarkdownBM25Index,
    build_retrieval_query,
    question_context,
)
from common.retrieval.qa_short_audit import (  # noqa: E402
    audit_short_label,
    audit_short_trace,
)

DEFAULT_CLEAN_TRAIN_PATH = Path(
    "/shared/outputs/wanghaonan/qa_training_clean_wanghaonan/"
    "qa_training_clean_wanghaonan-wanghaonan-20260718-092437/"
    "cleaned_data/clean_train.jsonl"
)
DEFAULT_SELECTION_PATH = Path(
    "/shared/outputs/wanghaonan/qa_sft_data_select_wanghaonan/"
    "qa_sft_data_select_wanghaonan-wanghaonan-20260718-111114/"
    "sft_selection/selection_manifest.jsonl"
)
DEFAULT_TRAJECTORY_PATH = Path(
    "/shared/outputs/wanghaonan/qa_sft_trajectory_build_wanghaonan/"
    "qa_sft_trajectory_build_wanghaonan-wanghaonan-20260718-123640/"
    "sft_trajectories/trajectory_manifest.jsonl"
)
DEFAULT_SFT_EVAL_PATH = Path(
    "/shared/outputs/wanghaonan/qa_sft_multiturn_eval_wanghaonan/"
    "qa_sft_multiturn_eval_wanghaonan-wanghaonan-20260718-141126/"
    "sft_multiturn_eval/step_50/trajectories.jsonl"
)
DEFAULT_GRPO_LOG_DIR = Path(
    "/shared/outputs/wanghaonan/grpo_qwen3.5-9b_qa-retrieval-sft-v1-short_wanghaonan/"
    "grpo_qwen3.5-9b_qa-retrieval-sft-v1-short_wanghaonan-wanghaonan-20260718-144929/"
    "logs/exp_001"
)
GRPO_STEPS = (0, 10, 20)
_EXPECTED_TYPE = re.compile(r"^\s*\[(\w+)\]")


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _output_dir(overrides: Sequence[str]) -> Path:
    for override in overrides:
        if override.startswith("logger.log_dir="):
            output = Path(override.split("=", 1)[1]).parent / "short_gold_audit"
            output.mkdir(parents=True, exist_ok=True)
            return output
    output = THIS_DIR / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default)))


def _question_type(expected: str) -> str:
    match = _EXPECTED_TYPE.match(str(expected))
    return match.group(1).lower() if match else "unknown"


class _CachedIndex:
    def __init__(self, index: MarkdownBM25Index):
        self.index = index
        self.cache: dict[tuple[str, int, int, bool], list] = {}

    def search(
        self,
        query: str,
        top_k: int,
        *,
        candidate_k: int,
        quality_rerank: bool,
    ) -> list:
        key = (query, top_k, candidate_k, quality_rerank)
        if key not in self.cache:
            self.cache[key] = self.index.search(
                query,
                top_k=top_k,
                candidate_k=candidate_k,
                quality_rerank=quality_rerank,
            )
        return list(self.cache[key])


def _primary_gold_attribution(label: dict[str, Any]) -> str:
    if label["label_defect"]:
        return "label_defect"
    if label["support_level"] == "none":
        return "unsupported_retrieval"
    if label["support_level"] == "partial":
        return "partial_evidence"
    return "clean_full_evidence"


def _gold_record(
    row: dict[str, Any],
    *,
    dataset_split: str,
    source_row_id: int,
    index: _CachedIndex,
    selection: dict[str, Any] | None,
    verified_trajectory: dict[str, Any] | None,
) -> dict[str, Any]:
    query = str(row["query"])
    expected = str(row["expected_answer"])
    metadata = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    bank = str(metadata.get("bank", ""))
    model_search_query = question_context(query)[:256]
    retrieval_query = build_retrieval_query(model_search_query, query, bank)
    results = index.search(
        retrieval_query,
        top_k=20,
        candidate_k=50,
        quality_rerank=True,
    )
    label = audit_short_label(
        query=query,
        expected=expected,
        bank=bank,
        results=results,
        top_k=20,
    )
    has_verified_route = bool(
        verified_trajectory
        and float((verified_trajectory.get("_audit") or {}).get("cumulative_coverage", 0.0)) >= 1.0
    )
    strict_rebuild_candidate = bool(
        dataset_split == "train"
        and not label["label_defect"]
        and 2 <= int(label["keypoint_count"]) <= 6
        and label["support_level"] == "full"
        and has_verified_route
    )
    return {
        "dataset_split": dataset_split,
        "source_row_id": int(source_row_id),
        "query": query,
        "expected_answer": expected,
        "full_gold_keypoints": label["full_gold_keypoints"],
        "bank": bank,
        "search_query": model_search_query,
        "search_hops": [
            {
                "hop": 1,
                "model_search_query": model_search_query,
                "retrieval_query": retrieval_query,
                **label["evidence"],
            }
        ],
        "model_completion": None,
        "rule_matched_keypoints": [],
        "official_rule_reward": None,
        "logged_reward": None,
        "label_defect": label["label_defect"],
        "label_defect_reasons": label["label_defect_reasons"],
        "label_issue_codes": label["label_issue_codes"],
        "duplicate_keypoint_pairs": label["duplicate_keypoint_pairs"],
        "support_level": label["support_level"],
        "literal_support_level": label["evidence"]["literal_support_level"],
        "reward_attacks": label["reward_attacks"],
        "attack_full_reward": label["attack_full_reward"],
        "false_perfect_attack": label["false_perfect_attack"],
        "selection_status": selection.get("selection_status") if selection else None,
        "selection_first_observation_coverage": (
            selection.get("first_observation_coverage") if selection else None
        ),
        "verified_trajectory_search_turns": (
            verified_trajectory.get("search_turns") if verified_trajectory else None
        ),
        "strict_rebuild_candidate": strict_rebuild_candidate,
        "sft_v2_ready": False,
        "required_next_action": (
            "reconstruct_complete_evidence_bound_target"
            if strict_rebuild_candidate
            else "reject_or_retrieve_better_evidence"
        ),
        "failure_labels": [
            category
            for category, enabled in (
                ("label_defect", label["label_defect"]),
                ("unsupported_retrieval", label["support_level"] == "none"),
                ("partial_evidence", label["support_level"] == "partial"),
            )
            if enabled
        ],
        "primary_attribution": _primary_gold_attribution(label),
        "_label_audit": label,
        "_source_row": row,
    }


def _eval_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "user", "content": str(row["query"])}
    ]
    responses = list(row.get("assistant_responses") or [])
    observations = list(row.get("environment_observations") or [])
    for response, observation in zip_longest(responses, observations):
        if response is not None:
            messages.append({"role": "assistant", "content": str(response)})
        if observation is not None:
            messages.append({"role": "environment", "content": str(observation)})
    return messages


def _public_gold(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if not key.startswith("_")
    }


def _audit_sft_manifest(
    trajectory_rows: Sequence[dict[str, Any]],
    train_gold: dict[int, dict[str, Any]],
    index: _CachedIndex,
) -> list[dict[str, Any]]:
    audited = []
    for row in trajectory_rows:
        if str(row.get("question_type")) != "short":
            continue
        source_row_id = int(row["row_id"])
        gold = train_gold[source_row_id]
        audited.append(
            audit_short_trace(
                trace_source="sft_v1_supervision",
                source_row_id=source_row_id,
                query=str(row["query"]),
                expected=str(row["expected_answer"]),
                bank=str(row.get("bank", "")),
                messages=list(row["messages"]),
                label_audit=gold["_label_audit"],
                index=index,
            )
        )
    return audited


def _audit_sft_eval(
    eval_rows: Sequence[dict[str, Any]],
    val_gold: dict[int, dict[str, Any]],
    index: _CachedIndex,
) -> list[dict[str, Any]]:
    audited = []
    for row in eval_rows:
        if str(row.get("question_type")) != "short":
            continue
        source_row_id = int(row["row_index"])
        gold = val_gold[source_row_id]
        audited.append(
            audit_short_trace(
                trace_source="sft_v1_step50_greedy",
                source_row_id=source_row_id,
                query=str(row["query"]),
                expected=str(row["expected_answer"]),
                bank=str(row.get("bank", "")),
                messages=_eval_messages(row),
                label_audit=gold["_label_audit"],
                index=index,
                logged_reward=float(row["reward"]),
                sample_number=source_row_id + 1,
                checkpoint_step=50,
            )
        )
    return audited


def _audit_grpo_step(
    path: Path,
    *,
    step: int,
    val_rows: Sequence[dict[str, Any]],
    val_gold: dict[int, dict[str, Any]],
    index: _CachedIndex,
) -> list[dict[str, Any]]:
    logged_rows = _read_jsonl(path)
    if len(logged_rows) > len(val_rows):
        raise ValueError(f"{path}: more logged rows than official validation rows")
    audited = []
    for source_row_id, logged in enumerate(logged_rows):
        raw_row = val_rows[source_row_id]
        if _question_type(str(raw_row["expected_answer"])) != "short":
            continue
        try:
            content = _logged_messages(logged)
            reward_value = _logged_scalar(logged, "rewards")
            logged_idx = _logged_scalar(logged, "idx")
        except ValueError as exc:
            raise ValueError(f"{path}:{source_row_id + 1}: {exc}") from exc
        if logged_idx is not None and int(logged_idx) != source_row_id:
            raise ValueError(
                f"{path}:{source_row_id + 1}: logged idx {logged_idx} "
                f"does not match source row {source_row_id}"
            )
        prompt = str(content[0].get("content", "")) if content else ""
        query = str(raw_row["query"])
        if normalize_for_mapping(query) not in normalize_for_mapping(prompt):
            raise ValueError(
                f"{path}:{source_row_id + 1}: validation row no longer matches logged prompt"
            )
        if reward_value is None:
            raise ValueError(f"{path}:{source_row_id + 1}: invalid logged reward")
        metadata = raw_row.get("meta") if isinstance(raw_row.get("meta"), dict) else {}
        audited.append(
            audit_short_trace(
                trace_source=f"short_grpo_sampled_step{step}",
                source_row_id=source_row_id,
                query=query,
                expected=str(raw_row["expected_answer"]),
                bank=str(metadata.get("bank", "")),
                messages=content,
                label_audit=val_gold[source_row_id]["_label_audit"],
                index=index,
                logged_reward=float(reward_value),
                sample_number=source_row_id + 1,
                checkpoint_step=step,
            )
        )
    return audited


def normalize_for_mapping(value: str) -> str:
    return "".join(str(value).split())


def _logged_messages(logged: dict[str, Any]) -> list[dict[str, Any]]:
    content = logged.get("content")
    if (
        isinstance(content, list)
        and len(content) == 1
        and isinstance(content[0], list)
    ):
        content = content[0]
    if not isinstance(content, list) or not all(
        isinstance(message, dict) for message in content
    ):
        raise ValueError("invalid logged message content")
    return content


def _logged_scalar(logged: dict[str, Any], key: str) -> int | float | None:
    value = logged.get(key)
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"{key} must be a scalar or singleton list")
        value = value[0]
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return value


def _nested_cross_table(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        support = str(row.get("evidence_support_level", row.get("support_level", "unknown")))
        table[support][str(row["primary_attribution"])] += 1
    return {
        support: dict(sorted(counts.items()))
        for support, counts in sorted(table.items())
    }


def _count_block(values: Sequence[str]) -> dict[str, dict[str, float | int]]:
    total = len(values)
    counts = Counter(values)
    return {
        key: {
            "count": count,
            "rate": count / total if total else 0.0,
        }
        for key, count in sorted(counts.items())
    }


def _compact_example(row: dict[str, Any], why_selected: str) -> dict[str, Any]:
    hops = list(row.get("search_hops") or [])
    evidence = []
    for hop in hops[:2]:
        for result in list(hop.get("top_k_results") or [])[:2]:
            evidence.append(
                {
                    "hop": hop.get("hop"),
                    "source": result.get("source"),
                    "heading": result.get("heading"),
                    "quality_category": result.get("quality_category"),
                    "keypoint_matches": result.get("keypoint_matches"),
                    "text_excerpt": str(result.get("text", ""))[:320],
                }
            )
    return {
        "why_selected": why_selected,
        "trace_source": row.get("trace_source"),
        "dataset_split": row.get("dataset_split"),
        "source_row_id": row.get("source_row_id"),
        "sample_number": row.get("sample_number"),
        "checkpoint_step": row.get("checkpoint_step"),
        "primary_attribution": row.get("primary_attribution"),
        "failure_labels": row.get("failure_labels"),
        "label_defect_reasons": row.get("label_defect_reasons"),
        "query": row.get("query"),
        "expected_answer": row.get("expected_answer"),
        "full_gold_keypoints": row.get("full_gold_keypoints"),
        "search_queries": [
            hop.get("model_search_query")
            for hop in hops
        ],
        "evidence_support_level": row.get("evidence_support_level", row.get("support_level")),
        "evidence": evidence,
        "model_completion": row.get("model_completion"),
        "rule_matched_keypoints": row.get("rule_matched_keypoints"),
        "official_rule_reward": row.get("official_rule_reward"),
        "logged_reward": row.get("logged_reward"),
        "false_perfect": row.get("false_perfect", row.get("false_perfect_attack")),
        "reward_hacking": row.get("reward_hacking"),
    }


def _representative_examples(
    gold_rows: Sequence[dict[str, Any]],
    trace_rows: Sequence[dict[str, Any]],
    *,
    limit: int = 15,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def add(row: dict[str, Any], reason: str) -> None:
        key = (
            row.get("trace_source", row.get("dataset_split")),
            row.get("checkpoint_step"),
            row.get("source_row_id"),
        )
        if key in seen or len(selected) >= limit:
            return
        seen.add(key)
        selected.append(_compact_example(row, reason))

    regression = next(
        (
            row
            for row in trace_rows
            if row.get("trace_source") == "short_grpo_sampled_step20"
            and row.get("sample_number") == 180
        ),
        None,
    )
    if regression:
        add(regression, "required_reward_hacking_regression")

    for category in (
        "label_defect",
        "unsupported_retrieval",
        "partial_evidence",
        "synthesis_failure",
        "protocol_failure",
        "success",
    ):
        for row in trace_rows:
            if row["primary_attribution"] == category:
                add(row, f"trace_primary:{category}")
                if sum(
                    example["why_selected"] == f"trace_primary:{category}"
                    for example in selected
                ) >= 2:
                    break

    for reason in (
        "singleton_generic_keypoint",
        "singleton_very_short_keypoint",
        "singleton_question_high_overlap",
        "near_duplicate_keypoints",
        "no_answer_bearing_evidence_mapping",
    ):
        for row in gold_rows:
            if reason in row["label_defect_reasons"]:
                add(row, f"label_reason:{reason}")
                break

    for row in gold_rows:
        if row.get("strict_rebuild_candidate"):
            add(row, "strict_rebuild_candidate")
            break
    for row in [*trace_rows, *gold_rows]:
        add(row, "coverage_fill")
        if len(selected) >= limit:
            break
    return selected


def _assert_sample_180(trace_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    matches = [
        row
        for row in trace_rows
        if row.get("trace_source") == "short_grpo_sampled_step20"
        and row.get("sample_number") == 180
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one step20 sample 180 trace, found {len(matches)}")
    row = matches[0]
    boxed = str(row.get("boxed") or "").strip()
    checks = {
        "expected_is_degenerate_code": str(row["expected_answer"]).strip() == "[short] 代码",
        "completion_is_bare_code": boxed == "代码",
        "official_reward_is_one": abs(float(row["official_rule_reward"]) - 1.0) <= 1e-9,
        "logged_reward_is_one": abs(float(row["logged_reward"]) - 1.0) <= 1e-9,
        "classified_label_defect": row["primary_attribution"] == "label_defect",
        "classified_false_perfect": bool(row["false_perfect"]),
        "classified_reward_hacking": bool(row["reward_hacking"]),
    }
    if not all(checks.values()):
        raise RuntimeError(f"step20 sample180 reward-hacking regression failed: {checks}")
    return {
        "trace_source": row["trace_source"],
        "source_row_id": row["source_row_id"],
        "sample_number": row["sample_number"],
        "query": row["query"],
        "expected_answer": row["expected_answer"],
        "search_queries": [
            hop["model_search_query"] for hop in row["search_hops"]
        ],
        "evidence_support_level": row["evidence_support_level"],
        "model_completion": row["model_completion"],
        "official_rule_reward": row["official_rule_reward"],
        "logged_reward": row["logged_reward"],
        "label_defect_reasons": row["label_defect_reasons"],
        "checks": checks,
    }


def main() -> None:
    _, overrides = _parse_args()
    data_dir = Path(os.environ.get("QA_RL_DATA_DIR", "/data/datasets/qa_rl"))
    docs_dir = Path(os.environ.get("QA_DOCS_DIR", "/data/docs"))
    clean_train_path = _env_path("QA_CLEAN_TRAIN_PATH", DEFAULT_CLEAN_TRAIN_PATH)
    selection_path = _env_path("QA_SFT_SELECTION_PATH", DEFAULT_SELECTION_PATH)
    trajectory_path = _env_path("QA_SFT_TRAJECTORY_PATH", DEFAULT_TRAJECTORY_PATH)
    sft_eval_path = _env_path("QA_SFT_EVAL_PATH", DEFAULT_SFT_EVAL_PATH)
    grpo_log_dir = _env_path("QA_SHORT_GRPO_LOG_DIR", DEFAULT_GRPO_LOG_DIR)
    val_path = data_dir / "val.jsonl"
    grpo_paths = {
        step: grpo_log_dir / f"val_data_step{step}.jsonl"
        for step in GRPO_STEPS
    }
    required = [
        docs_dir,
        clean_train_path,
        selection_path,
        trajectory_path,
        sft_eval_path,
        val_path,
        *grpo_paths.values(),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing short-audit inputs: {missing}")

    clean_rows = _read_jsonl(clean_train_path)
    val_rows = _read_jsonl(val_path)
    selection_rows = _read_jsonl(selection_path)
    trajectory_rows = _read_jsonl(trajectory_path)
    sft_eval_rows = _read_jsonl(sft_eval_path)
    selection_by_id = {
        int(row["row_id"]): row
        for row in selection_rows
        if row.get("row_id") is not None
    }
    verified_trajectory_by_id = {
        int(row["row_id"]): row
        for row in trajectory_rows
        if row.get("row_id") is not None
        and row.get("question_type") in {"fill", "short"}
    }

    index_start = time.perf_counter()
    cached_index = _CachedIndex(
        MarkdownBM25Index.from_directory(
            docs_dir,
            chunk_chars=1200,
            overlap_chars=160,
            k1=1.5,
            b=0.75,
        )
    )
    index_seconds = time.perf_counter() - index_start

    train_short_rows = [
        row
        for row in clean_rows
        if _question_type(str(row.get("expected_answer", ""))) == "short"
    ]
    val_short_ids = [
        row_id
        for row_id, row in enumerate(val_rows)
        if _question_type(str(row.get("expected_answer", ""))) == "short"
    ]
    gold_rows = []
    train_gold: dict[int, dict[str, Any]] = {}
    for position, row in enumerate(train_short_rows, start=1):
        clean = row.get("_clean") if isinstance(row.get("_clean"), dict) else {}
        source_row_id = int(clean["row_id"])
        record = _gold_record(
            row,
            dataset_split="train",
            source_row_id=source_row_id,
            index=cached_index,
            selection=selection_by_id.get(source_row_id),
            verified_trajectory=verified_trajectory_by_id.get(source_row_id),
        )
        train_gold[source_row_id] = record
        gold_rows.append(record)
        if position % 100 == 0:
            print(f"[short-audit] train labels {position}/{len(train_short_rows)}", flush=True)

    val_gold: dict[int, dict[str, Any]] = {}
    for source_row_id in val_short_ids:
        record = _gold_record(
            val_rows[source_row_id],
            dataset_split="official_validation",
            source_row_id=source_row_id,
            index=cached_index,
            selection=None,
            verified_trajectory=None,
        )
        val_gold[source_row_id] = record
        gold_rows.append(record)
    print(f"[short-audit] gold labels={len(gold_rows)}", flush=True)

    trace_rows = []
    trace_rows.extend(_audit_sft_manifest(trajectory_rows, train_gold, cached_index))
    trace_rows.extend(_audit_sft_eval(sft_eval_rows, val_gold, cached_index))
    for step, path in grpo_paths.items():
        trace_rows.extend(
            _audit_grpo_step(
                path,
                step=step,
                val_rows=val_rows,
                val_gold=val_gold,
                index=cached_index,
            )
        )
    regression = _assert_sample_180(trace_rows)

    public_gold = [_public_gold(row) for row in gold_rows]
    strict_candidates = [
        {
            **_public_gold(row),
            "original_row": row["_source_row"],
        }
        for row in gold_rows
        if row["strict_rebuild_candidate"]
    ]
    rejected = [
        _public_gold(row)
        for row in gold_rows
        if row["label_defect"]
    ]
    representatives = _representative_examples(public_gold, trace_rows)
    if not 10 <= len(representatives) <= 20:
        raise RuntimeError(
            f"representative example count must be in [10, 20], got {len(representatives)}"
        )

    output_dir = _output_dir(overrides)
    paths = {
        "short_gold_audit": output_dir / "short_gold_audit.jsonl",
        "short_trajectory_audit": output_dir / "short_trajectory_audit.jsonl",
        "rebuild_candidates": output_dir / "rebuild_candidates.jsonl",
        "rejected_short_labels": output_dir / "rejected_short_labels.jsonl",
        "representative_examples": output_dir / "representative_examples.jsonl",
        "summary": output_dir / "summary.json",
    }
    _write_jsonl(paths["short_gold_audit"], public_gold)
    _write_jsonl(paths["short_trajectory_audit"], trace_rows)
    _write_jsonl(paths["rebuild_candidates"], strict_candidates)
    _write_jsonl(paths["rejected_short_labels"], rejected)
    _write_jsonl(paths["representative_examples"], representatives)

    gold_primary = [str(row["primary_attribution"]) for row in public_gold]
    trace_primary = [str(row["primary_attribution"]) for row in trace_rows]
    trace_failure_any = Counter(
        failure
        for row in trace_rows
        for failure in row["failure_labels"]
    )
    label_issue_counts = Counter(
        reason
        for row in public_gold
        for reason in row["label_defect_reasons"]
    )
    evidence_mapping_failure_count = sum(
        bool(row["evidence_mapping_failure"])
        for row in public_gold
    )
    trace_source_counts = Counter(str(row["trace_source"]) for row in trace_rows)
    false_perfect_by_source = Counter(
        str(row["trace_source"])
        for row in trace_rows
        if row["false_perfect"]
    )
    attack_full_reward_counts = Counter(
        attack["attack"]
        for row in public_gold
        for attack in row["reward_attacks"]
        if float(attack["official_rule_reward"]) >= 1.0
    )
    reward_mismatches = [
        row
        for row in trace_rows
        if row["logged_reward"] is not None
        and not row["logged_reward_matches_official_rule"]
    ]
    summary = {
        "audit_version": 2,
        "read_only": True,
        "official_rule_reward_modified": False,
        "semantic_judge_used": False,
        "inputs": {
            "docs": str(docs_dir),
            "clean_train": str(clean_train_path),
            "official_validation": str(val_path),
            "selection_manifest": str(selection_path),
            "trajectory_manifest": str(trajectory_path),
            "sft_step50_eval": str(sft_eval_path),
            "short_grpo_logs": {
                str(step): str(path) for step, path in grpo_paths.items()
            },
        },
        "gold_labels": {
            "total": len(public_gold),
            "train": len(train_short_rows),
            "official_validation": len(val_short_ids),
            "label_defect_count": sum(bool(row["label_defect"]) for row in public_gold),
            "label_defect_rate": (
                sum(bool(row["label_defect"]) for row in public_gold) / len(public_gold)
                if public_gold
                else 0.0
            ),
            "primary_attribution": _count_block(gold_primary),
            "support_levels": _count_block(
                [str(row["support_level"]) for row in public_gold]
            ),
            "label_defect_reasons": dict(sorted(label_issue_counts.items())),
            "evidence_mapping_failure_count": evidence_mapping_failure_count,
            "evidence_mapping_failure_rate": (
                evidence_mapping_failure_count / len(public_gold)
                if public_gold
                else 0.0
            ),
            "attack_full_reward_counts": dict(sorted(attack_full_reward_counts.items())),
            "false_perfect_attack_count": sum(
                bool(row["false_perfect_attack"]) for row in public_gold
            ),
            "evidence_by_primary_attribution": _nested_cross_table(public_gold),
        },
        "trajectory_audit": {
            "total": len(trace_rows),
            "source_counts": dict(sorted(trace_source_counts.items())),
            "primary_attribution": _count_block(trace_primary),
            "failure_attribution_any": {
                category: {
                    "count": trace_failure_any.get(category, 0),
                    "rate": trace_failure_any.get(category, 0) / len(trace_rows)
                    if trace_rows
                    else 0.0,
                }
                for category in (
                    "label_defect",
                    "unsupported_retrieval",
                    "partial_evidence",
                    "synthesis_failure",
                    "protocol_failure",
                )
            },
            "evidence_by_primary_attribution": _nested_cross_table(trace_rows),
            "false_perfect_count": sum(bool(row["false_perfect"]) for row in trace_rows),
            "false_perfect_by_source": dict(sorted(false_perfect_by_source.items())),
            "logged_reward_mismatch_count": len(reward_mismatches),
        },
        "cleaning_decision": {
            "strict_rebuild_candidate_count": len(strict_candidates),
            "rejected_label_count": len(rejected),
            "sft_v2_ready_count": 0,
            "sft_v2_ready_reason": (
                "No original keyword gold is train-ready until a complete 2-6 point "
                "target is reconstructed and each point is bound to visible evidence."
            ),
        },
        "reward_hacking_regression_step20_sample180": regression,
        "representative_example_count": len(representatives),
        "index": {
            "num_documents": cached_index.index.num_documents,
            "quality_category_counts": cached_index.index.quality_category_counts,
            "build_seconds": index_seconds,
            "cached_queries": len(cached_index.cache),
        },
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[short-audit] summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
