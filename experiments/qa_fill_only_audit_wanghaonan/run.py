#!/usr/bin/env python
"""Audit fill-only retrieval trajectories without reading short-answer outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

from common.retrieval.markdown_bm25 import MarkdownBM25Index  # noqa: E402
from common.retrieval.qa_sft_v2 import (  # noqa: E402
    OBJECTIVE_TYPES,
    SPLITS,
    assert_question_split_isolation,
    objective_replay_fraction,
    select_balanced_objective_replay,
    select_objective_validation,
)
from common.retrieval.qa_target_rebuild import question_fingerprint  # noqa: E402
from experiments.qa_sft_v2_data_build_wanghaonan.run import (  # noqa: E402
    DEFAULT_V1_MANIFEST,
    MAX_TOKENS,
    _fill_trajectory,
    _load_tokenizer,
)

EXPECTED_OFFICIAL_FINGERPRINTS = 313
MIN_SPLIT_COUNTS = {
    "train": 40,
    "validation": 8,
    "rl_holdout": 8,
}
ANSWER_LEAK_REASONS = frozenset(
    {
        "answer_visible_in_question",
        "answer_terms_visible_in_question",
        "first_query_answer_leak",
        "first_query_answer_term_leak",
        "second_query_answer_leak",
        "second_query_answer_term_leak",
    }
)


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
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _output_dir(overrides: Sequence[str]) -> Path:
    for override in overrides:
        if override.startswith("logger.log_dir="):
            output = Path(override.split("=", 1)[1]).parent / "fill_only_audit"
            output.mkdir(parents=True, exist_ok=True)
            return output
    output = THIS_DIR / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _row_id(row: Mapping[str, Any]) -> int:
    return int(row.get("row_id", row.get("source_row_id", -1)))


def _source_metadata(
    source: Mapping[str, Any],
    *,
    fingerprint: str,
    source_row_ids: Sequence[int],
) -> dict[str, Any]:
    audit = source.get("_audit")
    audit = audit if isinstance(audit, Mapping) else {}
    return {
        "source_row_id": _row_id(source),
        "source_row_ids": list(source_row_ids),
        "source_duplicate_count": len(source_row_ids) - 1,
        "question_fingerprint": fingerprint,
        "query": str(source.get("query", "")),
        "expected_answer": str(source.get("expected_answer", "")),
        "source_split": str(source.get("split", "")),
        "first_query": str(audit.get("first_query", "")),
        "second_query": str(audit.get("second_query", "")),
    }


def _unique_fill_sources(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    fill_rows = [
        dict(record)
        for record in records
        if record.get("question_type") == "fill"
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fill_rows:
        groups[question_fingerprint(str(row.get("query", "")))].append(row)

    selected = []
    rejected = []
    for fingerprint, group in sorted(groups.items()):
        ordered = sorted(group, key=_row_id)
        source_row_ids = [_row_id(row) for row in ordered]
        source = ordered[0]
        metadata = _source_metadata(
            source,
            fingerprint=fingerprint,
            source_row_ids=source_row_ids,
        )
        splits = {str(row.get("split", "")) for row in ordered}
        expected_answers = {
            str(row.get("expected_answer", "")).strip()
            for row in ordered
        }
        if not str(source.get("query", "")).strip():
            reason = "missing_query"
        elif splits - set(SPLITS):
            reason = "invalid_source_split"
        elif len(splits) != 1:
            reason = "source_question_cross_split"
        elif len(expected_answers) != 1:
            reason = "source_question_conflicting_answers"
        else:
            reason = None

        if reason:
            rejected.append(
                {
                    **metadata,
                    "accepted": False,
                    "decision": reason,
                    "output_split": None,
                    "search_turns": None,
                    "token_length": None,
                    "answer_leak_detected": False,
                    "protocol_check": "not_reached",
                    "token_check": "not_reached",
                    "official_validation_overlap": False,
                    "search_queries": [],
                }
            )
            continue

        selected.append(
            {
                **source,
                "question_fingerprint": fingerprint,
                "_fill_audit_source": metadata,
            }
        )

    return (
        selected,
        rejected,
        {
            "raw_fill_rows": len(fill_rows),
            "unique_source_questions": len(groups),
            "same_question_duplicate_rows": len(fill_rows) - len(groups),
            "eligible_unique_questions": len(selected),
            "source_prefilter_rejections": len(rejected),
        },
    )


def _audit_record(
    source: Mapping[str, Any],
    record: Mapping[str, Any] | None,
    reason: str,
) -> dict[str, Any]:
    metadata = source["_fill_audit_source"]
    if not isinstance(metadata, Mapping):
        raise TypeError("fill audit source metadata must be an object")
    accepted = record is not None
    token_length = (
        int(record["_audit"]["token_length"])
        if accepted
        else None
    )
    return {
        **dict(metadata),
        "accepted": accepted,
        "decision": reason,
        "output_split": str(record["split"]) if accepted else None,
        "search_turns": int(record["search_turns"]) if accepted else None,
        "token_length": token_length,
        "answer_leak_detected": reason in ANSWER_LEAK_REASONS,
        "protocol_check": (
            "passed"
            if accepted
            else "failed"
            if reason.startswith("message_validation:")
            else "not_reached"
        ),
        "token_check": (
            "passed"
            if accepted
            else "failed"
            if reason == "trajectory_too_long"
            else "not_reached"
        ),
        "official_validation_overlap": reason == "official_validation_overlap",
        "search_queries": (
            [
                str(hop["model_search_query"])
                for hop in record["search_hops"]
            ]
            if accepted
            else []
        ),
    }


def _accepted_record_issues(record: Mapping[str, Any]) -> list[str]:
    issues = []
    audit = record.get("_audit")
    audit = audit if isinstance(audit, Mapping) else {}
    fingerprint = str(record.get("question_fingerprint", ""))
    if record.get("question_type") != "fill":
        issues.append("question_type")
    if fingerprint != question_fingerprint(str(record.get("query", ""))):
        issues.append("question_fingerprint")
    if record.get("split") not in SPLITS:
        issues.append("split")
    if record.get("human_reviewed") is not False:
        issues.append("human_review_state")
    search_turns = int(record.get("search_turns", 0))
    search_hops = record.get("search_hops")
    if search_turns not in {1, 2}:
        issues.append("search_turns")
    if not isinstance(search_hops, list) or len(search_hops) != search_turns:
        issues.append("search_hops")
    if audit.get("trusted_visible_coverage") != 1.0:
        issues.append("trusted_visible_coverage")
    if audit.get("query_leakage_check") is not True:
        issues.append("query_leakage_check")
    if audit.get("official_validation_fingerprint_overlap") is not False:
        issues.append("official_validation_overlap")
    if audit.get("runtime_raw_chunk_alignment") is not True:
        issues.append("runtime_protocol")
    if search_turns == 2 and audit.get("incremental_two_hop") is not True:
        issues.append("nonincremental_two_hop")
    token_length = int(audit.get("token_length", MAX_TOKENS + 1))
    if token_length > MAX_TOKENS:
        issues.append("token_length")
    return issues


def _audit_fill_sources(
    records: Sequence[Mapping[str, Any]],
    index: MarkdownBM25Index,
    tokenizer: Any,
    official_fingerprints: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    candidates, audit_rows, source_stats = _unique_fill_sources(records)
    accepted = []
    for position, source in enumerate(candidates, start=1):
        record, reason = _fill_trajectory(
            source,
            index,
            tokenizer,
            official_fingerprints,
        )
        if record:
            record = dict(record)
            record["_audit"] = {
                **dict(record["_audit"]),
                "source_duplicate_count": int(
                    source["_fill_audit_source"]["source_duplicate_count"]
                ),
            }
            issues = _accepted_record_issues(record)
            if issues:
                raise RuntimeError(
                    f"accepted fill row {record['source_row_id']} failed re-audit: "
                    f"{issues}"
                )
            accepted.append(record)
        audit_rows.append(_audit_record(source, record, reason))
        if position % 25 == 0 or position == len(candidates):
            print(
                f"[fill-only-audit] audited={position}/{len(candidates)} "
                f"accepted={len(accepted)}",
                flush=True,
            )

    accepted.sort(key=lambda record: int(record["source_row_id"]))
    audit_rows.sort(
        key=lambda row: (
            str(row["question_fingerprint"]),
            int(row["source_row_id"]),
        )
    )
    if accepted:
        assert_question_split_isolation(accepted)
    return accepted, audit_rows, source_stats


def _count_by(
    records: Sequence[Mapping[str, Any]],
    *keys: str,
) -> dict[str, int]:
    counts = Counter(
        ":".join(str(record.get(key)) for key in keys)
        for record in records
    )
    return dict(sorted(counts.items()))


def _split_overlap_report(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    fingerprints = {
        split: {
            str(record["question_fingerprint"])
            for record in records
            if record.get("split") == split
        }
        for split in SPLITS
    }
    report = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            report[f"{left}:{right}"] = sorted(
                fingerprints[left] & fingerprints[right]
            )
    return report


def _review_samples(
    accepted: Sequence[Mapping[str, Any]],
    *,
    random_count: int = 20,
    seed: int = 42,
) -> list[dict[str, Any]]:
    two_hop = sorted(
        (
            record
            for record in accepted
            if int(record["search_turns"]) == 2
        ),
        key=lambda record: str(record["question_fingerprint"]),
    )
    two_hop_fingerprints = {
        str(record["question_fingerprint"])
        for record in two_hop
    }
    remaining = [
        record
        for record in accepted
        if str(record["question_fingerprint"]) not in two_hop_fingerprints
    ]
    sampled = sorted(
        remaining,
        key=lambda record: hashlib.sha256(
            f"{seed}:{record['question_fingerprint']}".encode("utf-8")
        ).hexdigest(),
    )[:random_count]

    selected = [
        *(("all_two_hop", record) for record in two_hop),
        *(("deterministic_random", record) for record in sampled),
    ]
    return [
        {
            **dict(record),
            "review_selection": reason,
            "human_reviewed": False,
            "human_review_checklist": {
                "all_gold_points_visible": None,
                "queries_do_not_leak_answer": None,
                "second_hop_is_incremental": None,
                "final_answer_is_complete": None,
                "runtime_protocol_is_valid": None,
                "decision": "pending_human_review",
            },
        }
        for reason, record in selected
    ]


def _representative_rejections(
    audit_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    representatives = []
    seen = set()
    for row in sorted(
        (row for row in audit_rows if not row["accepted"]),
        key=lambda row: (
            str(row["decision"]),
            str(row["question_fingerprint"]),
        ),
    ):
        decision = str(row["decision"])
        if decision in seen:
            continue
        seen.add(decision)
        representatives.append(dict(row))
    return representatives


def _objective_replay_availability(
    records: Sequence[Mapping[str, Any]],
    accepted_fill: Sequence[Mapping[str, Any]],
    official_fingerprints: set[str],
) -> dict[str, Any]:
    raw = [
        dict(record)
        for record in records
        if record.get("question_type") in OBJECTIVE_TYPES
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in raw:
        groups[
            question_fingerprint(str(record.get("query", "")))
        ].append(record)

    accepted_fill_fingerprints = {
        str(record["question_fingerprint"])
        for record in accepted_fill
    }
    candidates = []
    exclusions: Counter[str] = Counter()
    for fingerprint, group in sorted(groups.items()):
        ordered = sorted(group, key=_row_id)
        splits = {str(record.get("split", "")) for record in ordered}
        question_types = {
            str(record.get("question_type", ""))
            for record in ordered
        }
        if splits - set(SPLITS):
            exclusions["invalid_source_split"] += 1
            continue
        if len(splits) != 1:
            exclusions["source_question_cross_split"] += 1
            continue
        if len(question_types) != 1:
            exclusions["source_question_conflicting_type"] += 1
            continue
        if fingerprint in official_fingerprints:
            exclusions["official_validation_overlap"] += 1
            continue
        if fingerprint in accepted_fill_fingerprints:
            exclusions["fill_question_overlap"] += 1
            continue
        candidates.append(
            {
                **ordered[0],
                "question_fingerprint": fingerprint,
            }
        )

    fill_train_count = sum(
        record.get("split") == "train"
        for record in accepted_fill
    )
    train_candidates = [
        record
        for record in candidates
        if record.get("split") == "train"
    ]
    validation_candidates = [
        record
        for record in candidates
        if record.get("split") == "validation"
    ]
    train_selection = []
    train_selection_error = None
    try:
        train_selection = select_balanced_objective_replay(
            train_candidates,
            open_train_count=fill_train_count,
        )
    except ValueError as error:
        train_selection_error = str(error)

    validation_selection = []
    validation_selection_error = None
    try:
        validation_selection = select_objective_validation(
            validation_candidates,
            per_type=2,
        )
    except ValueError as error:
        validation_selection_error = str(error)

    return {
        "raw_rows": len(raw),
        "unique_source_questions": len(groups),
        "eligible_unique_questions": len(candidates),
        "exclusion_counts": dict(sorted(exclusions.items())),
        "available_counts": _count_by(
            candidates,
            "split",
            "question_type",
        ),
        "balanced_train_replay": {
            "open_fill_train_count": fill_train_count,
            "selected_count": len(train_selection),
            "selected_per_type": _count_by(
                train_selection,
                "question_type",
            ),
            "resulting_fraction": (
                objective_replay_fraction(
                    [*accepted_fill, *train_selection]
                )
                if train_selection
                else None
            ),
            "selection_error": train_selection_error,
        },
        "validation_replay": {
            "selected_count": len(validation_selection),
            "selected_per_type": _count_by(
                validation_selection,
                "question_type",
            ),
            "selection_error": validation_selection_error,
        },
        "informational_only": True,
        "included_in_training_outputs": False,
    }


def _machine_gate(
    accepted: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
    official_fingerprints: set[str],
) -> dict[str, Any]:
    split_counts = Counter(str(record["split"]) for record in accepted)
    split_overlaps = _split_overlap_report(accepted)
    official_overlap = sorted(
        {
            str(record["question_fingerprint"])
            for record in accepted
        }
        & official_fingerprints
    )
    issue_counts: Counter[str] = Counter()
    for record in accepted:
        issue_counts.update(_accepted_record_issues(record))
    source_cross_split = sum(
        row.get("decision") == "source_question_cross_split"
        for row in audit_rows
    )
    threshold_checks = {
        split: split_counts.get(split, 0) >= minimum
        for split, minimum in MIN_SPLIT_COUNTS.items()
    }
    passed = (
        all(threshold_checks.values())
        and not official_overlap
        and not any(split_overlaps.values())
        and source_cross_split == 0
        and not issue_counts
    )
    return {
        "passed": passed,
        "minimum_split_counts": MIN_SPLIT_COUNTS,
        "actual_split_counts": {
            split: split_counts.get(split, 0)
            for split in SPLITS
        },
        "split_threshold_checks": threshold_checks,
        "split_fingerprint_overlaps": split_overlaps,
        "source_cross_split_rejection_count": source_cross_split,
        "official_validation_overlap_count": len(official_overlap),
        "official_validation_overlap_fingerprints": official_overlap,
        "accepted_record_issue_counts": dict(sorted(issue_counts.items())),
        "human_review_still_required": True,
        "human_review_passed": False,
        "training_ready_count": 0,
        "auto_training_allowed": False,
    }


def main() -> None:
    _, overrides = _parse_args()
    output_dir = _output_dir(overrides)
    v1_manifest_path = Path(
        os.environ.get("QA_SFT_V1_MANIFEST", str(DEFAULT_V1_MANIFEST))
    )
    data_dir = Path(os.environ.get("QA_RL_DATA_DIR", "/data/datasets/qa_rl"))
    docs_dir = Path(os.environ.get("QA_DOCS_DIR", "/data/docs"))
    validation_path = data_dir / "val.jsonl"
    missing = [
        str(path)
        for path in (v1_manifest_path, validation_path, docs_dir)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing fill-only audit inputs: {missing}")

    v1_records = _read_jsonl(v1_manifest_path)
    official_rows = _read_jsonl(validation_path)
    official_fingerprints = {
        question_fingerprint(str(row["query"]))
        for row in official_rows
    }
    if (
        len(official_rows) != EXPECTED_OFFICIAL_FINGERPRINTS
        or len(official_fingerprints) != EXPECTED_OFFICIAL_FINGERPRINTS
    ):
        raise RuntimeError(
            "official validation integrity check failed: "
            f"rows={len(official_rows)}, "
            f"fingerprints={len(official_fingerprints)}"
        )

    index_start = time.perf_counter()
    index = MarkdownBM25Index.from_directory(
        docs_dir,
        chunk_chars=1200,
        overlap_chars=160,
        k1=1.5,
        b=0.75,
    )
    index_seconds = time.perf_counter() - index_start
    print(
        f"[fill-only-audit] indexed={index.num_documents} "
        f"seconds={index_seconds:.1f}",
        flush=True,
    )
    tokenizer = _load_tokenizer()
    accepted, audit_rows, source_stats = _audit_fill_sources(
        v1_records,
        index,
        tokenizer,
        official_fingerprints,
    )
    rejected = [
        row
        for row in audit_rows
        if not row["accepted"]
    ]
    review_samples = _review_samples(accepted)
    representative_rejections = _representative_rejections(audit_rows)
    machine_gate = _machine_gate(
        accepted,
        audit_rows,
        official_fingerprints,
    )
    objective_replay_availability = _objective_replay_availability(
        v1_records,
        accepted,
        official_fingerprints,
    )

    paths = {
        "accepted_manifest": output_dir / "accepted_fill_manifest.jsonl",
        "rejected": output_dir / "rejected_fill_questions.jsonl",
        "audit": output_dir / "fill_question_audit.jsonl",
        "train_candidates": output_dir / "accepted_fill_train.jsonl",
        "validation_candidates": output_dir / "accepted_fill_validation.jsonl",
        "rl_holdout_candidates": output_dir / "accepted_fill_rl_holdout.jsonl",
        "human_review": output_dir / "fill_human_review_samples.jsonl",
        "representative_rejections": (
            output_dir / "representative_fill_rejections.jsonl"
        ),
        "summary": output_dir / "summary.json",
    }
    _write_jsonl(paths["accepted_manifest"], accepted)
    _write_jsonl(paths["rejected"], rejected)
    _write_jsonl(paths["audit"], audit_rows)
    for split, path_key in (
        ("train", "train_candidates"),
        ("validation", "validation_candidates"),
        ("rl_holdout", "rl_holdout_candidates"),
    ):
        _write_jsonl(
            paths[path_key],
            [
                record
                for record in accepted
                if record["split"] == split
            ],
        )
    _write_jsonl(paths["human_review"], review_samples)
    _write_jsonl(
        paths["representative_rejections"],
        representative_rejections,
    )

    token_lengths = [
        int(record["_audit"]["token_length"])
        for record in accepted
    ]
    summary = {
        "mode": "read_only_fill_audit",
        "sources": {
            "v1_manifest": str(v1_manifest_path),
            "v1_manifest_sha256": _sha256(v1_manifest_path),
            "official_validation": str(validation_path),
            "official_validation_fingerprint_count": len(
                official_fingerprints
            ),
            "docs": str(docs_dir),
        },
        "source_counts": source_stats,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "decision_counts": _count_by(audit_rows, "decision"),
        "source_split_decision_counts": _count_by(
            audit_rows,
            "source_split",
            "accepted",
        ),
        "rejection_counts_by_source_split": _count_by(
            rejected,
            "source_split",
            "decision",
        ),
        "accepted_source_split_counts": _count_by(accepted, "split"),
        "accepted_output_split_counts": _count_by(accepted, "split"),
        "search_turn_counts": _count_by(accepted, "search_turns"),
        "answer_leak_rejection_count": sum(
            bool(row["answer_leak_detected"])
            for row in rejected
        ),
        "protocol_rejection_count": sum(
            row["protocol_check"] == "failed"
            for row in rejected
        ),
        "over_length_rejection_count": sum(
            row["token_check"] == "failed"
            for row in rejected
        ),
        "objective_replay_availability": objective_replay_availability,
        "token_lengths": {
            "min": min(token_lengths) if token_lengths else None,
            "mean": mean(token_lengths) if token_lengths else None,
            "max": max(token_lengths) if token_lengths else None,
        },
        "human_review": {
            "all_two_hop_count": sum(
                int(record["search_turns"]) == 2
                for record in accepted
            ),
            "deterministic_random_count": sum(
                row["review_selection"] == "deterministic_random"
                for row in review_samples
            ),
            "sample_count": len(review_samples),
            "required": True,
            "passed": False,
            "all_records_human_reviewed_false": all(
                record.get("human_reviewed") is False
                for record in accepted
            ),
        },
        "machine_training_gate": machine_gate,
        "quality_category_counts": index.quality_category_counts,
        "timing_seconds": {"index_build": index_seconds},
        "outputs": {
            name: str(path)
            for name, path in paths.items()
        },
    }
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[fill-only-audit] summary", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("[fill-only-audit] pending human review", flush=True)
    print(
        json.dumps(review_samples[:3], ensure_ascii=False, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
