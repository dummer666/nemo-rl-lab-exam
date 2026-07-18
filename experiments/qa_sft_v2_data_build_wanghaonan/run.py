#!/usr/bin/env python
"""Build and audit short/fill retrieval SFT v2 data with objective replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
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
    SearchResult,
    build_retrieval_query,
    format_search_results_with_visible_snippets,
)
from common.retrieval.qa_sft import (  # noqa: E402
    build_objective_messages,
    build_search_messages,
    canonical_answer,
    format_agent_prompt,
    validate_messages,
)
from common.retrieval.qa_sft_v2 import (  # noqa: E402
    OBJECTIVE_TYPES,
    assert_question_split_isolation,
    grounded_answer_term_leak,
    objective_replay_fraction,
    open_answer_leak_points,
    question_keypoint_leak,
    select_balanced_objective_replay,
    select_objective_validation,
    short_target_issues,
    source_question_answer_term_leak,
)
from common.retrieval.qa_target_rebuild import question_fingerprint  # noqa: E402

MODEL_NAME = "Qwen/Qwen3.5-9B"
DEFAULT_REBUILD_ROOT = Path(
    "/shared/outputs/wanghaonan/qa_short_target_rebuild_wanghaonan"
)
DEFAULT_V1_MANIFEST = Path(
    "/shared/outputs/wanghaonan/qa_sft_trajectory_build_wanghaonan/"
    "qa_sft_trajectory_build_wanghaonan-wanghaonan-20260718-123640/"
    "sft_trajectories/trajectory_manifest.jsonl"
)
MAX_TOKENS = 6000


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _output_dir(overrides: Sequence[str]) -> Path:
    for override in overrides:
        if override.startswith("logger.log_dir="):
            output = Path(override.split("=", 1)[1]).parent / "sft_v2_data"
            output.mkdir(parents=True, exist_ok=True)
            return output
    output = THIS_DIR / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _resolve_rebuild_dir() -> Path:
    override = os.environ.get("QA_SFT_V2_REBUILD_DIR")
    if override:
        rebuild_dir = Path(override)
        if not (rebuild_dir / "summary.json").is_file():
            raise FileNotFoundError(f"explicit rebuild directory is incomplete: {rebuild_dir}")
        return rebuild_dir

    candidates = []
    for summary_path in DEFAULT_REBUILD_ROOT.glob(
        "*/short_target_rebuild/summary.json"
    ):
        summary = _read_json(summary_path)
        if (
            summary.get("mode") == "full"
            and int(summary.get("machine_verified_route_targets", 0)) > 0
        ):
            candidates.append(summary_path.parent)
    if not candidates:
        raise FileNotFoundError("no completed full short-target rebuild was found")
    return max(candidates, key=lambda path: path.parent.name)


def _load_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _runtime_messages(tokenizer, messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    if [message.get("role") for message in messages[:2]] != ["system", "user"]:
        raise ValueError("trajectory must begin with system and user")
    initial = format_agent_prompt(
        tokenizer,
        str(messages[1]["content"]),
        system_prompt=str(messages[0]["content"]),
    )
    return [
        {"role": "user", "content": initial},
        *[dict(message) for message in messages[2:]],
    ]


def _token_length(tokenizer, messages: Sequence[Mapping[str, str]]) -> int:
    rendered = "".join(str(message["content"]) for message in messages)
    return len(
        tokenizer(
            rendered,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
    )


def _visible_result_records(
    results: Sequence[SearchResult],
    visible_snippets: Sequence[str],
) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "source": result.source,
            "heading": result.heading,
            "quality_category": result.quality_category,
            "raw_score": result.raw_score,
            "text": snippet,
        }
        for rank, (result, snippet) in enumerate(
            zip(results, visible_snippets, strict=True),
            start=1,
        )
    ]


def _search_fill_hop(
    index: MarkdownBM25Index,
    *,
    model_query: str,
    original_query: str,
    bank: str,
    keypoints: Sequence[Sequence[str]],
    hop: int,
) -> dict[str, Any]:
    retrieval_query = build_retrieval_query(model_query, original_query, bank)
    results = index.search(
        retrieval_query,
        top_k=4,
        candidate_k=50,
        quality_rerank=True,
    )
    observation, visible_snippets = format_search_results_with_visible_snippets(
        results,
        retrieval_query,
        max_chars=1800,
        per_result_chars=360,
    )
    hits = visible_evidence_keypoint_hits(results, visible_snippets, keypoints)
    return {
        "hop": hop,
        "model_search_query": model_query,
        "retrieval_query": retrieval_query,
        "observation": observation,
        "hits": hits,
        "top_k_results": _visible_result_records(results, visible_snippets),
    }


def _fill_trajectory(
    source: Mapping[str, Any],
    index: MarkdownBM25Index,
    tokenizer,
    official_fingerprints: set[str],
) -> tuple[dict[str, Any] | None, str]:
    expected = str(source.get("expected_answer", ""))
    question_type, keypoints = expected_keypoints(expected)
    if question_type != "fill" or not keypoints:
        return None, "not_fill"
    query = str(source.get("query", "")).strip()
    fingerprint = question_fingerprint(query)
    if fingerprint in official_fingerprints:
        return None, "official_validation_overlap"
    if any(
        alternative in normalize_evidence_text(query)
        for alternatives in keypoints
        for alternative in alternatives
        if alternative
    ):
        return None, "answer_visible_in_question"
    leak_points = open_answer_leak_points(expected)
    if source_question_answer_term_leak(query, leak_points):
        return None, "answer_terms_visible_in_question"

    audit = source.get("_audit") if isinstance(source.get("_audit"), Mapping) else {}
    first_query = str(audit.get("first_query", "")).strip()
    second_query = str(audit.get("second_query", "")).strip()
    if not first_query:
        return None, "missing_first_query"
    if question_keypoint_leak(first_query, query, keypoints):
        return None, "first_query_answer_leak"
    if grounded_answer_term_leak(first_query, query, leak_points):
        return None, "first_query_answer_term_leak"
    bank = str(source.get("bank", ""))
    first = _search_fill_hop(
        index,
        model_query=first_query,
        original_query=query,
        bank=bank,
        keypoints=keypoints,
        hop=1,
    )
    cumulative_hits = set(first["hits"])
    hops = [first]
    if len(cumulative_hits) == len(keypoints):
        second_query = ""
    else:
        if not second_query:
            return None, "incomplete_one_hop_evidence"
        if question_keypoint_leak(second_query, query, keypoints):
            return None, "second_query_answer_leak"
        if grounded_answer_term_leak(
            second_query,
            query + "\n" + str(first["observation"]),
            leak_points,
        ):
            return None, "second_query_answer_term_leak"
        second = _search_fill_hop(
            index,
            model_query=second_query,
            original_query=query,
            bank=bank,
            keypoints=keypoints,
            hop=2,
        )
        new_hits = set(second["hits"]) - cumulative_hits
        if not new_hits:
            return None, "no_second_hop_evidence_gain"
        cumulative_hits.update(second["hits"])
        if len(cumulative_hits) != len(keypoints):
            return None, "incomplete_cumulative_evidence"
        second["new_hits"] = sorted(new_hits)
        hops.append(second)

    messages = build_search_messages(
        query=query,
        expected=expected,
        first_query=first_query,
        first_observation=str(first["observation"]),
        second_query=second_query or None,
        second_observation=(
            str(hops[1]["observation"])
            if len(hops) == 2
            else None
        ),
    )
    issues = validate_messages(messages)
    if issues:
        return None, "message_validation:" + ",".join(issues)
    runtime_messages = _runtime_messages(tokenizer, messages)
    token_length = _token_length(tokenizer, runtime_messages)
    if token_length > MAX_TOKENS:
        return None, "trajectory_too_long"

    for hop in hops:
        hop["hits"] = sorted(hop["hits"])
        if "new_hits" not in hop:
            hop["new_hits"] = list(hop["hits"])
    return (
        {
            "source_row_id": int(source["row_id"]),
            "question_fingerprint": fingerprint,
            "question_type": "fill",
            "query": query,
            "bank": bank,
            "expected_answer": expected,
            "search_turns": len(hops),
            "search_hops": hops,
            "split": str(source["split"]),
            "messages": runtime_messages,
            "machine_verified": True,
            "human_reviewed": False,
            "_audit": {
                "trusted_visible_coverage": 1.0,
                "incremental_two_hop": len(hops) == 2,
                "query_leakage_check": True,
                "official_validation_fingerprint_overlap": False,
                "token_length": token_length,
                "runtime_raw_chunk_alignment": True,
            },
        },
        "accepted",
    )


def _objective_record(
    source: Mapping[str, Any],
    official_fingerprints: set[str],
    tokenizer,
) -> tuple[dict[str, Any] | None, str]:
    question_type = str(source.get("question_type", ""))
    if question_type not in OBJECTIVE_TYPES:
        return None, "not_objective"
    query = str(source.get("query", "")).strip()
    expected = str(source.get("expected_answer", ""))
    try:
        canonical_type, _answer = canonical_answer(expected)
    except ValueError:
        return None, "invalid_objective_answer"
    if canonical_type != question_type:
        return None, "objective_type_mismatch"
    fingerprint = question_fingerprint(query)
    if fingerprint in official_fingerprints:
        return None, "official_validation_overlap"
    messages = _runtime_messages(
        tokenizer,
        build_objective_messages(query=query, expected=expected),
    )
    token_length = _token_length(tokenizer, messages)
    if token_length > MAX_TOKENS:
        return None, "trajectory_too_long"
    return (
        {
            **dict(source),
            "source_row_id": int(source["row_id"]),
            "question_fingerprint": fingerprint,
            "split": str(source["split"]),
            "messages": [dict(message) for message in messages],
            "_audit": {
                **dict(source.get("_audit") or {}),
                "official_validation_fingerprint_overlap": False,
                "token_length": token_length,
                "runtime_raw_chunk_alignment": True,
            },
        },
        "accepted",
    )


def _deduplicate(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for record in records:
        fingerprint = str(record["question_fingerprint"])
        candidate = dict(record)
        previous = selected.get(fingerprint)
        if previous is not None and previous["split"] != candidate["split"]:
            raise ValueError(
                "question fingerprint crosses splits before deduplication: "
                f"{fingerprint}"
            )
        if previous is None or int(candidate["source_row_id"]) < int(previous["source_row_id"]):
            selected[fingerprint] = candidate
    return sorted(selected.values(), key=lambda record: int(record["source_row_id"]))


def _human_review_samples(shorts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_turns: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in shorts:
        by_turns[int(record["search_turns"])].append(record)
    selected = []
    for turns, limit in ((2, 8), (1, 12)):
        selected.extend(
            sorted(
                by_turns[turns],
                key=lambda record: str(record["question_fingerprint"]),
            )[:limit]
        )
    if len(selected) < min(10, len(shorts)):
        selected = sorted(
            shorts,
            key=lambda record: str(record["question_fingerprint"]),
        )[:20]
    return [
        {
            "source_row_id": record["source_row_id"],
            "question_fingerprint": record["question_fingerprint"],
            "split": record["split"],
            "query": record["query"],
            "answer_points": record["answer_points"],
            "search_turns": record["search_turns"],
            "search_hops": record["search_hops"],
            "final_completion": record["messages"][-1]["content"],
            "human_review_checklist": {
                "all_points_answer_question": None,
                "all_quotes_literal_and_sufficient": None,
                "all_supports_trusted_and_visible": None,
                "queries_do_not_leak_answer": None,
                "second_hop_is_incremental": None,
                "complete_readable_answer": None,
                "decision": "pending_human_review",
            },
        }
        for record in selected[:20]
    ]


def _count_by(records: Sequence[Mapping[str, Any]], *keys: str) -> dict[str, int]:
    counts = Counter(
        ":".join(str(record.get(key)) for key in keys)
        for record in records
    )
    return dict(sorted(counts.items()))


def main() -> None:
    _, overrides = _parse_args()
    output_dir = _output_dir(overrides)
    rebuild_dir = _resolve_rebuild_dir()
    rebuild_summary_path = rebuild_dir / "summary.json"
    rebuild_manifest_path = rebuild_dir / "machine_verified_targets.jsonl"
    v1_manifest_path = Path(
        os.environ.get("QA_SFT_V1_MANIFEST", str(DEFAULT_V1_MANIFEST))
    )
    data_dir = Path(os.environ.get("QA_RL_DATA_DIR", "/data/datasets/qa_rl"))
    docs_dir = Path(os.environ.get("QA_DOCS_DIR", "/data/docs"))
    val_path = data_dir / "val.jsonl"
    required = [
        rebuild_summary_path,
        rebuild_manifest_path,
        v1_manifest_path,
        val_path,
        docs_dir,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing SFT v2 data inputs: {missing}")

    rebuild_summary = _read_json(rebuild_summary_path)
    machine_gate = rebuild_summary.get("training_gate")
    if not isinstance(machine_gate, Mapping) or machine_gate.get("passed_machine_gate") is not True:
        raise RuntimeError("short target rebuild did not pass its machine gate")
    shorts = _read_jsonl(rebuild_manifest_path)
    tokenizer = _load_tokenizer()
    invalid_shorts = []
    for record in shorts:
        issues = short_target_issues(
            record,
            expected_initial_prompt=format_agent_prompt(
                tokenizer,
                str(record.get("query", "")),
            ),
        )
        if issues:
            invalid_shorts.append(
                {
                    "source_row_id": record.get("source_row_id"),
                    "issues": issues,
                }
            )
    if invalid_shorts:
        raise RuntimeError(f"rebuilt short targets failed re-audit: {invalid_shorts[:3]}")

    official_rows = _read_jsonl(val_path)
    official_fingerprints = {
        question_fingerprint(str(record["query"]))
        for record in official_rows
    }
    overlap = {
        str(record["question_fingerprint"])
        for record in shorts
    } & official_fingerprints
    if overlap:
        raise RuntimeError(f"short targets overlap official validation: {len(overlap)}")

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
        f"[sft-v2-data] indexed={index.num_documents} seconds={index_seconds:.1f}",
        flush=True,
    )

    v1_records = _read_jsonl(v1_manifest_path)
    fill_candidates = [
        record
        for record in v1_records
        if record.get("question_type") == "fill"
    ]
    fill_rejections: Counter[str] = Counter()
    fill_audit = []
    fills = []
    for position, source in enumerate(fill_candidates, start=1):
        record, reason = _fill_trajectory(
            source,
            index,
            tokenizer,
            official_fingerprints,
        )
        fill_rejections[reason] += int(reason != "accepted")
        fill_audit.append(
            {
                "source_row_id": source.get("row_id"),
                "query": source.get("query"),
                "decision": reason,
            }
        )
        if record:
            fills.append(record)
        if position % 25 == 0:
            print(
                f"[sft-v2-data] fill audit {position}/{len(fill_candidates)} "
                f"accepted={len(fills)}",
                flush=True,
            )
    fills = _deduplicate(fills)
    open_records = [*shorts, *fills]

    objective_candidates = []
    objective_rejections: Counter[str] = Counter()
    open_fingerprints = {
        str(record["question_fingerprint"])
        for record in open_records
    }
    for source in v1_records:
        record, reason = _objective_record(
            source,
            official_fingerprints,
            tokenizer,
        )
        if reason == "not_objective":
            continue
        if record and record["question_fingerprint"] in open_fingerprints:
            record, reason = None, "open_question_overlap"
        objective_rejections[reason] += int(reason != "accepted")
        if record:
            objective_candidates.append(record)
    objective_candidates = _deduplicate(objective_candidates)
    objective_train = select_balanced_objective_replay(
        [
            record
            for record in objective_candidates
            if record["split"] == "train"
        ],
        open_train_count=sum(
            record["split"] == "train"
            for record in open_records
        ),
    )
    objective_validation = select_objective_validation(
        [
            record
            for record in objective_candidates
            if record["split"] == "validation"
        ],
        per_type=2,
    )
    all_records = [
        *open_records,
        *objective_train,
        *objective_validation,
    ]
    assert_question_split_isolation(all_records)
    replay_fraction = objective_replay_fraction(all_records)
    if not 0.25 <= replay_fraction <= 0.35:
        raise RuntimeError(f"objective replay fraction is outside gate: {replay_fraction}")

    train_manifest = [
        record
        for record in all_records
        if record["split"] == "train"
    ]
    validation_manifest = [
        record
        for record in all_records
        if record["split"] == "validation"
    ]
    holdout_manifest = [
        record
        for record in all_records
        if record["split"] == "rl_holdout"
    ]
    if not train_manifest or not validation_manifest or not holdout_manifest:
        raise RuntimeError("SFT v2 split unexpectedly produced an empty partition")

    rng = random.Random(42)
    rng.shuffle(train_manifest)
    rng.shuffle(validation_manifest)
    paths = {
        "train": output_dir / "sft_v2_train.jsonl",
        "validation": output_dir / "sft_v2_validation.jsonl",
        "short_holdout": output_dir / "curated_short_holdout.jsonl",
        "fill_holdout": output_dir / "curated_fill_holdout.jsonl",
        "manifest": output_dir / "trajectory_manifest.jsonl",
        "fill_audit": output_dir / "fill_retrieval_audit.jsonl",
        "human_review": output_dir / "short_human_review_samples.jsonl",
        "summary": output_dir / "summary.json",
    }
    _write_jsonl(
        paths["train"],
        [{"messages": record["messages"]} for record in train_manifest],
    )
    _write_jsonl(
        paths["validation"],
        [{"messages": record["messages"]} for record in validation_manifest],
    )
    _write_jsonl(
        paths["short_holdout"],
        [
            record
            for record in holdout_manifest
            if record["question_type"] == "short"
        ],
    )
    _write_jsonl(
        paths["fill_holdout"],
        [
            record
            for record in holdout_manifest
            if record["question_type"] == "fill"
        ],
    )
    _write_jsonl(paths["manifest"], all_records)
    _write_jsonl(paths["fill_audit"], fill_audit)
    review_samples = _human_review_samples(shorts)
    _write_jsonl(paths["human_review"], review_samples)

    token_lengths = [
        int(record["_audit"]["token_length"])
        for record in all_records
    ]
    short_two_hop = sum(
        record["question_type"] == "short" and record["search_turns"] == 2
        for record in shorts
    )
    summary = {
        "sources": {
            "short_rebuild_dir": str(rebuild_dir),
            "short_manifest_sha256": _sha256(rebuild_manifest_path),
            "v1_manifest": str(v1_manifest_path),
            "v1_manifest_sha256": _sha256(v1_manifest_path),
            "official_validation": str(val_path),
            "docs": str(docs_dir),
        },
        "short_target_count": len(shorts),
        "short_two_hop_count": short_two_hop,
        "fill_candidates": len(fill_candidates),
        "verified_fill_count": len(fills),
        "fill_rejection_counts": dict(sorted(fill_rejections.items())),
        "objective_rejection_counts": dict(sorted(objective_rejections.items())),
        "record_counts": _count_by(all_records, "split", "question_type"),
        "search_turn_counts": _count_by(open_records, "question_type", "search_turns"),
        "objective_replay_fraction": replay_fraction,
        "official_validation_overlap_count": 0,
        "question_fingerprint_count": len(
            {record["question_fingerprint"] for record in all_records}
        ),
        "token_lengths": {
            "min": min(token_lengths),
            "mean": mean(token_lengths),
            "max": max(token_lengths),
        },
        "human_review": {
            "sample_count": len(review_samples),
            "required": True,
            "passed": False,
            "sft_v2_ready_count": 0,
        },
        "machine_training_gate": {
            "passed": (
                len(shorts) >= 24
                and short_two_hop >= 4
                and any(record["split"] == "train" for record in fills)
                and any(record["split"] == "validation" for record in fills)
                and any(record["split"] == "rl_holdout" for record in fills)
                and 0.25 <= replay_fraction <= 0.35
                and max(token_lengths) <= MAX_TOKENS
            ),
            "human_review_still_required": True,
        },
        "timing_seconds": {"index_build": index_seconds},
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if summary["machine_training_gate"]["passed"] is not True:
        raise RuntimeError(f"SFT v2 data machine gate failed: {summary}")
    print("[sft-v2-data] summary", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("[sft-v2-data] pending human-review samples", flush=True)
    print(json.dumps(review_samples[:3], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
