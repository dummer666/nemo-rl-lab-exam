"""Strict open-answer filtering and balanced mixed GRPO curriculum."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from common.retrieval.evidence import (
    expected_keypoints,
    fragile_keypoint_indexes,
    visible_evidence_keypoint_hits,
)
from common.retrieval.markdown_bm25 import (
    MarkdownBM25Index,
    build_retrieval_query,
    format_search_results_with_visible_snippets,
    question_context,
)
from common.retrieval.qa_short_audit import parse_short_gold
from common.retrieval.qa_target_rebuild import question_fingerprint


def _stable_key(seed: int, fingerprint: str) -> str:
    return hashlib.sha256(f"{seed}:{fingerprint}".encode()).hexdigest()


def _source_row_id(row: Mapping[str, Any], fallback: int) -> int:
    clean = row.get("_clean") if isinstance(row.get("_clean"), Mapping) else {}
    return int(clean.get("row_id", fallback))


def _support_level(row: Mapping[str, Any]) -> str:
    clean = row.get("_clean") if isinstance(row.get("_clean"), Mapping) else {}
    return str(clean.get("support_level", "none"))


def strict_open_candidates(
    rows: Sequence[Mapping[str, Any]],
    index: MarkdownBM25Index,
    excluded_fingerprints: set[str],
    excluded_row_ids: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Keep only full-support labels visible in the production Top-4 snippets."""
    accepted: dict[str, list[dict[str, Any]]] = {"fill": [], "short": []}
    rejection_counts: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    for fallback, source in enumerate(rows):
        query = str(source.get("query", "")).strip()
        expected = str(source.get("expected_answer", "")).strip()
        question_type, keypoints = expected_keypoints(expected)
        if question_type not in accepted:
            continue
        source_row_id = _source_row_id(source, fallback)
        fingerprint = question_fingerprint(query)
        if source_row_id in excluded_row_ids:
            reject("prior_sft_row")
            continue
        if fingerprint in excluded_fingerprints:
            reject("fingerprint_overlap")
            continue
        if _support_level(source) != "full":
            reject(f"{question_type}:not_full_support")
            continue
        if not keypoints or fragile_keypoint_indexes(keypoints):
            reject(f"{question_type}:unsafe_keypoints")
            continue
        if question_type == "short":
            bank = str(
                (source.get("meta") or {}).get("bank", "")
                if isinstance(source.get("meta"), Mapping)
                else ""
            )
            parsed = parse_short_gold(expected, query=query, bank=bank)
            if (
                parsed["label_issue_codes"]
                or parsed["defective_keypoint_indexes"]
                or not 2 <= int(parsed["keypoint_count"]) <= 6
            ):
                reject("short:label_defect")
                continue

        metadata = source.get("meta") if isinstance(source.get("meta"), Mapping) else {}
        bank = str(metadata.get("bank", ""))
        retrieval_query = build_retrieval_query(
            question_context(query),
            query,
            bank,
        )
        results = index.search(
            retrieval_query,
            top_k=4,
            candidate_k=50,
            quality_rerank=True,
        )
        _rendered, snippets = format_search_results_with_visible_snippets(
            results,
            retrieval_query,
            max_chars=1800,
            per_result_chars=360,
        )
        visible_hits = visible_evidence_keypoint_hits(
            results,
            snippets,
            keypoints,
        )
        if len(visible_hits) != len(keypoints):
            reject(f"{question_type}:not_fully_visible")
            continue
        accepted[question_type].append(
            {
                **dict(source),
                "source_row_id": source_row_id,
                "question_fingerprint": fingerprint,
                "_open_audit": {
                    "question_type": question_type,
                    "keypoint_count": len(keypoints),
                    "visible_hits": sorted(visible_hits),
                    "retrieval_query": retrieval_query,
                    "sources": [result.source for result in results],
                },
            }
        )

    for question_type in accepted:
        accepted[question_type].sort(
            key=lambda row: (
                str(row["question_fingerprint"]),
                int(row["source_row_id"]),
            )
        )
    return accepted["fill"], accepted["short"], rejection_counts


def build_mixed_open_curriculum(
    objective_rows: Sequence[Mapping[str, Any]],
    fill_rows: Sequence[Mapping[str, Any]],
    short_rows: Sequence[Mapping[str, Any]],
    *,
    total_steps: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Use three unique objectives plus alternating fill/short per step."""
    required_objectives = total_steps * 3
    objectives = sorted(
        (dict(row) for row in objective_rows),
        key=lambda row: _stable_key(
            seed,
            str(row["question_fingerprint"]),
        ),
    )
    fills = sorted(
        (dict(row) for row in fill_rows),
        key=lambda row: _stable_key(
            seed + 1,
            str(row["question_fingerprint"]),
        ),
    )
    shorts = sorted(
        (dict(row) for row in short_rows),
        key=lambda row: _stable_key(
            seed + 2,
            str(row["question_fingerprint"]),
        ),
    )
    if len(objectives) < required_objectives:
        raise RuntimeError(
            f"insufficient isolated objective rows: "
            f"{len(objectives)} < {required_objectives}"
        )
    if not fills:
        raise RuntimeError("no strict fill rows available")
    if not shorts:
        raise RuntimeError("no strict short rows available")

    curriculum = []
    objective_offset = 0
    open_offsets = {"fill": 0, "short": 0}
    open_pools = {"fill": fills, "short": shorts}
    for step in range(1, total_steps + 1):
        for slot in range(3):
            selected = dict(objectives[objective_offset])
            objective_offset += 1
            selected["_curriculum"] = {
                "step": step,
                "slot": f"objective:{slot}",
                "phase": "strict_mixed_open",
                "force_search": False,
                "minimum_searches": 0,
                "source_row_id": int(selected["source_row_id"]),
            }
            curriculum.append(selected)

        open_type = "fill" if step % 2 else "short"
        pool = open_pools[open_type]
        offset = open_offsets[open_type]
        selected = dict(pool[offset % len(pool)])
        open_offsets[open_type] = offset + 1
        selected["_curriculum"] = {
            "step": step,
            "slot": open_type,
            "phase": "strict_mixed_open",
            "force_search": True,
            "minimum_searches": 1,
            "source_row_id": int(selected["source_row_id"]),
        }
        curriculum.append(selected)
    return curriculum
