#!/usr/bin/env python
"""Select evidence-grounded open QA candidates for multi-turn retrieval SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.retrieval.evidence import (  # noqa: E402
    evidence_keypoint_hits,
    expected_keypoints,
    text_keypoint_hits,
)
from common.retrieval.markdown_bm25 import (  # noqa: E402
    MarkdownBM25Index,
    SearchResult,
    build_retrieval_query,
    format_search_results,
    question_context,
)
from common.retrieval.qa_sft import visible_retrieval_text  # noqa: E402

DEFAULT_CLEAN_TRAIN_PATH = (
    "/shared/outputs/wanghaonan/qa_training_clean_wanghaonan/"
    "qa_training_clean_wanghaonan-wanghaonan-20260718-092437/"
    "cleaned_data/clean_train.jsonl"
)
OPEN_TYPES = {"fill", "short"}


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Select retrieval SFT candidates")
    parser.add_argument("--config")
    return parser.parse_known_args()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _output_dir(overrides: Sequence[str]) -> Path:
    for override in overrides:
        if override.startswith("logger.log_dir="):
            output = Path(override.split("=", 1)[1]).parent / "sft_selection"
            output.mkdir(parents=True, exist_ok=True)
            return output
    output = THIS_DIR / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _coverage(hits: set[int], keypoints: Sequence[Sequence[str]]) -> float:
    return len(hits) / len(keypoints) if keypoints else 0.0


def _support_level(coverage: float) -> str:
    if coverage >= 1.0:
        return "full"
    return "partial" if coverage > 0.0 else "none"


def _selection_status(displayed_coverage: float, pool_coverage: float) -> str:
    """Classify candidates by evidence actually visible to the deployed agent."""
    if displayed_coverage >= 1.0:
        return "ready_one_search"
    if pool_coverage >= 1.0:
        return "needs_query_rewrite"
    if max(displayed_coverage, pool_coverage) > 0.0:
        return "partial_review"
    return "excluded_unsupported"


def _dataset_role(status: str) -> str:
    if status in {"ready_one_search", "needs_query_rewrite"}:
        return "primary"
    if status == "partial_review":
        return "secondary"
    return "excluded"


def _result_hits(
    result: SearchResult,
    keypoints: Sequence[Sequence[str]],
) -> set[int]:
    return evidence_keypoint_hits([result], keypoints, top_k=1)


def _result_record(
    rank: int,
    result: SearchResult,
    keypoints: Sequence[Sequence[str]],
) -> dict:
    return {
        "rank": rank,
        "source": result.source,
        "heading": result.heading,
        "quality_category": result.quality_category,
        "quality_weight": result.quality_weight,
        "raw_score": result.raw_score,
        "keypoint_hits": sorted(_result_hits(result, keypoints)),
        "text": result.text,
    }


def _greedy_support_results(
    results: Sequence[SearchResult],
    keypoints: Sequence[Sequence[str]],
) -> list[dict]:
    """Keep the smallest high-ranked set that covers all retrievable keypoints."""
    candidates = []
    for rank, result in enumerate(results, start=1):
        hits = _result_hits(result, keypoints)
        if hits:
            candidates.append((rank, result, hits))

    selected: list[dict] = []
    covered: set[int] = set()
    while candidates:
        rank, result, hits = max(
            candidates,
            key=lambda item: (
                len(item[2] - covered),
                item[1].quality_weight,
                float(item[1].raw_score or 0.0),
                -item[0],
            ),
        )
        new_hits = hits - covered
        if not new_hits:
            break
        selected.append(_result_record(rank, result, keypoints))
        covered.update(hits)
        candidates = [candidate for candidate in candidates if candidate[0] != rank]
    return selected


def _candidate_record(
    row: dict,
    index: MarkdownBM25Index,
    *,
    first_top_k: int = 4,
    pool_top_k: int = 20,
    candidate_k: int = 50,
    max_result_chars: int = 1800,
    per_result_chars: int = 360,
) -> dict:
    expected = str(row.get("expected_answer", ""))
    question_type, keypoints = expected_keypoints(expected)
    if question_type not in OPEN_TYPES or not keypoints:
        raise ValueError("candidate row must have non-empty fill or short keypoints")

    query = str(row.get("query", "")).strip()
    metadata = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    clean = row.get("_clean") if isinstance(row.get("_clean"), dict) else {}
    bank = str(metadata.get("bank", ""))
    first_search_query = question_context(query)[:256]
    retrieval_query = build_retrieval_query(first_search_query, query, bank)
    results = index.search(
        retrieval_query,
        top_k=pool_top_k,
        candidate_k=candidate_k,
        quality_rerank=True,
    )
    first_results = results[:first_top_k]
    first_retrieval_output = format_search_results(
        first_results,
        retrieval_query,
        max_chars=max_result_chars,
        per_result_chars=per_result_chars,
    )
    displayed_hits = text_keypoint_hits(
        visible_retrieval_text(first_retrieval_output),
        keypoints,
    )
    pool_hits = evidence_keypoint_hits(results, keypoints, top_k=pool_top_k)
    displayed_coverage = _coverage(displayed_hits, keypoints)
    pool_coverage = _coverage(pool_hits, keypoints)
    status = _selection_status(displayed_coverage, pool_coverage)
    declared_support = str(clean.get("support_level", "unknown"))
    recomputed_support = _support_level(pool_coverage)

    return {
        "row_id": clean.get("row_id"),
        "question_type": question_type,
        "query": query,
        "expected_answer": expected,
        "bank": bank,
        "keypoints": keypoints,
        "declared_support_level": declared_support,
        "recomputed_support_level": recomputed_support,
        "support_level_mismatch": declared_support != recomputed_support,
        "selection_status": status,
        "dataset_role": _dataset_role(status),
        "recommended_search_turns": (
            1
            if status == "ready_one_search"
            else 2
            if status == "needs_query_rewrite"
            else None
        ),
        "first_search_query": first_search_query,
        "retrieval_query": retrieval_query,
        "first_observation_coverage": displayed_coverage,
        "top20_pool_coverage": pool_coverage,
        "first_observation_hits": sorted(displayed_hits),
        "top20_pool_hits": sorted(pool_hits),
        "first_retrieval_output": first_retrieval_output,
        "first_results": [
            _result_record(rank, result, keypoints)
            for rank, result in enumerate(first_results, start=1)
        ],
        "support_results": _greedy_support_results(results, keypoints),
    }


def _split_key(record: dict, seed: int) -> str:
    identity = f"{seed}:{record.get('row_id')}:{record.get('query', '')}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _assign_primary_splits(
    records: Sequence[dict],
    *,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> list[dict]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")

    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[str(record["question_type"])].append(dict(record))

    assigned: list[dict] = []
    for group in groups.values():
        ordered = sorted(group, key=lambda record: _split_key(record, seed))
        validation_count = round(len(ordered) * validation_fraction)
        if validation_fraction > 0 and len(ordered) >= 2:
            validation_count = max(1, min(len(ordered) - 1, validation_count))
        validation_ids = {
            record.get("row_id")
            for record in ordered[:validation_count]
        }
        for record in group:
            record["split"] = (
                "validation" if record.get("row_id") in validation_ids else "train"
            )
            assigned.append(record)
    return sorted(assigned, key=lambda record: int(record.get("row_id") or -1))


def _mean(records: Sequence[dict], key: str) -> float:
    values = [float(record[key]) for record in records]
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    _, overrides = _parse_args()
    clean_train_path = Path(
        os.environ.get("QA_CLEAN_TRAIN_PATH", DEFAULT_CLEAN_TRAIN_PATH)
    )
    docs_dir = Path(os.environ.get("QA_DOCS_DIR", "/data/docs"))
    if not clean_train_path.is_file():
        raise FileNotFoundError(f"Clean training data does not exist: {clean_train_path}")

    rows = _read_jsonl(clean_train_path)
    open_rows = [
        row
        for row in rows
        if expected_keypoints(str(row.get("expected_answer", "")))[0] in OPEN_TYPES
    ]
    index_start = time.perf_counter()
    index = MarkdownBM25Index.from_directory(
        docs_dir,
        chunk_chars=1200,
        overlap_chars=160,
        k1=1.5,
        b=0.75,
    )
    index_seconds = time.perf_counter() - index_start

    selection_start = time.perf_counter()
    records = []
    for position, row in enumerate(open_rows, start=1):
        records.append(_candidate_record(row, index))
        if position % 100 == 0:
            print(f"[sft-selection] processed {position}/{len(open_rows)}")
    selection_seconds = time.perf_counter() - selection_start

    primary = _assign_primary_splits(
        [record for record in records if record["dataset_role"] == "primary"],
        validation_fraction=0.1,
        seed=42,
    )
    primary_by_id = {record["row_id"]: record for record in primary}
    finalized = []
    for record in records:
        if record["dataset_role"] == "primary":
            finalized.append(primary_by_id[record["row_id"]])
        else:
            finalized.append(
                {
                    **record,
                    "split": "review" if record["dataset_role"] == "secondary" else "excluded",
                }
            )
    finalized.sort(key=lambda record: int(record.get("row_id") or -1))

    primary_train = [record for record in finalized if record["split"] == "train"]
    primary_validation = [
        record for record in finalized if record["split"] == "validation"
    ]
    secondary = [record for record in finalized if record["split"] == "review"]
    output_dir = _output_dir(overrides)
    paths = {
        "manifest": output_dir / "selection_manifest.jsonl",
        "primary_train": output_dir / "primary_train_candidates.jsonl",
        "primary_validation": output_dir / "primary_validation_candidates.jsonl",
        "secondary_review": output_dir / "secondary_review_candidates.jsonl",
        "summary": output_dir / "summary.json",
    }
    _write_jsonl(paths["manifest"], finalized)
    _write_jsonl(paths["primary_train"], primary_train)
    _write_jsonl(paths["primary_validation"], primary_validation)
    _write_jsonl(paths["secondary_review"], secondary)

    summary = {
        "source": str(clean_train_path),
        "input_clean_rows": len(rows),
        "open_rows": len(open_rows),
        "question_type_counts": dict(
            sorted(Counter(record["question_type"] for record in finalized).items())
        ),
        "declared_support_counts": dict(
            sorted(Counter(record["declared_support_level"] for record in finalized).items())
        ),
        "recomputed_support_counts": dict(
            sorted(Counter(record["recomputed_support_level"] for record in finalized).items())
        ),
        "selection_status_counts": dict(
            sorted(Counter(record["selection_status"] for record in finalized).items())
        ),
        "dataset_role_counts": dict(
            sorted(Counter(record["dataset_role"] for record in finalized).items())
        ),
        "split_counts": dict(
            sorted(Counter(record["split"] for record in finalized).items())
        ),
        "primary_type_counts": dict(
            sorted(Counter(record["question_type"] for record in primary).items())
        ),
        "support_level_mismatches": sum(
            bool(record["support_level_mismatch"]) for record in finalized
        ),
        "mean_first_observation_coverage": _mean(
            finalized,
            "first_observation_coverage",
        ),
        "mean_top20_pool_coverage": _mean(finalized, "top20_pool_coverage"),
        "quality_category_counts": index.quality_category_counts,
        "timing_seconds": {
            "index_build": index_seconds,
            "candidate_selection": selection_seconds,
            "selection_per_open_row": selection_seconds / max(1, len(open_rows)),
        },
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[sft-selection] summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
