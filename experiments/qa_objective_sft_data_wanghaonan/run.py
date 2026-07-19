#!/usr/bin/env python
"""Build a clean objective-heavy SFT pack with retrieval replay."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.retrieval.evidence import normalize_evidence_text  # noqa: E402
from common.retrieval.markdown_bm25 import question_context  # noqa: E402
from common.retrieval.qa_sft import (  # noqa: E402
    build_objective_messages,
    canonical_answer,
)
from common.retrieval.qa_target_rebuild import question_fingerprint  # noqa: E402
from experiments.qa_sft_v2_data_build_wanghaonan.run import (  # noqa: E402
    _load_tokenizer,
    _runtime_messages,
    _token_length,
)

CLEAN_TRAIN_PATH = Path(
    "/shared/outputs/wanghaonan/qa_training_clean_wanghaonan/"
    "qa_training_clean_wanghaonan-wanghaonan-20260718-092437/"
    "cleaned_data/clean_train.jsonl"
)
OFFICIAL_VALIDATION_PATH = Path("/data/datasets/qa_rl/val.jsonl")
RETRIEVAL_SFT_ROOT = Path(
    "/shared/outputs/wanghaonan/qa_sft_trajectory_build_wanghaonan/"
    "qa_sft_trajectory_build_wanghaonan-wanghaonan-20260718-123640/"
    "sft_trajectories"
)
TRAIN_TARGETS = {"single": 760, "multiple": 340, "bool": 500}
VALIDATION_TARGETS = {"single": 130, "multiple": 58, "bool": 87}
RETRIEVAL_TRAIN_EXPOSURES = 400
MAX_TOKENS = 3072
SEED = 191
OBJECTIVE_TYPES = tuple(TRAIN_TARGETS)
_ANSWER_LETTER = re.compile(r"[A-Z]")
_OPTION = re.compile(r"^\s*([A-Z])\.\s*(.+?)\s*$", re.MULTILINE)
_ALL_ABOVE = re.compile(r"^(?:以上|上述).*(?:都是|都对|均正确|全[部都]?正确)$")
_METADATA_OPTION = re.compile(
    r"^(?:较难|困难|难|中等|一般|简单|容易|答案|解析|无|未知)$",
    re.IGNORECASE,
)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def _output_dir(overrides: Sequence[str]) -> Path:
    for override in overrides:
        if override.startswith("logger.log_dir="):
            output = (
                Path(override.split("=", 1)[1]).parent
                / "objective_sft_data"
            )
            output.mkdir(parents=True, exist_ok=True)
            return output
    output = THIS_DIR / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    return output


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
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _stable_key(seed: int, fingerprint: str) -> str:
    return hashlib.sha256(f"{seed}:{fingerprint}".encode()).hexdigest()


def objective_quality_issues(
    query: str,
    question_type: str | None = None,
    expected_answer: str | None = None,
) -> list[str]:
    issues = []
    options = _OPTION.findall(str(query))
    normalized_options = [
        normalize_evidence_text(text) for _letter, text in options
    ]
    if len(normalized_options) != len(set(normalized_options)):
        issues.append("duplicate_option_text")
    if any(_METADATA_OPTION.fullmatch(text.strip()) for _letter, text in options):
        issues.append("metadata_option")
    if any(
        _ALL_ABOVE.fullmatch(text.strip()) and index != len(options) - 1
        for index, (_letter, text) in enumerate(options)
    ):
        issues.append("all_above_not_last")
    if question_type == "single" and len(options) >= 6:
        issues.append("single_too_many_options")
    if (
        question_type == "multiple"
        and expected_answer is not None
        and len(
            _ANSWER_LETTER.findall(
                expected_answer.split("]", 1)[-1].upper()
            )
        )
        == 1
    ):
        issues.append("multiple_single_answer")
    if len(normalize_evidence_text(question_context(query))) < 6:
        issues.append("question_too_short")
    return issues


def _objective_candidates(
    rows: Sequence[Mapping[str, Any]],
    official_fingerprints: set[str],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rejection_counts: Counter[str] = Counter()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fallback_row_id, row in enumerate(rows):
        query = str(row.get("query", "")).strip()
        expected = str(row.get("expected_answer", "")).strip()
        try:
            question_type, answer = canonical_answer(expected)
        except ValueError:
            rejection_counts["invalid_answer"] += 1
            continue
        if question_type not in OBJECTIVE_TYPES:
            rejection_counts["not_objective"] += 1
            continue
        quality_issues = objective_quality_issues(
            query,
            question_type,
            expected,
        )
        if quality_issues:
            for issue in quality_issues:
                rejection_counts[f"quality:{issue}"] += 1
            continue
        fingerprint = question_fingerprint(query)
        if fingerprint in official_fingerprints:
            rejection_counts["official_overlap"] += 1
            continue
        clean = row.get("_clean") if isinstance(row.get("_clean"), Mapping) else {}
        grouped[fingerprint].append(
            {
                "source_row_id": int(clean.get("row_id", fallback_row_id)),
                "question_fingerprint": fingerprint,
                "question_type": question_type,
                "query": query,
                "expected_answer": expected,
                "answer": answer,
                "bank": str(
                    (row.get("meta") or {}).get("bank", "")
                    if isinstance(row.get("meta"), Mapping)
                    else ""
                ),
            }
        )

    candidates = []
    for group in grouped.values():
        expected_keys = {
            normalize_evidence_text(record["expected_answer"])
            for record in group
        }
        if len(expected_keys) != 1:
            rejection_counts["duplicate_answer_conflict"] += len(group)
            continue
        if len(group) > 1:
            rejection_counts["duplicate_rows"] += len(group) - 1
        candidates.append(min(group, key=lambda record: record["source_row_id"]))
    return candidates, rejection_counts


def select_objectives(
    candidates: Sequence[Mapping[str, Any]],
    train_targets: Mapping[str, int],
    validation_targets: Mapping[str, int],
    *,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_type[str(candidate["question_type"])].append(dict(candidate))

    train = []
    validation = []
    available = {}
    for offset, question_type in enumerate(OBJECTIVE_TYPES):
        group = sorted(
            by_type[question_type],
            key=lambda record: _stable_key(
                seed + offset,
                str(record["question_fingerprint"]),
            ),
        )
        train_target = int(train_targets[question_type])
        validation_target = int(validation_targets[question_type])
        required = train_target + validation_target
        available[question_type] = len(group)
        if len(group) < required:
            raise RuntimeError(
                f"insufficient {question_type}: {len(group)} < {required}"
            )
        validation.extend(
            {**record, "split": "validation"}
            for record in group[:validation_target]
        )
        train.extend(
            {**record, "split": "train"}
            for record in group[
                validation_target : validation_target + train_target
            ]
        )
    return train, validation, available


def _render_objective(record: Mapping[str, Any], tokenizer) -> dict[str, Any]:
    messages = _runtime_messages(
        tokenizer,
        build_objective_messages(
            query=str(record["query"]),
            expected=str(record["expected_answer"]),
        ),
    )
    token_length = _token_length(tokenizer, messages)
    if token_length > MAX_TOKENS:
        raise RuntimeError(
            f"objective row exceeds {MAX_TOKENS} tokens: "
            f"{record['question_fingerprint']}"
        )
    return {
        **dict(record),
        "source_kind": "clean_objective",
        "messages": messages,
        "_audit": {
            "official_validation_overlap": False,
            "token_length": token_length,
            "runtime_raw_chunk_alignment": True,
        },
    }


def _search_turns(row: Mapping[str, Any]) -> int:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("SFT row is missing messages")
    return sum(
        isinstance(message, Mapping) and message.get("role") == "environment"
        for message in messages
    )


def _retrieval_replay(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, int]]:
    unique_train = [
        copy.deepcopy(dict(row))
        for row in train_rows
        if _search_turns(row) > 0
    ]
    validation = [
        copy.deepcopy(dict(row))
        for row in validation_rows
        if _search_turns(row) > 0
    ]
    if not unique_train or not validation:
        raise RuntimeError("retrieval replay source is empty")
    exposures = []
    for exposure in range(RETRIEVAL_TRAIN_EXPOSURES):
        clone = copy.deepcopy(unique_train[exposure % len(unique_train)])
        clone["source_kind"] = "retrieval_replay"
        clone["replay_exposure"] = exposure + 1
        exposures.append(clone)
    profile = Counter(_search_turns(row) for row in unique_train)
    return exposures, validation, dict(sorted(profile.items()))


def _answer_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        answer = "".join(
            sorted(set(_ANSWER_LETTER.findall(str(row["answer"]).upper())))
        )
        counts[f"{row['question_type']}:{answer}"] += 1
    return dict(sorted(counts.items()))


def _profile(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        turns = _search_turns(row)
        if turns:
            counts[f"retrieval:{turns}"] += 1
        else:
            counts[f"objective:{row['question_type']}"] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    _, overrides = _parse_args()
    output_dir = _output_dir(overrides)
    required = [
        CLEAN_TRAIN_PATH,
        OFFICIAL_VALIDATION_PATH,
        RETRIEVAL_SFT_ROOT / "sft_train.jsonl",
        RETRIEVAL_SFT_ROOT / "sft_validation.jsonl",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing objective SFT inputs: {missing}")

    clean_rows = _read_jsonl(CLEAN_TRAIN_PATH)
    official_rows = _read_jsonl(OFFICIAL_VALIDATION_PATH)
    official_fingerprints = {
        question_fingerprint(str(row["query"])) for row in official_rows
    }
    candidates, rejection_counts = _objective_candidates(
        clean_rows,
        official_fingerprints,
    )
    selected_train, selected_validation, available = select_objectives(
        candidates,
        TRAIN_TARGETS,
        VALIDATION_TARGETS,
    )

    tokenizer = _load_tokenizer()
    objective_train = [
        _render_objective(record, tokenizer) for record in selected_train
    ]
    objective_validation = [
        _render_objective(record, tokenizer)
        for record in selected_validation
    ]
    replay_train, replay_validation, replay_unique_profile = _retrieval_replay(
        _read_jsonl(RETRIEVAL_SFT_ROOT / "sft_train.jsonl"),
        _read_jsonl(RETRIEVAL_SFT_ROOT / "sft_validation.jsonl"),
    )

    train = [*objective_train, *replay_train]
    validation = [*objective_validation, *replay_validation]
    random.Random(SEED).shuffle(train)
    random.Random(SEED + 1).shuffle(validation)

    train_objective_fingerprints = {
        str(row["question_fingerprint"]) for row in objective_train
    }
    validation_objective_fingerprints = {
        str(row["question_fingerprint"]) for row in objective_validation
    }
    overlap = train_objective_fingerprints & validation_objective_fingerprints
    official_overlap = (
        train_objective_fingerprints | validation_objective_fingerprints
    ) & official_fingerprints
    if overlap or official_overlap:
        raise RuntimeError(
            f"objective split leakage: split={len(overlap)} "
            f"official={len(official_overlap)}"
        )

    token_lengths = [
        _token_length(tokenizer, row["messages"]) for row in [*train, *validation]
    ]
    if max(token_lengths) > MAX_TOKENS:
        raise RuntimeError(f"pack exceeds {MAX_TOKENS} tokens")
    profile = {
        "train": _profile(train),
        "validation": _profile(validation),
    }
    expected_train = {
        "objective:single": 760,
        "objective:multiple": 340,
        "objective:bool": 500,
        **{
            f"retrieval:{turns}": count
            for turns, count in Counter(
                _search_turns(row) for row in replay_train
            ).items()
        },
    }
    if profile["train"] != dict(sorted(expected_train.items())):
        raise RuntimeError(
            f"unexpected training profile: {profile['train']}"
        )

    review_sample = []
    for question_type in OBJECTIVE_TYPES:
        review_sample.extend(
            sorted(
                (
                    row
                    for row in objective_train
                    if row["question_type"] == question_type
                ),
                key=lambda row: str(row["question_fingerprint"]),
            )[:8]
        )
    summary = {
        "inputs": {
            "clean_train": str(CLEAN_TRAIN_PATH),
            "official_validation": str(OFFICIAL_VALIDATION_PATH),
            "retrieval_sft_root": str(RETRIEVAL_SFT_ROOT),
        },
        "available_unique_objectives": available,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "selected_unique_objectives": {
            "train": dict(Counter(row["question_type"] for row in objective_train)),
            "validation": dict(
                Counter(row["question_type"] for row in objective_validation)
            ),
        },
        "answer_distribution": {
            "train": _answer_distribution(objective_train),
            "validation": _answer_distribution(objective_validation),
        },
        "retrieval_replay": {
            "unique_train_profile": replay_unique_profile,
            "train_exposures": len(replay_train),
            "validation_rows": len(replay_validation),
        },
        "profile": profile,
        "rows": {"train": len(train), "validation": len(validation)},
        "fractions": {
            "objective_train": len(objective_train) / len(train),
            "retrieval_train": len(replay_train) / len(train),
        },
        "isolation": {
            "train_validation_overlap": len(overlap),
            "official_validation_overlap": len(official_overlap),
        },
        "token_lengths": {
            "min": min(token_lengths),
            "mean": mean(token_lengths),
            "max": max(token_lengths),
        },
        "human_reviewed": False,
        "pretraining_behavior_gate_required": True,
        "outputs": {
            "train": str(output_dir / "train.jsonl"),
            "validation": str(output_dir / "validation.jsonl"),
            "objective_train_manifest": str(
                output_dir / "objective_train_manifest.jsonl"
            ),
            "objective_validation_manifest": str(
                output_dir / "objective_validation_manifest.jsonl"
            ),
            "review_sample": str(output_dir / "review_sample.jsonl"),
        },
    }
    _write_jsonl(output_dir / "train.jsonl", train)
    _write_jsonl(output_dir / "validation.jsonl", validation)
    _write_jsonl(
        output_dir / "objective_train_manifest.jsonl",
        objective_train,
    )
    _write_jsonl(
        output_dir / "objective_validation_manifest.jsonl",
        objective_validation,
    )
    _write_jsonl(output_dir / "review_sample.jsonl", review_sample)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("[objective-sft-data]", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[objective-sft-data] saved={output_dir}", flush=True)


if __name__ == "__main__":
    main()
