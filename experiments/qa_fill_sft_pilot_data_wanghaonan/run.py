#!/usr/bin/env python
"""Build the user-authorized fill-only SFT pilot pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.retrieval.qa_sft_v2 import (  # noqa: E402
    assert_question_split_isolation,
    objective_replay_fraction,
    select_balanced_objective_replay,
    select_objective_validation,
)
from common.retrieval.qa_target_rebuild import question_fingerprint  # noqa: E402
from experiments.qa_fill_only_audit_wanghaonan.run import (  # noqa: E402
    _accepted_record_issues,
)
from experiments.qa_sft_v2_data_build_wanghaonan.run import (  # noqa: E402
    DEFAULT_V1_MANIFEST,
    MAX_TOKENS,
    _deduplicate,
    _load_tokenizer,
    _objective_record,
)

DEFAULT_FILL_AUDIT_DIR = Path(
    "/shared/outputs/wanghaonan/qa_fill_only_audit_wanghaonan/"
    "qa_fill_only_audit_wanghaonan-wanghaonan-20260719-025356/"
    "fill_only_audit"
)
EXPECTED_FILL_SPLITS = {"train": 38, "validation": 5, "rl_holdout": 7}
EXPECTED_OBJECTIVE_TRAIN = 15
EXPECTED_OBJECTIVE_VALIDATION = 6
EXPECTED_OFFICIAL_QUESTIONS = 313


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
            output = Path(override.split("=", 1)[1]).parent / "fill_sft_pilot_data"
            output.mkdir(parents=True, exist_ok=True)
            return output
    output = THIS_DIR / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _count_by(
    records: Sequence[Mapping[str, Any]],
    *keys: str,
) -> dict[str, int]:
    counts = Counter(
        ":".join(str(record.get(key, "")) for key in keys)
        for record in records
    )
    return dict(sorted(counts.items()))


def build_pilot_manifests(
    accepted_fill: Sequence[Mapping[str, Any]],
    objective_candidates: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    fill_counts = Counter(str(record.get("split", "")) for record in accepted_fill)
    if dict(fill_counts) != EXPECTED_FILL_SPLITS:
        raise ValueError(
            f"fill split changed: expected={EXPECTED_FILL_SPLITS}, actual={dict(fill_counts)}"
        )
    train_two_hop = sum(
        record.get("split") == "train"
        and int(record.get("search_turns", 0)) == 2
        for record in accepted_fill
    )
    if train_two_hop < 1:
        raise ValueError("fill training split has no two-hop trajectory")

    objective_train = select_balanced_objective_replay(
        [
            record
            for record in objective_candidates
            if record.get("split") == "train"
        ],
        open_train_count=EXPECTED_FILL_SPLITS["train"],
    )
    objective_validation = select_objective_validation(
        [
            record
            for record in objective_candidates
            if record.get("split") == "validation"
        ],
        per_type=2,
    )
    if len(objective_train) != EXPECTED_OBJECTIVE_TRAIN:
        raise ValueError("unexpected objective train replay count")
    if len(objective_validation) != EXPECTED_OBJECTIVE_VALIDATION:
        raise ValueError("unexpected objective validation replay count")

    all_records = [
        *[dict(record) for record in accepted_fill],
        *objective_train,
        *objective_validation,
    ]
    assert_question_split_isolation(all_records)
    replay_fraction = objective_replay_fraction(all_records)
    if not 0.25 <= replay_fraction <= 0.35:
        raise ValueError(f"objective replay fraction outside gate: {replay_fraction}")

    manifests = {
        split: [
            dict(record)
            for record in all_records
            if record.get("split") == split
        ]
        for split in ("train", "validation", "rl_holdout")
    }
    rng = random.Random(42)
    rng.shuffle(manifests["train"])
    rng.shuffle(manifests["validation"])
    return manifests, {
        "fill_split_counts": dict(fill_counts),
        "fill_search_turn_counts": _count_by(
            accepted_fill,
            "split",
            "search_turns",
        ),
        "objective_train_counts": _count_by(
            objective_train,
            "question_type",
        ),
        "objective_validation_counts": _count_by(
            objective_validation,
            "question_type",
        ),
        "objective_replay_fraction": replay_fraction,
        "output_split_counts": {
            split: len(records)
            for split, records in manifests.items()
        },
    }


def main() -> None:
    _, overrides = _parse_args()
    output_dir = _output_dir(overrides)
    fill_audit_dir = Path(
        os.environ.get("QA_FILL_AUDIT_DIR", str(DEFAULT_FILL_AUDIT_DIR))
    )
    accepted_path = fill_audit_dir / "accepted_fill_manifest.jsonl"
    audit_summary_path = fill_audit_dir / "summary.json"
    v1_manifest_path = Path(
        os.environ.get("QA_SFT_V1_MANIFEST", str(DEFAULT_V1_MANIFEST))
    )
    data_dir = Path(os.environ.get("QA_RL_DATA_DIR", "/data/datasets/qa_rl"))
    official_path = data_dir / "val.jsonl"
    missing = [
        str(path)
        for path in (
            accepted_path,
            audit_summary_path,
            v1_manifest_path,
            official_path,
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing fill pilot inputs: {missing}")

    accepted_fill = _read_jsonl(accepted_path)
    audit_summary = _read_json(audit_summary_path)
    if int(audit_summary.get("accepted_count", -1)) != len(accepted_fill):
        raise RuntimeError("fill audit summary count does not match its manifest")
    issue_counts: Counter[str] = Counter()
    for record in accepted_fill:
        issue_counts.update(_accepted_record_issues(record))
    if issue_counts:
        raise RuntimeError(f"accepted fill manifest has issues: {dict(issue_counts)}")
    assert_question_split_isolation(accepted_fill)

    official_rows = _read_jsonl(official_path)
    official_fingerprints = {
        question_fingerprint(str(row.get("query", "")))
        for row in official_rows
    }
    if (
        len(official_rows) != EXPECTED_OFFICIAL_QUESTIONS
        or len(official_fingerprints) != EXPECTED_OFFICIAL_QUESTIONS
    ):
        raise RuntimeError("official validation integrity check failed")
    fill_fingerprints = {
        str(record["question_fingerprint"])
        for record in accepted_fill
    }
    if fill_fingerprints & official_fingerprints:
        raise RuntimeError("fill manifest overlaps official validation")

    tokenizer = _load_tokenizer()
    objective_candidates = []
    for source in _read_jsonl(v1_manifest_path):
        record, reason = _objective_record(
            source,
            official_fingerprints,
            tokenizer,
        )
        if reason == "not_objective":
            continue
        if record is not None and record["question_fingerprint"] in fill_fingerprints:
            continue
        if record is not None:
            objective_candidates.append(record)
    objective_candidates = _deduplicate(objective_candidates)

    manifests, pack_summary = build_pilot_manifests(
        accepted_fill,
        objective_candidates,
    )
    all_records = [
        *manifests["train"],
        *manifests["validation"],
        *manifests["rl_holdout"],
    ]
    token_lengths = [
        int(record.get("_audit", {}).get("token_length", MAX_TOKENS + 1))
        for record in all_records
    ]
    if not token_lengths or max(token_lengths) > MAX_TOKENS:
        raise RuntimeError("fill pilot contains an over-length trajectory")

    paths = {
        "train": output_dir / "fill_pilot_train.jsonl",
        "validation": output_dir / "fill_pilot_validation.jsonl",
        "holdout": output_dir / "fill_pilot_holdout.jsonl",
        "manifest": output_dir / "fill_pilot_manifest.jsonl",
        "summary": output_dir / "summary.json",
    }
    _write_jsonl(
        paths["train"],
        [{"messages": record["messages"]} for record in manifests["train"]],
    )
    _write_jsonl(
        paths["validation"],
        [{"messages": record["messages"]} for record in manifests["validation"]],
    )
    _write_jsonl(paths["holdout"], manifests["rl_holdout"])
    _write_jsonl(paths["manifest"], all_records)
    summary = {
        "mode": "user_authorized_fill_sft_pilot",
        "sources": {
            "fill_audit_dir": str(fill_audit_dir),
            "accepted_fill_sha256": _sha256(accepted_path),
            "v1_manifest": str(v1_manifest_path),
            "v1_manifest_sha256": _sha256(v1_manifest_path),
            "official_validation": str(official_path),
        },
        **pack_summary,
        "token_lengths": {
            "min": min(token_lengths),
            "max": max(token_lengths),
        },
        "leaked_fill_rows_restored": 0,
        "source_splits_changed": False,
        "official_validation_overlap_count": 0,
        "human_reviewed": False,
        "training_authorized_by_user": True,
        "machine_gate": {
            "passed": True,
            "requires_train_two_hop": True,
            "maximum_tokens": MAX_TOKENS,
        },
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("[fill-sft-pilot-data] summary", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
