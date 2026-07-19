#!/usr/bin/env python
"""Build a two-hop-oversampled fill SFT pack without changing source splits."""

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
TWO_HOP_REPEAT = 4
EXPECTED_TRAIN_PROFILE = {
    "objective": 24,
    "fill_one_hop": 31,
    "fill_two_hop": 28,
}
EXPECTED_VALIDATION_PROFILE = {
    "objective": 6,
    "fill_one_hop": 4,
    "fill_two_hop": 1,
}
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
            output = Path(override.split("=", 1)[1]).parent / "fill_sft_v3_data"
            output.mkdir(parents=True, exist_ok=True)
            return output
    output = THIS_DIR / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _profile(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        if record.get("question_type") != "fill":
            counts["objective"] += 1
        elif int(record.get("search_turns", 0)) == 1:
            counts["fill_one_hop"] += 1
        elif int(record.get("search_turns", 0)) == 2:
            counts["fill_two_hop"] += 1
        else:
            counts["invalid_fill"] += 1
    return dict(sorted(counts.items()))


def build_oversampled_pack(
    accepted_fill: Sequence[Mapping[str, Any]],
    objective_candidates: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    fill_counts = Counter(str(record.get("split", "")) for record in accepted_fill)
    if dict(fill_counts) != EXPECTED_FILL_SPLITS:
        raise ValueError(
            f"fill split changed: expected={EXPECTED_FILL_SPLITS}, actual={dict(fill_counts)}"
        )
    train_fill = [
        dict(record)
        for record in accepted_fill
        if record.get("split") == "train"
    ]
    validation_fill = [
        dict(record)
        for record in accepted_fill
        if record.get("split") == "validation"
    ]
    holdout_fill = [
        dict(record)
        for record in accepted_fill
        if record.get("split") == "rl_holdout"
    ]
    one_hop = [
        record
        for record in train_fill
        if int(record.get("search_turns", 0)) == 1
    ]
    two_hop = [
        record
        for record in train_fill
        if int(record.get("search_turns", 0)) == 2
    ]
    if len(one_hop) != 31 or len(two_hop) != 7:
        raise ValueError(
            f"unexpected train fill profile: one_hop={len(one_hop)}, two_hop={len(two_hop)}"
        )

    oversampled_two_hop = []
    for repeat_index in range(TWO_HOP_REPEAT):
        for record in two_hop:
            oversampled_two_hop.append(
                {
                    **record,
                    "_sampling": {
                        "strategy": "two_hop_repeat",
                        "repeat_index": repeat_index,
                        "repeat_count": TWO_HOP_REPEAT,
                    },
                }
            )
    one_hop_exposures = [
        {
            **record,
            "_sampling": {
                "strategy": "single_exposure",
                "repeat_index": 0,
                "repeat_count": 1,
            },
        }
        for record in one_hop
    ]
    open_exposure_count = len(one_hop_exposures) + len(oversampled_two_hop)
    objective_train = select_balanced_objective_replay(
        [
            record
            for record in objective_candidates
            if record.get("split") == "train"
        ],
        open_train_count=open_exposure_count,
    )
    objective_validation = select_objective_validation(
        [
            record
            for record in objective_candidates
            if record.get("split") == "validation"
        ],
        per_type=2,
    )

    unique_records = [
        *train_fill,
        *validation_fill,
        *holdout_fill,
        *objective_train,
        *objective_validation,
    ]
    assert_question_split_isolation(unique_records)
    train = [
        *one_hop_exposures,
        *oversampled_two_hop,
        *objective_train,
    ]
    validation = [*validation_fill, *objective_validation]
    rng = random.Random(84)
    rng.shuffle(train)
    rng.shuffle(validation)

    profiles = {
        "train": _profile(train),
        "validation": _profile(validation),
        "rl_holdout": _profile(holdout_fill),
    }
    if profiles["train"] != EXPECTED_TRAIN_PROFILE:
        raise ValueError(
            f"unexpected train exposure profile: {profiles['train']}"
        )
    if profiles["validation"] != EXPECTED_VALIDATION_PROFILE:
        raise ValueError(
            f"unexpected validation profile: {profiles['validation']}"
        )
    replay_fraction = objective_replay_fraction(train)
    if not 0.25 <= replay_fraction <= 0.35:
        raise ValueError(f"objective replay fraction outside gate: {replay_fraction}")
    return {
        "train": train,
        "validation": validation,
        "rl_holdout": holdout_fill,
        "unique_manifest": unique_records,
    }, {
        "profiles": profiles,
        "two_hop_repeat": TWO_HOP_REPEAT,
        "unique_train_fill_count": len(train_fill),
        "train_fill_exposure_count": open_exposure_count,
        "objective_replay_fraction": replay_fraction,
        "objective_train_per_type": dict(
            sorted(
                Counter(
                    str(record["question_type"])
                    for record in objective_train
                ).items()
            )
        ),
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
        raise FileNotFoundError(f"missing fill SFT v3 inputs: {missing}")

    accepted_fill = _read_jsonl(accepted_path)
    audit_summary = _read_json(audit_summary_path)
    if int(audit_summary.get("accepted_count", -1)) != len(accepted_fill):
        raise RuntimeError("fill audit summary count does not match its manifest")
    issues: Counter[str] = Counter()
    for record in accepted_fill:
        issues.update(_accepted_record_issues(record))
    if issues:
        raise RuntimeError(f"accepted fill manifest has issues: {dict(issues)}")
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
    pack, pack_summary = build_oversampled_pack(
        accepted_fill,
        objective_candidates,
    )
    all_exposures = [*pack["train"], *pack["validation"], *pack["rl_holdout"]]
    token_lengths = [
        int(record.get("_audit", {}).get("token_length", MAX_TOKENS + 1))
        for record in all_exposures
    ]
    if not token_lengths or max(token_lengths) > MAX_TOKENS:
        raise RuntimeError("fill SFT v3 contains an over-length trajectory")

    paths = {
        "train": output_dir / "fill_sft_v3_train.jsonl",
        "validation": output_dir / "fill_sft_v3_validation.jsonl",
        "holdout": output_dir / "fill_sft_v3_holdout.jsonl",
        "exposure_manifest": output_dir / "fill_sft_v3_exposure_manifest.jsonl",
        "unique_manifest": output_dir / "fill_sft_v3_unique_manifest.jsonl",
        "summary": output_dir / "summary.json",
    }
    _write_jsonl(
        paths["train"],
        [{"messages": record["messages"]} for record in pack["train"]],
    )
    _write_jsonl(
        paths["validation"],
        [{"messages": record["messages"]} for record in pack["validation"]],
    )
    _write_jsonl(paths["holdout"], pack["rl_holdout"])
    _write_jsonl(paths["exposure_manifest"], all_exposures)
    _write_jsonl(paths["unique_manifest"], pack["unique_manifest"])

    summary = {
        "mode": "two_hop_oversampled_fill_sft_v3",
        "sources": {
            "fill_audit_dir": str(fill_audit_dir),
            "accepted_fill_sha256": _sha256(accepted_path),
            "v1_manifest": str(v1_manifest_path),
            "v1_manifest_sha256": _sha256(v1_manifest_path),
            "official_validation": str(official_path),
        },
        **pack_summary,
        "output_counts": {
            "train": len(pack["train"]),
            "validation": len(pack["validation"]),
            "rl_holdout": len(pack["rl_holdout"]),
            "unique_manifest": len(pack["unique_manifest"]),
        },
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
            "expected_train_profile": EXPECTED_TRAIN_PROFILE,
            "maximum_tokens": MAX_TOKENS,
        },
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("[fill-sft-v3-data] summary", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
