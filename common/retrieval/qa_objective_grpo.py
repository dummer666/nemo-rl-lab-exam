"""Deterministic objective-only GRPO curriculum selection."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

OBJECTIVE_TYPES = ("single", "multiple", "bool")
SLOTS = ("single", "single", "multiple", "bool")


def _stable_key(seed: int, fingerprint: str) -> str:
    return hashlib.sha256(f"{seed}:{fingerprint}".encode()).hexdigest()


def select_objective_curriculum(
    candidates: Sequence[Mapping[str, Any]],
    excluded_fingerprints: set[str],
    *,
    total_steps: int,
    seed: int,
) -> list[dict[str, Any]]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        fingerprint = str(candidate["question_fingerprint"])
        if fingerprint not in excluded_fingerprints:
            pools[str(candidate["question_type"])].append(dict(candidate))
    required = Counter(SLOTS * total_steps)
    for offset, question_type in enumerate(OBJECTIVE_TYPES):
        pools[question_type] = sorted(
            pools[question_type],
            key=lambda row: _stable_key(
                seed + offset,
                str(row["question_fingerprint"]),
            ),
        )
        if len(pools[question_type]) < required[question_type]:
            raise RuntimeError(
                f"insufficient isolated {question_type} rows: "
                f"{len(pools[question_type])} < {required[question_type]}"
            )

    offsets = Counter()
    curriculum = []
    for step in range(1, total_steps + 1):
        for slot_index, question_type in enumerate(SLOTS):
            selected = dict(pools[question_type][offsets[question_type]])
            offsets[question_type] += 1
            selected["_curriculum"] = {
                "step": step,
                "slot": f"{slot_index}:{question_type}",
                "phase": "objective_only",
                "force_search": False,
                "minimum_searches": 0,
                "source_row_id": selected["source_row_id"],
            }
            curriculum.append(selected)
    return curriculum
