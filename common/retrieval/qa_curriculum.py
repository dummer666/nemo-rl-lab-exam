"""Deterministic, evidence-aware curriculum construction for QA GRPO."""

from __future__ import annotations

import random
import re
from collections.abc import Sequence

_EXPECTED_TYPE = re.compile(r"^\s*\[(\w+)\]")
_OBJECTIVE_TYPES = {"single", "multiple", "bool"}


def question_type(row: dict) -> str:
    match = _EXPECTED_TYPE.match(str(row.get("expected_answer", "")))
    return match.group(1).lower() if match else "unknown"


def _sample_weight(row: dict) -> float:
    clean = row.get("_clean") if isinstance(row.get("_clean"), dict) else {}
    try:
        return max(0.0, float(clean.get("sample_weight", 1.0)))
    except (TypeError, ValueError):
        return 0.0


def _support_level(row: dict) -> str:
    clean = row.get("_clean") if isinstance(row.get("_clean"), dict) else {}
    return str(clean.get("support_level", "not_applicable"))


class _WeightedCycle:
    def __init__(self, rows: Sequence[dict], rng: random.Random, name: str):
        self.rows = [row for row in rows if _sample_weight(row) > 0]
        if not self.rows:
            raise ValueError(f"curriculum pool is empty: {name}")
        self.rng = rng
        self.remaining: list[dict] = []

    def take(self) -> dict:
        if not self.remaining:
            self.remaining = sorted(
                self.rows,
                key=lambda row: self.rng.random() ** (1.0 / _sample_weight(row)),
            )
        return self.remaining.pop()


def _annotate(row: dict, *, step: int, slot: str, phase: str, force_search: bool) -> dict:
    selected = dict(row)
    selected["_curriculum"] = {
        "step": step,
        "slot": slot,
        "phase": phase,
        "force_search": force_search,
    }
    return selected


def build_v3_curriculum(
    rows: Sequence[dict],
    *,
    warmup_steps: int = 10,
    total_steps: int = 30,
    prompts_per_step: int = 4,
    seed: int = 42,
) -> list[dict]:
    """Build 2+1+1 search warmup batches, then 3+1 mixed batches."""
    if not 0 <= warmup_steps <= total_steps:
        raise ValueError("warmup_steps must be between zero and total_steps")
    if prompts_per_step != 4:
        raise ValueError("v3 curriculum requires exactly four prompts per step")

    rng = random.Random(seed)
    objective = [row for row in rows if question_type(row) in _OBJECTIVE_TYPES]
    supported_fill = [
        row
        for row in rows
        if question_type(row) == "fill" and _support_level(row) in {"full", "partial"}
    ]
    supported_short = [
        row
        for row in rows
        if question_type(row) == "short" and _support_level(row) in {"full", "partial"}
    ]
    fill = [row for row in rows if question_type(row) == "fill"]
    short = [row for row in rows if question_type(row) == "short"]

    pools = {
        "objective": _WeightedCycle(objective, rng, "objective"),
        "supported_fill": _WeightedCycle(supported_fill, rng, "supported_fill"),
        "supported_short": _WeightedCycle(supported_short, rng, "supported_short"),
        "fill": _WeightedCycle(fill, rng, "fill"),
        "short": _WeightedCycle(short, rng, "short"),
    }

    curriculum: list[dict] = []
    for step in range(1, total_steps + 1):
        if step <= warmup_steps:
            phase = "search_warmup"
            selections = [
                ("objective", pools["objective"].take(), False),
                ("objective", pools["objective"].take(), False),
                ("fill", pools["supported_fill"].take(), True),
                ("short", pools["supported_short"].take(), True),
            ]
        else:
            phase = "mixed"
            open_type = "fill" if (step - warmup_steps) % 2 else "short"
            selections = [
                ("objective", pools["objective"].take(), False),
                ("objective", pools["objective"].take(), False),
                ("objective", pools["objective"].take(), False),
                (open_type, pools[open_type].take(), False),
            ]
        curriculum.extend(
            _annotate(
                row,
                step=step,
                slot=slot,
                phase=phase,
                force_search=force_search,
            )
            for slot, row, force_search in selections
        )

    return curriculum
