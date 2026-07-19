"""Build a question-isolated curriculum for short post-SFT retrieval GRPO."""

from __future__ import annotations

import random
from collections.abc import Sequence

from common.retrieval.qa_curriculum import question_type

_OBJECTIVE_TYPES = {"single", "multiple", "bool"}
_OPEN_TYPES = {"fill", "short"}


class _Cycle:
    def __init__(self, rows: Sequence[dict], rng: random.Random, name: str):
        self.rows = list(rows)
        if not self.rows:
            raise ValueError(f"short-GRPO pool is empty: {name}")
        self.rng = rng
        self.remaining: list[dict] = []

    def take(self) -> dict:
        if not self.remaining:
            self.remaining = list(self.rows)
            self.rng.shuffle(self.remaining)
        return dict(self.remaining.pop())


def _take_excluding(
    primary: _Cycle,
    fallback: _Cycle,
    excluded_ids: set[int],
) -> dict:
    for pool in (primary, fallback):
        for _ in range(len(pool.rows)):
            row = pool.take()
            if _holdout_row_id(row) not in excluded_ids:
                return row
    raise ValueError("not enough distinct RL holdout questions for one training batch")


def _take_distinct(
    pool: _Cycle,
    count: int,
    row_id,
    name: str,
) -> list[dict]:
    unique_ids = {row_id(row) for row in pool.rows}
    if len(unique_ids) < count:
        raise ValueError(
            f"not enough distinct {name} questions for one training batch: "
            f"need {count}, found {len(unique_ids)}"
        )

    selected = []
    selected_ids = set()
    for _ in range(count):
        for _ in range(2 * len(pool.rows)):
            row = pool.take()
            source_row_id = row_id(row)
            if source_row_id not in selected_ids:
                selected.append(row)
                selected_ids.add(source_row_id)
                break
        else:
            raise ValueError(
                f"could not draw {count} distinct {name} questions"
            )
    return selected


def _clean_row_id(row: dict) -> int:
    clean = row.get("_clean") if isinstance(row.get("_clean"), dict) else {}
    if "row_id" not in clean:
        raise ValueError("clean training row is missing _clean.row_id")
    return int(clean["row_id"])


def _holdout_row_id(row: dict) -> int:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    if "source_row_id" not in meta:
        raise ValueError("RL holdout row is missing meta.source_row_id")
    return int(meta["source_row_id"])


def _annotate(
    row: dict,
    *,
    step: int,
    slot: str,
    source_row_id: int,
    minimum_searches: int,
) -> dict:
    selected = dict(row)
    selected["_curriculum"] = {
        "step": step,
        "slot": slot,
        "phase": "post_sft_retrieval_refinement",
        "source_row_id": source_row_id,
        "force_search": minimum_searches > 0,
        "minimum_searches": minimum_searches,
    }
    return selected


def build_short_grpo_curriculum(
    rl_holdout: Sequence[dict],
    clean_rows: Sequence[dict],
    trajectory_manifest: Sequence[dict],
    *,
    total_steps: int = 20,
    prompts_per_step: int = 4,
    seed: int = 42,
) -> list[dict]:
    """Mix three isolated open questions with one unseen objective per step."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if prompts_per_step != 4:
        raise ValueError("short GRPO requires exactly four prompts per step")

    sft_row_ids = {int(row["row_id"]) for row in trajectory_manifest if row.get("split") in {"train", "validation"}}
    holdout_ids = {_holdout_row_id(row) for row in rl_holdout}
    overlap = sft_row_ids & holdout_ids
    if overlap:
        raise ValueError(f"RL holdout overlaps SFT questions: {sorted(overlap)}")

    objective = [
        row
        for row in clean_rows
        if question_type(row) in _OBJECTIVE_TYPES
        and _clean_row_id(row) not in sft_row_ids
        and _clean_row_id(row) not in holdout_ids
    ]
    open_rows = [row for row in rl_holdout if question_type(row) in _OPEN_TYPES]
    fill = [row for row in open_rows if question_type(row) == "fill"]
    short = [row for row in open_rows if question_type(row) == "short"]
    two_search = [row for row in open_rows if int((row.get("meta") or {}).get("search_turns", 1)) == 2]
    if len(open_rows) != len(rl_holdout):
        raise ValueError("RL holdout contains non-open questions")

    rng = random.Random(seed)
    pools = {
        "objective": _Cycle(objective, rng, "objective"),
        "fill": _Cycle(fill, rng, "fill"),
        "short": _Cycle(short, rng, "short"),
        "open": _Cycle(open_rows, rng, "open"),
        "two_search": _Cycle(two_search, rng, "two_search"),
    }

    curriculum = []
    for step in range(1, total_steps + 1):
        objective_row = pools["objective"].take()
        fill_row = pools["fill"].take()
        short_row = pools["short"].take()
        bonus_row = _take_excluding(
            pools["two_search"] if step % 2 else pools["open"],
            pools["open"],
            {
                _holdout_row_id(fill_row),
                _holdout_row_id(short_row),
            },
        )
        bonus_meta = bonus_row.get("meta") if isinstance(bonus_row.get("meta"), dict) else {}
        bonus_slot = "two_search" if int(bonus_meta.get("search_turns", 1)) == 2 else "open"
        selections = [
            ("objective", objective_row),
            ("fill", fill_row),
            ("short", short_row),
            (bonus_slot, bonus_row),
        ]
        for slot, row in selections:
            if slot == "objective":
                source_row_id = _clean_row_id(row)
                minimum_searches = 0
            else:
                source_row_id = _holdout_row_id(row)
                meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
                minimum_searches = int(meta.get("search_turns", 1))
            curriculum.append(
                _annotate(
                    row,
                    step=step,
                    slot=slot,
                    source_row_id=source_row_id,
                    minimum_searches=minimum_searches,
                )
            )

    selected_ids = {int(row["_curriculum"]["source_row_id"]) for row in curriculum}
    leaked = selected_ids & sft_row_ids
    if leaked:
        raise ValueError(f"short-GRPO curriculum leaked SFT questions: {sorted(leaked)}")
    return curriculum


def build_balanced_open_grpo_curriculum(
    rl_holdout: Sequence[dict],
    clean_rows: Sequence[dict],
    trajectory_manifest: Sequence[dict],
    *,
    total_steps: int = 100,
    prompts_per_step: int = 4,
    seed: int = 42,
) -> list[dict]:
    """Mix 50% objective, 25% fill, and 25% short prompts per step."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if prompts_per_step <= 0 or prompts_per_step % 4:
        raise ValueError(
            "balanced open GRPO prompts_per_step must be a positive multiple of four"
        )

    sft_row_ids = {
        int(row["row_id"])
        for row in trajectory_manifest
        if row.get("split") in {"train", "validation"}
    }
    holdout_ids = {_holdout_row_id(row) for row in rl_holdout}
    overlap = sft_row_ids & holdout_ids
    if overlap:
        raise ValueError(
            f"RL holdout overlaps SFT questions: {sorted(overlap)}"
        )

    objective = [
        row
        for row in clean_rows
        if question_type(row) in _OBJECTIVE_TYPES
        and _clean_row_id(row) not in sft_row_ids
        and _clean_row_id(row) not in holdout_ids
    ]
    fill = [
        row for row in rl_holdout if question_type(row) == "fill"
    ]
    short = [
        row for row in rl_holdout if question_type(row) == "short"
    ]
    if len(fill) + len(short) != len(rl_holdout):
        raise ValueError("RL holdout contains non-open questions")

    rng = random.Random(seed)
    pools = {
        "objective": _Cycle(objective, rng, "objective"),
        "fill": _Cycle(fill, rng, "fill"),
        "short": _Cycle(short, rng, "short"),
    }
    objective_count = prompts_per_step // 2
    open_count = prompts_per_step // 4
    curriculum = []
    for step in range(1, total_steps + 1):
        objective_rows = _take_distinct(
            pools["objective"],
            objective_count,
            _clean_row_id,
            "objective",
        )
        fill_rows = _take_distinct(
            pools["fill"],
            open_count,
            _holdout_row_id,
            "fill",
        )
        short_rows = _take_distinct(
            pools["short"],
            open_count,
            _holdout_row_id,
            "short",
        )
        selections = [
            *[
                (f"objective:{index}", row)
                for index, row in enumerate(objective_rows)
            ],
            *[
                (
                    "fill" if open_count == 1 else f"fill:{index}",
                    row,
                )
                for index, row in enumerate(fill_rows)
            ],
            *[
                (
                    "short" if open_count == 1 else f"short:{index}",
                    row,
                )
                for index, row in enumerate(short_rows)
            ],
        ]
        for slot, row in selections:
            if slot.startswith("objective"):
                source_row_id = _clean_row_id(row)
                minimum_searches = 0
            else:
                source_row_id = _holdout_row_id(row)
                meta = (
                    row.get("meta")
                    if isinstance(row.get("meta"), dict)
                    else {}
                )
                minimum_searches = max(
                    1,
                    min(2, int(meta.get("search_turns", 1))),
                )
            curriculum.append(
                _annotate(
                    row,
                    step=step,
                    slot=slot,
                    source_row_id=source_row_id,
                    minimum_searches=minimum_searches,
                )
            )

    selected_ids = {
        int(row["_curriculum"]["source_row_id"]) for row in curriculum
    }
    leaked = selected_ids & sft_row_ids
    if leaked:
        raise ValueError(
            f"balanced open GRPO leaked SFT questions: {sorted(leaked)}"
        )
    return curriculum
