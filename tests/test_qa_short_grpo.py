from collections import Counter

from common.retrieval.qa_short_grpo import build_short_grpo_curriculum


def _holdout(row_id: int, question_type: str, search_turns: int) -> dict:
    return {
        "query": f"{question_type}-{row_id}",
        "expected_answer": f"[{question_type}] answer",
        "meta": {
            "source_row_id": row_id,
            "question_type": question_type,
            "search_turns": search_turns,
            "bank": "",
        },
    }


def _clean(row_id: int, question_type: str) -> dict:
    return {
        "query": f"{question_type}-{row_id}",
        "expected_answer": f"[{question_type}] A",
        "_clean": {"row_id": row_id},
    }


def test_short_grpo_curriculum_is_disjoint_balanced_and_two_hop_aware():
    holdout = [
        _holdout(100, "fill", 1),
        _holdout(101, "fill", 2),
        _holdout(200, "short", 1),
        _holdout(201, "short", 2),
        _holdout(202, "short", 1),
    ]
    clean_rows = [
        _clean(1, "single"),
        *[_clean(row_id, "single") for row_id in range(300, 308)],
        *[_clean(row_id, "multiple") for row_id in range(400, 408)],
        *[_clean(row_id, "bool") for row_id in range(500, 508)],
    ]
    manifest = [
        {"row_id": 1, "split": "train"},
        {"row_id": 2, "split": "validation"},
    ]

    curriculum = build_short_grpo_curriculum(
        holdout,
        clean_rows,
        manifest,
        total_steps=4,
        prompts_per_step=4,
    )

    assert len(curriculum) == 16
    assert not {row["_curriculum"]["source_row_id"] for row in curriculum}.intersection({1, 2})
    for offset in range(0, len(curriculum), 4):
        batch = curriculum[offset : offset + 4]
        slots = Counter(row["_curriculum"]["slot"] for row in batch)
        assert slots["objective"] == 1
        assert slots["fill"] == 1
        assert slots["short"] == 1
        assert len({row["_curriculum"]["source_row_id"] for row in batch}) == 4
        open_rows = [row for row in batch if row["_curriculum"]["slot"] != "objective"]
        assert all(row["_curriculum"]["force_search"] for row in open_rows)
    assert all(
        row["_curriculum"]["minimum_searches"] == 2 for row in curriculum if row["_curriculum"]["slot"] == "two_search"
    )
