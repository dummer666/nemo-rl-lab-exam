from __future__ import annotations

from collections import Counter

from common.retrieval.qa_curriculum import build_v3_curriculum, question_type


def _row(row_id: int, kind: str, support: str = "not_applicable", weight: float = 1.0) -> dict:
    return {
        "query": f"question-{row_id}",
        "expected_answer": f"[{kind}] answer-{row_id}",
        "_clean": {
            "row_id": row_id,
            "support_level": support,
            "sample_weight": weight,
        },
    }


def test_v3_curriculum_has_expected_step_composition():
    rows = [_row(index, "single") for index in range(20)]
    rows += [_row(100 + index, "fill", "full", 3.0) for index in range(6)]
    rows += [_row(200 + index, "short", "partial", 2.0) for index in range(6)]

    curriculum = build_v3_curriculum(
        rows,
        warmup_steps=2,
        total_steps=4,
        prompts_per_step=4,
        seed=7,
    )

    assert len(curriculum) == 16
    assert curriculum == build_v3_curriculum(
        rows,
        warmup_steps=2,
        total_steps=4,
        prompts_per_step=4,
        seed=7,
    )

    for step in range(1, 5):
        batch = curriculum[(step - 1) * 4 : step * 4]
        counts = Counter(question_type(row) for row in batch)
        if step <= 2:
            assert counts == {"single": 2, "fill": 1, "short": 1}
            assert [row["_curriculum"]["force_search"] for row in batch] == [
                False,
                False,
                True,
                True,
            ]
        else:
            assert counts["single"] == 3
            assert counts["fill"] + counts["short"] == 1
            assert not any(row["_curriculum"]["force_search"] for row in batch)


def test_v3_curriculum_rejects_missing_supported_pool():
    rows = [_row(index, "single") for index in range(8)]
    rows += [_row(100, "fill", "none", 0.25), _row(200, "short", "full", 3.0)]

    try:
        build_v3_curriculum(rows, warmup_steps=1, total_steps=1)
    except ValueError as exc:
        assert "supported_fill" in str(exc)
    else:
        raise AssertionError("missing supported fill pool should fail")
