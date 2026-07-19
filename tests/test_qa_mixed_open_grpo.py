from common.retrieval.qa_mixed_open_grpo import build_mixed_open_curriculum


def _row(question_type, index):
    return {
        "source_row_id": index,
        "question_fingerprint": f"{question_type}-{index}",
        "question_type": question_type,
        "query": f"question {index}",
        "expected_answer": f"[{question_type}] A",
    }


def test_mixed_open_curriculum_balances_and_forces_open_search():
    rows = build_mixed_open_curriculum(
        [_row("single", index) for index in range(6)],
        [_row("fill", 100)],
        [_row("short", 200)],
        total_steps=2,
        seed=7,
    )

    assert len(rows) == 8
    assert [row["_curriculum"]["slot"] for row in rows] == [
        "objective:0",
        "objective:1",
        "objective:2",
        "fill",
        "objective:0",
        "objective:1",
        "objective:2",
        "short",
    ]
    assert len(
        {
            row["question_fingerprint"]
            for row in rows
            if row["_curriculum"]["slot"].startswith("objective")
        }
    ) == 6
    assert rows[3]["_curriculum"]["minimum_searches"] == 1
    assert rows[7]["_curriculum"]["minimum_searches"] == 1
