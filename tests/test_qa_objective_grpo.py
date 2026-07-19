from common.retrieval.qa_objective_grpo import select_objective_curriculum


def _candidate(question_type, index):
    return {
        "question_type": question_type,
        "question_fingerprint": f"{question_type}-{index}",
        "source_row_id": index,
        "query": f"question {index}",
        "expected_answer": "[single] A",
    }


def test_select_objective_curriculum_is_isolated_and_balanced():
    candidates = [
        _candidate(question_type, index)
        for question_type in ("single", "multiple", "bool")
        for index in range(10)
    ]

    rows = select_objective_curriculum(
        candidates,
        {"single-0"},
        total_steps=2,
        seed=7,
    )

    assert len(rows) == 8
    assert [row["question_type"] for row in rows] == [
        "single",
        "single",
        "multiple",
        "bool",
        "single",
        "single",
        "multiple",
        "bool",
    ]
    assert len({row["question_fingerprint"] for row in rows}) == 8
    assert all(row["question_fingerprint"] != "single-0" for row in rows)
