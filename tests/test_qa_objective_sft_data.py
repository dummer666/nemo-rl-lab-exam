from experiments.qa_objective_sft_data_wanghaonan import run


def _candidate(question_type: str, index: int) -> dict:
    fingerprint = f"{question_type}-{index:03d}"
    return {
        "source_row_id": index,
        "question_fingerprint": fingerprint,
        "question_type": question_type,
        "query": f"question {fingerprint}",
        "expected_answer": f"[{question_type}] A",
        "answer": "A",
        "bank": "",
    }


def test_select_objectives_is_disjoint_and_sized():
    candidates = [
        _candidate(question_type, index)
        for question_type in run.OBJECTIVE_TYPES
        for index in range(10)
    ]
    train, validation, available = run.select_objectives(
        candidates,
        {question_type: 4 for question_type in run.OBJECTIVE_TYPES},
        {question_type: 2 for question_type in run.OBJECTIVE_TYPES},
    )

    assert available == {
        question_type: 10 for question_type in run.OBJECTIVE_TYPES
    }
    assert len(train) == 12
    assert len(validation) == 6
    assert {
        row["question_fingerprint"] for row in train
    }.isdisjoint(
        row["question_fingerprint"] for row in validation
    )


def test_profile_separates_objective_and_retrieval_rows():
    rows = [
        {
            "question_type": "single",
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "\\boxed{A}"},
            ],
        },
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "<search>x</search>"},
                {"role": "environment", "content": "e"},
                {"role": "assistant", "content": "\\boxed{x}"},
            ],
        },
    ]

    assert run._profile(rows) == {
        "objective:single": 1,
        "retrieval:1": 1,
    }


def test_objective_quality_rejects_metadata_after_all_above():
    query = (
        "题目：有哪些系统？\n"
        "A. GMS\nB. CCTV\nC. VESDA\nD. 以上都是\nE. 较难"
    )

    assert run.objective_quality_issues(query) == [
        "metadata_option",
        "all_above_not_last",
    ]


def test_objective_quality_accepts_normal_options():
    query = "题目：正确的是？\nA. 第一项\nB. 第二项\nC. 以上都是"

    assert run.objective_quality_issues(query) == []
