from experiments.qa_objective_sft_review_wanghaonan import run


def test_review_issues_accepts_valid_objective_row():
    row = {
        "query": "Question\nA. first\nB. second",
        "expected_answer": "[single] B",
        "messages": [
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": "分析后，\\boxed{B}"},
        ],
        "_audit": {"official_validation_overlap": False},
    }

    assert run.review_issues(row) == []


def test_review_issues_rejects_out_of_range_answer():
    row = {
        "query": "Question\nA. first\nB. second",
        "expected_answer": "[multiple] AC",
        "messages": [
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": "分析后，\\boxed{AC}"},
        ],
        "_audit": {"official_validation_overlap": False},
    }

    assert "answer_out_of_range" in run.review_issues(row)
