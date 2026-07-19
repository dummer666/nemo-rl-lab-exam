from experiments.qa_objective_sft_delta_review_wanghaonan import run


def _row(reward):
    return {
        "row_index": 1,
        "question_type": "single",
        "query": "Question",
        "expected_answer": "[single] A",
        "reward": reward,
        "assistant_responses": [f"reward={reward}"],
        "search_count": 0,
    }


def test_changed_rows_reports_reward_delta():
    assert run.changed_rows([_row(0.0)], [_row(1.0)]) == [
        {
            "row_index": 1,
            "question_type": "single",
            "query": "Question",
            "expected_answer": "[single] A",
            "baseline": {
                "reward": 0.0,
                "responses": ["reward=0.0"],
                "search_count": 0,
            },
            "candidate": {
                "reward": 1.0,
                "responses": ["reward=1.0"],
                "search_count": 0,
            },
            "reward_delta": 1.0,
        }
    ]
