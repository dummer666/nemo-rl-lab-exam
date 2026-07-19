from experiments.qa_grounded_cloze_sft_holdout_eval_wanghaonan import (
    run,
)


def _row(searches: int) -> dict:
    messages = [{"role": "user", "content": "q"}]
    for _ in range(searches):
        messages.extend(
            [
                {"role": "assistant", "content": "<search>x</search>"},
                {"role": "environment", "content": "e"},
            ]
        )
    messages.append({"role": "assistant", "content": "\\boxed{x}"})
    return {"messages": messages}


def test_expected_searches_and_profile_summary():
    rows = [_row(0), _row(1), _row(2)]
    results = [
        {"row_index": 0, "reward": 1.0, "search_count": 0},
        {"row_index": 1, "reward": 0.5, "search_count": 1},
        {"row_index": 2, "reward": 0.0, "search_count": 1},
    ]

    assert [run.expected_searches(row) for row in rows] == [0, 1, 2]
    summary = run.profile_summary(rows, results)
    assert summary["0"]["perfect_count"] == 1
    assert summary["1"]["accuracy"] == 0.5
    assert summary["2"]["matched_search_count"] == 0
