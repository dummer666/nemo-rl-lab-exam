from experiments.qa_fill_sft_v3_multiturn_eval_wanghaonan import run


def _row(index, question_type, reward, searches=1, response="answer"):
    return {
        "row_index": index,
        "question_type": question_type,
        "reward": reward,
        "search_count": searches,
        "assistant_responses": [response],
    }


def _summary(accuracy, two_search_count):
    return {
        "accuracy": accuracy,
        "question_types": {},
        "retrieval": {"two_search_count": two_search_count},
        "protocol": {"error_count": 0},
    }


def test_comparison_selects_fill_improvement_with_objective_retention(monkeypatch):
    monkeypatch.setattr(run, "CANDIDATE_STEPS", (20, 40, 60))
    baseline_rows = [
        _row(0, "single", 1.0),
        _row(1, "multiple", 1.0),
        _row(2, "bool", 1.0),
        _row(3, "fill", 0.0),
        _row(4, "short", 0.0),
    ]
    candidates = {
        20: [
            *baseline_rows[:3],
            _row(3, "fill", 0.5, searches=2, response="better"),
            baseline_rows[4],
        ],
        40: [
            _row(0, "single", 0.9),
            *baseline_rows[1:3],
            _row(3, "fill", 1.0, searches=2, response="best"),
            baseline_rows[4],
        ],
        60: baseline_rows,
    }
    summaries = {
        20: _summary(0.7, 1),
        40: _summary(0.78, 1),
        60: _summary(0.6, 0),
    }

    report = run.build_comparison(
        _summary(0.6, 0),
        baseline_rows,
        summaries,
        candidates,
    )

    assert report["candidates"]["20"]["grpo_gate"]["eligible"] is True
    assert report["candidates"]["40"]["grpo_gate"]["eligible"] is False
    assert report["decision"]["recommended_step_for_short_grpo"] == 20
