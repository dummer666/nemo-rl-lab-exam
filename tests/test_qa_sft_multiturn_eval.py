from experiments.qa_sft_multiturn_eval_wanghaonan.run import (
    _summarize,
    _truncate_at_search_close,
)


def test_truncate_at_first_search_close_matches_runtime_stop():
    response = "<search>first query</search>ignored <search>second</search>"

    assert _truncate_at_search_close(response) == "<search>first query</search>"


def test_summary_reports_open_perfect_retrieval_and_protocol_metrics():
    rows = [
        {
            "question_type": "fill",
            "reward": 1.0,
            "search_count": 1,
            "protocol_error": False,
            "terminated": True,
        },
        {
            "question_type": "short",
            "reward": 0.5,
            "search_count": 2,
            "protocol_error": False,
            "terminated": True,
        },
        {
            "question_type": "single",
            "reward": -0.5,
            "search_count": 0,
            "protocol_error": True,
            "terminated": True,
        },
    ]

    summary = _summarize(rows)

    assert summary["accuracy"] == 0.5
    assert summary["open_questions"]["perfect_count"] == 1
    assert summary["retrieval"]["retrieval_count"] == 2
    assert summary["retrieval"]["one_search_count"] == 1
    assert summary["retrieval"]["two_search_count"] == 1
    assert summary["retrieval"]["open_retrieval_rate"] == 1.0
    assert summary["protocol"]["error_count"] == 1
