from experiments.qa_sft_eval_report_wanghaonan.run import _build_report
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


def test_report_compares_paired_open_and_protocol_results():
    summaries = {
        step: {
            "accuracy": 0.5,
            "mean_reward": 0.5,
            "perfect_count": 1,
            "perfect_rate": 0.5,
            "open_questions": {"perfect_count": 1},
            "retrieval": {"retrieval_rate": 0.5},
            "protocol": {"error_count": int(step == 25)},
            "question_types": {},
            "evaluation_seconds": 10.0,
        }
        for step in (25, 50)
    }
    base = {
        "query": "question",
        "expected_answer": "[fill] answer",
        "search_count": 1,
        "search_queries": ["query"],
        "evidence_coverage": 1.0,
        "termination_reason": "final_answer",
        "assistant_responses": [r"\boxed{answer}"],
    }
    trajectories = {
        25: [
            {
                **base,
                "row_index": 0,
                "question_type": "fill",
                "reward": 1.0,
                "protocol_error": False,
            },
            {
                **base,
                "row_index": 1,
                "question_type": "single",
                "reward": -0.5,
                "protocol_error": True,
            },
        ],
        50: [
            {
                **base,
                "row_index": 0,
                "question_type": "fill",
                "reward": 1.0,
                "protocol_error": False,
            },
            {
                **base,
                "row_index": 1,
                "question_type": "single",
                "reward": 0.0,
                "protocol_error": False,
            },
        ],
    }

    report = _build_report(summaries, trajectories)

    assert report["paired_changes"]["gain_count"] == 1
    assert report["open_perfect"]["shared_ids"] == [0]
    assert report["protocol_errors"]["step25_ids"] == [1]
    assert report["protocol_errors"]["step50_ids"] == []
