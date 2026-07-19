from __future__ import annotations

from experiments.qa_fill_sft_eval_report_wanghaonan.run import build_report


def _row(index: int, question_type: str, reward: float) -> dict:
    return {
        "row_index": index,
        "question_type": question_type,
        "query": f"question {index}",
        "expected_answer": f"[{question_type}] answer",
        "reward": reward,
        "search_count": int(question_type in {"fill", "short"}),
        "assistant_responses": [f"answer {reward}"],
    }


def _summary(accuracy: float) -> dict:
    return {
        "accuracy": accuracy,
        "question_types": {},
        "retrieval": {},
        "protocol": {},
    }


def test_report_separates_objective_and_open_deltas():
    baseline = [
        _row(0, "single", 1.0),
        _row(1, "fill", 0.0),
        _row(2, "short", 0.0),
    ]
    candidate = [
        _row(0, "single", 1.0),
        _row(1, "fill", 0.5),
        _row(2, "short", 0.0),
    ]

    report = build_report(
        baseline,
        candidate,
        _summary(1 / 3),
        _summary(0.5),
    )

    assert report["deltas"]["objective_accuracy"] == 0.0
    assert report["deltas"]["open_accuracy"] == 0.25
    assert report["deltas"]["paired_by_type"]["fill"]["gain_count"] == 1
    assert report["changed_open_count"] == 1
