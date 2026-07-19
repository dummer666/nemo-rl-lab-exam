from experiments.qa_objective_sft_gate_eval_wanghaonan import run


def _summary(
    *,
    objective_perfect: int,
    overall_accuracy: float,
    open_accuracy: float,
    protocol_errors: int = 0,
):
    objective_count = 275
    return {
        "accuracy": overall_accuracy,
        "question_types": {
            "single": {
                "count": objective_count,
                "accuracy": objective_perfect / objective_count,
                "perfect_count": objective_perfect,
            }
        },
        "open_questions": {
            "count": 38,
            "accuracy": open_accuracy,
            "perfect_count": 3,
        },
        "protocol": {"error_count": protocol_errors},
    }


def test_gate_requires_both_internal_and_official_improvement():
    summaries = {
        0: {
            "internal": _summary(
                objective_perfect=200,
                overall_accuracy=200 / 275,
                open_accuracy=0.0,
            ),
            "official": _summary(
                objective_perfect=210,
                overall_accuracy=0.68,
                open_accuracy=0.10,
            ),
        },
        20: {
            "internal": _summary(
                objective_perfect=203,
                overall_accuracy=203 / 275,
                open_accuracy=0.0,
            ),
            "official": _summary(
                objective_perfect=211,
                overall_accuracy=0.69,
                open_accuracy=0.10,
            ),
        },
    }

    report = run.build_gate_report(summaries)

    assert report["decision"] == {
        "promote_to_full_sft": True,
        "selected_step": 20,
        "reason": "candidate passed every predeclared behavior gate",
    }


def test_gate_rejects_internal_only_gain():
    summaries = {
        0: {
            "internal": _summary(
                objective_perfect=200,
                overall_accuracy=200 / 275,
                open_accuracy=0.0,
            ),
            "official": _summary(
                objective_perfect=210,
                overall_accuracy=0.68,
                open_accuracy=0.10,
            ),
        },
        20: {
            "internal": _summary(
                objective_perfect=204,
                overall_accuracy=204 / 275,
                open_accuracy=0.0,
            ),
            "official": _summary(
                objective_perfect=210,
                overall_accuracy=0.68,
                open_accuracy=0.10,
            ),
        },
    }

    report = run.build_gate_report(summaries)

    assert report["decision"]["promote_to_full_sft"] is False
    assert (
        report["candidates"]["20"]["criteria"][
            "official_objective_gain_at_least_1"
        ]
        is False
    )
