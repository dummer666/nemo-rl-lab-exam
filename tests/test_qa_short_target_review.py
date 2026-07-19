from __future__ import annotations

import pytest

from experiments.qa_short_target_rebuild_review_wanghaonan.run import (
    _build_report,
)


def _target() -> dict:
    points = [
        {
            "index": 1,
            "statement": "第一项是完整事实句。",
            "quote": "第一条连续原文证据",
        },
        {
            "index": 2,
            "statement": "第二项是另一完整事实句。",
            "quote": "第二条连续原文证据",
        },
    ]
    final = (
        "依据检索证据，答案要点为：1. 第一项是完整事实句。；"
        "2. 第二项是另一完整事实句。\n"
        r"\boxed{1. 第一项是完整事实句。；2. 第二项是另一完整事实句。}"
    )
    return {
        "source_row_id": 7,
        "question_fingerprint": "fp",
        "split": "train",
        "query": "请给出两个事实。",
        "legacy_expected_answer": "[short] 旧标签",
        "expected_answer": "[short] 第一项是完整事实句。 ||| 第二项是另一完整事实句。",
        "answer_points": points,
        "search_turns": 1,
        "search_hops": [
            {
                "observation": (
                    "[检索结果]\n第一条连续原文证据\n第二条连续原文证据"
                )
            }
        ],
        "messages": [{"role": "assistant", "content": final}],
        "_audit": {"token_length": 100},
    }


def test_review_report_requires_visible_quotes_and_complete_final_answer():
    target = _target()
    report = _build_report(
        {
            "official_validation_overlap_count": 0,
            "machine_verified_route_targets": 1,
        },
        [target],
        [
            {
                "source_row_id": 7,
                "candidate_index": 1,
                "deterministic_decision": "accepted",
                "verifier_accept": True,
            },
            {
                "source_row_id": 8,
                "candidate_index": 2,
                "query": "请给出另两个事实。",
                "deterministic_decision": "accepted",
                "points": [
                    {"index": 1, "statement": "缺少支持的事实。"},
                    {"index": 2, "statement": "相关但不完整的事实。"},
                ],
                "verifier_raw": (
                    '{"decision":"reject","complete":false,'
                    '"point_checks":['
                    '{"index":1,"supported":false,"relevant":true},'
                    '{"index":2,"supported":true,"relevant":true}],'
                    '"reason":"第一点无证据且整体不完整"}'
                ),
                "verifier_accept": False,
            }
        ],
        [{"source_row_id": 7, "accepted": True}],
        [{"source_row_id": 8, "stage": "target", "reason": "rejected"}],
    )

    assert report["accepted_target_count"] == 1
    accepted = report["accepted_targets"][0]
    assert accepted["human_review_checklist"]["decision"] == "pending_human_review"
    assert accepted["teacher_and_verifier_attempts"][0]["verifier_accept"]
    assert report["generation_decision_counts"] == {"accepted": 2}
    assert report["independent_verifier_counts"] == {
        "accepted": 1,
        "rejected": 1,
    }
    assert report["independent_verifier_failure_counts"] == {
        "incomplete": 1,
        "unsupported_point": 1,
    }
    rejection = report["independent_verifier_rejections"][0]
    assert rejection["source_row_id"] == 8
    assert rejection["failure_categories"] == [
        "incomplete",
        "unsupported_point",
    ]

    broken = _target()
    broken["search_hops"][0]["observation"] = "[检索结果]\n无关文本"
    with pytest.raises(ValueError, match="quote is not visible"):
        _build_report(
            {
                "official_validation_overlap_count": 0,
                "machine_verified_route_targets": 1,
            },
            [broken],
            [],
            [],
            [],
        )
