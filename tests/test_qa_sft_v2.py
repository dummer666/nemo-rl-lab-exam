from __future__ import annotations

import pytest

from common.retrieval.qa_sft import (
    grounded_points_response,
    observation_with_guidance,
)
from common.retrieval.qa_sft_v2 import (
    assert_question_split_isolation,
    grounded_answer_term_leak,
    objective_replay_fraction,
    open_answer_leak_points,
    question_keypoint_leak,
    rendered_observation_from_records,
    select_balanced_objective_replay,
    select_objective_validation,
    short_target_issues,
    source_question_answer_term_leak,
)
from common.retrieval.qa_target_rebuild import question_fingerprint


def _short_target(*, category: str = "reference") -> dict:
    quote = "离子注入系统包含离子源，离子源用于产生离子"
    statement = "系统包含离子源，用于产生离子。"
    second_quote = "离子注入系统包含分析磁场，分析磁场用于筛选离子"
    second_statement = "系统包含分析磁场，用于筛选离子。"
    query = "离子注入系统由哪些部分组成？"
    ranked_results = [
        {
            "rank": 1,
            "source": "reference.md",
            "heading": "系统组成",
            "quality_category": category,
            "raw_score": 2.0,
            "text": f"{quote}；{second_quote}",
        }
    ]
    observation = rendered_observation_from_records(ranked_results)
    return {
        "source_row_id": 7,
        "question_fingerprint": question_fingerprint(query),
        "question_type": "short",
        "query": query,
        "expected_answer": (
            f"[short] {statement} ||| {second_statement}"
        ),
        "split": "train",
        "machine_verified": True,
        "search_turns": 1,
        "answer_points": [
            {
                "index": 1,
                "statement": statement,
                "quote": quote,
                "visible_supports": [
                    {
                        "hop": 1,
                        "rank": 1,
                        "quality_category": category,
                    }
                ],
            },
            {
                "index": 2,
                "statement": second_statement,
                "quote": second_quote,
                "visible_supports": [
                    {
                        "hop": 1,
                        "rank": 1,
                        "quality_category": category,
                    }
                ],
            },
        ],
        "search_hops": [
            {
                "hop": 1,
                "model_search_query": "离子注入 系统组成",
                "observation": observation,
                "answer_point_hit_indexes": [0, 1],
                "new_answer_point_hit_indexes": [0, 1],
                "top_k_results": ranked_results,
            }
        ],
        "messages": [
            {"role": "user", "content": "rendered prompt"},
            {"role": "assistant", "content": "<search>离子注入 系统组成</search>"},
            {
                "role": "environment",
                "content": observation_with_guidance(
                    observation,
                    searches_remaining=1,
                ),
            },
            {
                "role": "assistant",
                "content": grounded_points_response(
                    [statement, second_statement]
                ),
            },
        ],
    }


def _objective_records(split: str, count: int = 8) -> list[dict]:
    return [
        {
            "row_id": f"{question_type}-{index}",
            "question_fingerprint": f"{question_type}-{split}-{index}",
            "question_type": question_type,
            "split": split,
        }
        for question_type in ("single", "multiple", "bool")
        for index in range(count)
    ]


def test_short_target_requires_trusted_visible_same_result_binding():
    record = _short_target()
    assert short_target_issues(record) == []

    bad_category = _short_target(category="question-only")
    assert "unbound_visible_support" in short_target_issues(bad_category)


def test_short_target_rejects_spoofed_fingerprint_or_observation():
    record = _short_target()
    record["question_fingerprint"] = "spoofed"
    assert "question_fingerprint_mismatch" in short_target_issues(record)

    record["search_hops"][0]["observation"] += "\n额外泄漏答案"
    record["messages"][2]["content"] = observation_with_guidance(
        record["search_hops"][0]["observation"],
        searches_remaining=1,
    )
    assert "observation_ranked_results_mismatch" in short_target_issues(record)


def test_short_target_rejects_hidden_ranked_evidence():
    hidden = _short_target()
    hidden["search_hops"][0]["top_k_results"][0]["text"] = "截断后不可见"
    assert "unbound_visible_support" in short_target_issues(hidden)


def test_short_target_requires_complete_runtime_answer():
    record = _short_target()
    record["messages"][-1]["content"] = r"\boxed{第一项}"
    assert "incomplete_final_answer" in short_target_issues(record)
    record = _short_target()
    record["messages"][-1]["content"] += "\n额外虚构要点"
    assert "incomplete_final_answer" in short_target_issues(record)


def test_short_target_requires_exact_initial_runtime_prompt():
    record = _short_target()
    assert short_target_issues(
        record,
        expected_initial_prompt="rendered prompt",
    ) == []
    record["messages"][0]["content"] = "corrupted prompt with leaked answer"
    assert "initial_runtime_prompt_mismatch" in short_target_issues(
        record,
        expected_initial_prompt="rendered prompt",
    )


def test_short_target_rejects_leaky_or_mismatched_runtime_trace():
    leaky = _short_target()
    leaky["search_hops"][0]["model_search_query"] = "离子源 分析磁场"
    leaky["messages"][1]["content"] = "<search>离子源 分析磁场</search>"
    assert "search_query_answer_leak" in short_target_issues(leaky)

    mismatched = _short_target()
    mismatched["messages"][1]["content"] = "<search>其他检索词</search>"
    mismatched["messages"][2]["content"] = "[检索结果]\n无关文本"
    issues = short_target_issues(mismatched)
    assert "runtime_search_query_mismatch" in issues
    assert "runtime_observation_mismatch" in issues

    extra_payload = _short_target()
    extra_payload["messages"][1]["content"] += "答案是离子源和分析磁场"
    extra_payload["messages"][2]["content"] += "\n额外泄漏答案"
    issues = short_target_issues(extra_payload)
    assert "runtime_search_query_mismatch" in issues
    assert "runtime_observation_mismatch" in issues


def test_short_target_rejects_nonincremental_two_hop_trace():
    record = _short_target()
    second = {
        **record["search_hops"][0],
        "hop": 2,
        "model_search_query": "离子注入 其他资料",
        "new_answer_point_hit_indexes": [],
    }
    record["search_turns"] = 2
    record["search_hops"].append(second)
    record["messages"][-1:-1] = [
        {"role": "assistant", "content": "<search>离子注入 其他资料</search>"},
        {
            "role": "environment",
            "content": observation_with_guidance(
                second["observation"],
                searches_remaining=0,
            ),
        },
    ]
    issues = short_target_issues(record)
    assert "first_hop_already_complete" in issues
    assert "nonincremental_second_hop" in issues


def test_query_leak_only_flags_answers_absent_from_question():
    keypoints = [["离子源"], ["分析磁场"]]
    assert question_keypoint_leak(
        "离子源 分析磁场",
        "离子注入系统由哪些部分组成？",
        keypoints,
    )
    assert not question_keypoint_leak(
        "离子源 工作原理",
        "离子源的工作原理是什么？",
        keypoints,
    )
    leak_points = open_answer_leak_points(
        "[fill] chamber pressure target ||| response time window"
    )
    assert grounded_answer_term_leak(
        "pressure target",
        "设备参数是什么？",
        leak_points,
    )
    assert source_question_answer_term_leak(
        "设备的 beta gamma 参数是什么？",
        open_answer_leak_points("[fill] alpha beta gamma"),
    )
    assert not source_question_answer_term_leak(
        "离子源的作用是什么？",
        open_answer_leak_points("[fill] 产生离子"),
    )


def test_balanced_objective_replay_stays_within_gate():
    selected = select_balanced_objective_replay(
        _objective_records("train"),
        open_train_count=30,
    )
    records = [
        *selected,
        *[
            {
                "question_type": "short",
                "question_fingerprint": f"open-{index}",
                "split": "train",
            }
            for index in range(30)
        ],
    ]
    assert len(selected) % 3 == 0
    assert 0.25 <= objective_replay_fraction(records) <= 0.35
    assert {
        record["question_type"]
        for record in selected
    } == {"single", "multiple", "bool"}


def test_objective_validation_is_balanced_and_deterministic():
    records = _objective_records("validation")
    first = select_objective_validation(records, per_type=2)
    second = select_objective_validation(records, per_type=2)
    assert first == second
    assert len(first) == 6


def test_question_split_isolation_rejects_cross_split_overlap():
    records = [
        {"question_fingerprint": "a", "split": "train"},
        {"question_fingerprint": "b", "split": "validation"},
    ]
    assert_question_split_isolation(records)
    with pytest.raises(ValueError, match="overlaps"):
        assert_question_split_isolation(
            [*records, {"question_fingerprint": "a", "split": "rl_holdout"}]
        )
    with pytest.raises(ValueError, match="duplicate"):
        assert_question_split_isolation(
            [*records, {"question_fingerprint": "a", "split": "train"}]
        )
