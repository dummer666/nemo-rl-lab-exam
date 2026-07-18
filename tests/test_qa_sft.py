from __future__ import annotations

from common.retrieval.qa_sft import (
    assign_open_splits,
    build_objective_messages,
    build_search_messages,
    canonical_answer,
    observation_with_guidance,
    parse_query_candidate,
    query_rejection_reason,
    validate_messages,
)


def test_canonical_answer_formats_all_qa_types():
    assert canonical_answer("[single] B") == ("single", "B")
    assert canonical_answer("[multiple] A,C") == ("multiple", "A,C")
    assert canonical_answer("[fill] SQL server/SQL ||| clean room") == (
        "fill",
        "SQL server; clean room",
    )
    assert canonical_answer("[short] ion source/离子源 ||| Faraday") == (
        "short",
        "ion source; Faraday",
    )


def test_search_messages_match_runtime_role_order_and_string_observations():
    messages = build_search_messages(
        query="题目",
        expected="[fill] 答案",
        first_query="首次查询",
        first_observation="[检索结果]\n证据一",
        second_query="二次查询",
        second_observation="[检索结果]\n证据二",
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "environment",
        "assistant",
        "environment",
        "assistant",
    ]
    assert all(isinstance(message["content"], str) for message in messages)
    assert messages[2]["content"] == "<search>首次查询</search>"
    assert messages[-1]["content"].endswith(r"\boxed{答案}")
    assert validate_messages(messages) == []


def test_objective_messages_do_not_teach_unnecessary_search():
    messages = build_objective_messages(
        query="选择正确答案",
        expected="[bool] A",
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert "<search>" not in messages[-1]["content"]
    assert validate_messages(messages) == []


def test_query_candidate_parser_accepts_common_teacher_formats():
    assert parse_query_candidate("<search>设备 报警 原因</search>") == "设备 报警 原因"
    assert parse_query_candidate('{"query": "设备 处理步骤"}') == "设备 处理步骤"
    assert parse_query_candidate("<think>分析</think>\n查询：设备 参数") == "设备 参数"


def test_query_quality_gate_rejects_duplicates_leaks_and_ungrounded_terms():
    keypoints = [["secretanswer"], ["target"]]
    context = "题目设备报警，首轮结果提到 chamber pressure"

    assert (
        query_rejection_reason(
            "设备报警",
            first_query="设备报警",
            visible_context=context,
            keypoints=keypoints,
        )
        == "duplicate_query"
    )
    assert (
        query_rejection_reason(
            "设备 secretanswer",
            first_query="设备报警",
            visible_context=context,
            keypoints=keypoints,
        )
        == "answer_leak"
    )
    assert (
        query_rejection_reason(
            "查询详细信息",
            first_query="设备报警",
            visible_context=context,
            keypoints=keypoints,
        )
        == "ungrounded_query"
    )
    assert (
        query_rejection_reason(
            "设备 totallyhallucinated",
            first_query="设备报警",
            visible_context=context,
            keypoints=keypoints,
        )
        == "unsupported_query_terms"
    )
    assert (
        query_rejection_reason(
            "chamber pressure 参数",
            first_query="设备报警",
            visible_context=context,
            keypoints=keypoints,
        )
        is None
    )


def test_observation_guidance_uses_real_newlines():
    observation = observation_with_guidance(
        "[检索结果]\n证据",
        searches_remaining=0,
    )

    assert "\n\n检索次数已用完" in observation
    assert r"\n\n检索次数已用完" not in observation


def test_open_split_is_deterministic_and_disjoint():
    records = [
        {
            "row_id": index,
            "query": f"q-{index}",
            "question_type": "fill" if index < 20 else "short",
            "search_turns": 1 if index % 2 else 2,
        }
        for index in range(40)
    ]

    first = assign_open_splits(records, seed=7)
    second = assign_open_splits(records, seed=7)

    assert first == second
    split_ids = {
        split: {
            record["row_id"]
            for record in first
            if record["split"] == split
        }
        for split in ("train", "validation", "rl_holdout")
    }
    assert split_ids["train"]
    assert split_ids["validation"]
    assert split_ids["rl_holdout"]
    assert split_ids["train"].isdisjoint(split_ids["validation"])
    assert split_ids["train"].isdisjoint(split_ids["rl_holdout"])
    assert split_ids["validation"].isdisjoint(split_ids["rl_holdout"])
