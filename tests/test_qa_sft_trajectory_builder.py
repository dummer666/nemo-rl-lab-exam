from __future__ import annotations

from common.retrieval.markdown_bm25 import SearchResult
from common.retrieval.qa_sft import build_search_messages
from experiments.qa_sft_trajectory_build_wanghaonan.run import (
    _evaluate_rewrite_candidate,
    _select_objective_rows,
    _trajectory_token_audit,
)


class _StubIndex:
    def __init__(self, results: list[SearchResult]):
        self.results = results

    def search(self, _query, top_k, *, candidate_k, quality_rerank):
        assert top_k == 4
        assert candidate_k == 50
        assert quality_rerank
        return self.results[:top_k]


class _StubTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        add_special_tokens,
        enable_thinking=False,
    ):
        assert [message["role"] for message in messages] == ["system", "user"]
        assert not tokenize
        assert add_generation_prompt
        assert not add_special_tokens
        assert not enable_thinking
        return "  <initial-agent-prompt>\n"

    def __call__(self, text, *, add_special_tokens, return_tensors):
        assert add_special_tokens is False
        assert return_tensors == "pt"
        return {"input_ids": [list(range(len(text)))]}


def _result(text: str) -> SearchResult:
    return SearchResult(
        source="answer.md",
        heading="answer",
        text=text,
        score=1.0,
        raw_score=1.0,
        quality_category="answer-bearing",
        quality_weight=1.15,
    )


def test_token_audit_converts_observations_to_runtime_raw_chunks():
    semantic_messages = build_search_messages(
        query="题目",
        expected="[fill] 答案",
        first_query="查询",
        first_observation="[检索结果]\n答案",
    )
    trajectory = {
        "messages": semantic_messages,
        "_audit": {},
    }

    audited, reason = _trajectory_token_audit(
        trajectory,
        _StubTokenizer(),
    )

    assert reason is None
    assert audited is not None
    assert [message["role"] for message in audited["messages"]] == [
        "user",
        "assistant",
        "environment",
        "assistant",
    ]
    assert audited["messages"][0]["content"] == "<initial-agent-prompt>"
    assert audited["_audit"]["runtime_raw_chunk_alignment"]


def test_rewrite_candidate_requires_incremental_full_visible_evidence():
    record = {
        "query": "题目：设备 chamber pressure target",
        "bank": "",
        "first_search_query": "设备 chamber",
        "first_retrieval_output": "[检索结果]\nalpha chamber",
        "first_observation_hits": [0],
        "keypoints": [["alpha"], ["beta"]],
    }

    accepted, reason = _evaluate_rewrite_candidate(
        record,
        "chamber pressure target",
        _StubIndex([_result("beta")]),
    )

    assert reason == "accepted"
    assert accepted is not None
    assert accepted["new_hits"] == [1]
    assert accepted["cumulative_hits"] == [0, 1]


def test_rewrite_candidate_rejects_partial_second_observation():
    record = {
        "query": "题目：设备 chamber pressure target",
        "bank": "",
        "first_search_query": "设备 chamber",
        "first_retrieval_output": "[检索结果]\nalpha chamber",
        "first_observation_hits": [0],
        "keypoints": [["alpha"], ["beta"], ["gamma"]],
    }

    accepted, reason = _evaluate_rewrite_candidate(
        record,
        "chamber pressure target",
        _StubIndex([_result("beta")]),
    )

    assert accepted is None
    assert reason == "incomplete_cumulative_evidence"


def test_objective_retention_sampling_is_balanced_and_disjoint():
    rows = []
    row_id = 0
    for question_type in ("single", "multiple", "bool"):
        for index in range(40):
            rows.append(
                {
                    "query": f"{question_type}-{index}",
                    "expected_answer": f"[{question_type}] A",
                    "_clean": {"row_id": row_id},
                }
            )
            row_id += 1

    train, validation = _select_objective_rows(
        rows,
        train_per_type=6,
        validation_per_type=2,
    )

    assert len(train) == 18
    assert len(validation) == 6
    assert {
        row["_clean"]["row_id"]
        for row in train
    }.isdisjoint(row["_clean"]["row_id"] for row in validation)
