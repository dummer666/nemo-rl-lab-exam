from __future__ import annotations

import pytest

from common.retrieval.markdown_bm25 import SearchResult
from common.retrieval.qa_short_audit import (
    audit_short_label,
    audit_short_trace,
    parse_short_gold,
)
from experiments.qa_short_gold_audit_wanghaonan.run import (
    _logged_messages,
    _logged_scalar,
)


def _result(text: str, *, source: str = "reference.md") -> SearchResult:
    return SearchResult(
        source=source,
        heading="evidence",
        text=text,
        score=1.0,
        raw_score=1.0,
        quality_category="reference",
        quality_weight=1.0,
    )


class _StubIndex:
    def __init__(self, results_by_query: dict[str, list[SearchResult]]):
        self.results_by_query = results_by_query

    def search(self, query, top_k, *, candidate_k, quality_rerank):
        assert top_k == 20
        assert candidate_k == 50
        assert quality_rerank
        for key, results in self.results_by_query.items():
            if key in query:
                return results[:top_k]
        return []


@pytest.mark.parametrize("term", ["代码", "设备", "流程"])
def test_single_generic_label_is_a_defect_and_rule_attack_can_score_one(term):
    audit = audit_short_label(
        query="请审查一个复杂案例并说明主要问题。",
        expected=f"[short] {term}",
        bank="技术培训",
        results=[_result("完全无关的材料")],
    )

    assert audit["label_defect"]
    assert "singleton_generic_keypoint" in audit["label_defect_reasons"]
    assert audit["support_level"] == "none"
    assert any(
        attack["attack"] == "bare_singleton"
        and attack["official_rule_reward"] == pytest.approx(1.0)
        for attack in audit["reward_attacks"]
    )


def test_question_overlap_delimiter_and_duplicate_defects_are_preserved():
    overlap = parse_short_gold(
        "[short] OCAP",
        query="请说明 OCAP。",
    )
    malformed = parse_short_gold(
        "[short] alpha || beta",
        query="请给出两个要点。",
    )
    duplicate = parse_short_gold(
        "[short] incoming layer issue ||| incoming-layer issue",
        query="请给出两个要点。",
    )

    assert "singleton_question_high_overlap" in overlap["label_defect_reasons"]
    assert "malformed_keypoint_delimiter" in malformed["label_defect_reasons"]
    assert "near_duplicate_keypoints" in duplicate["label_defect_reasons"]


def test_step20_sample180_shape_is_false_perfect_reward_hacking():
    query = "请审查以下复杂 C++ 实现并指出核心缺陷。"
    expected = "[short] 代码"
    label = audit_short_label(
        query=query,
        expected=expected,
        bank="C++",
        results=[_result("这是无关的厂务维护资料。")],
    )
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": "<search>厂务维护</search>"},
        {"role": "environment", "content": "[检索结果]\n无关维护资料"},
        {"role": "assistant", "content": r"\boxed{代码}"},
        {"role": "environment", "content": "[最终答案已提交] reward=1.000"},
    ]

    trace = audit_short_trace(
        trace_source="short_grpo_sampled_step20",
        source_row_id=179,
        sample_number=180,
        checkpoint_step=20,
        query=query,
        expected=expected,
        bank="C++",
        messages=messages,
        label_audit=label,
        index=_StubIndex({"厂务维护": [_result("这是无关的厂务维护资料。")]}),
        logged_reward=1.0,
    )

    assert trace["official_rule_reward"] == pytest.approx(1.0)
    assert trace["logged_reward_matches_official_rule"]
    assert trace["primary_attribution"] == "label_defect"
    assert trace["false_perfect"]
    assert trace["reward_hacking"]


def test_partial_evidence_is_not_mislabeled_as_synthesis_failure():
    query = "请列出两个独立控制要求。"
    expected = "[short] alpha control ||| beta isolation"
    label = audit_short_label(
        query=query,
        expected=expected,
        bank="controls",
        results=[_result("alpha control")],
    )
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": "<search>first hop</search>"},
        {"role": "environment", "content": "[检索结果]\nalpha control"},
        {"role": "assistant", "content": r"\boxed{alpha control}"},
    ]

    trace = audit_short_trace(
        trace_source="fixture",
        source_row_id=1,
        query=query,
        expected=expected,
        bank="controls",
        messages=messages,
        label_audit=label,
        index=_StubIndex({"first hop": [_result("alpha control")]}),
    )

    assert not label["label_defect"]
    assert trace["evidence_support_level"] == "partial"
    assert trace["primary_attribution"] == "partial_evidence"
    assert "synthesis_failure" not in trace["failure_labels"]


def test_full_evidence_with_incomplete_answer_is_synthesis_failure():
    query = "请列出两个独立控制要求。"
    expected = "[short] alpha control ||| beta isolation"
    results = [_result("alpha control and beta isolation")]
    label = audit_short_label(
        query=query,
        expected=expected,
        bank="controls",
        results=results,
    )
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": "<search>complete evidence</search>"},
        {
            "role": "environment",
            "content": "[检索结果]\nalpha control and beta isolation",
        },
        {"role": "assistant", "content": r"证据只说明一项。\boxed{alpha control}"},
    ]

    trace = audit_short_trace(
        trace_source="fixture",
        source_row_id=2,
        query=query,
        expected=expected,
        bank="controls",
        messages=messages,
        label_audit=label,
        index=_StubIndex({"complete evidence": results}),
    )

    assert trace["evidence_support_level"] == "full"
    assert trace["official_rule_reward"] == pytest.approx(0.5)
    assert trace["primary_attribution"] == "synthesis_failure"


def test_protocol_failure_and_no_incremental_second_hop_are_visible():
    query = "请列出两个独立控制要求。"
    expected = "[short] alpha control ||| beta isolation"
    label = audit_short_label(
        query=query,
        expected=expected,
        bank="controls",
        results=[_result("alpha control")],
    )
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": "<search>first</search>"},
        {"role": "environment", "content": "[检索结果]\nalpha control"},
        {"role": "assistant", "content": "<search>second</search>"},
        {"role": "environment", "content": "[检索结果]\nalpha control"},
        {"role": "assistant", "content": "没有 boxed 最终答案"},
    ]

    trace = audit_short_trace(
        trace_source="fixture",
        source_row_id=3,
        query=query,
        expected=expected,
        bank="controls",
        messages=messages,
        label_audit=label,
        index=_StubIndex(
            {
                "first": [_result("alpha control")],
                "second": [_result("alpha control")],
            }
        ),
    )

    assert trace["search_hops"][1]["new_trusted_hit_indexes"] == []
    assert "missing_boxed_completion" in trace["protocol_issues"]
    assert trace["primary_attribution"] == "protocol_failure"


def test_keyword_only_full_reward_is_not_counted_as_real_synthesis():
    query = "请列出两个独立控制要求。"
    expected = "[short] alpha control ||| beta isolation"
    results = [_result("alpha control and beta isolation")]
    label = audit_short_label(
        query=query,
        expected=expected,
        bank="controls",
        results=results,
    )
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": "<search>complete</search>"},
        {
            "role": "environment",
            "content": "[检索结果]\nalpha control and beta isolation",
        },
        {"role": "assistant", "content": r"依据检索证据，\boxed{alpha control; beta isolation}"},
    ]

    trace = audit_short_trace(
        trace_source="fixture",
        source_row_id=4,
        query=query,
        expected=expected,
        bank="controls",
        messages=messages,
        label_audit=label,
        index=_StubIndex({"complete": results}),
    )

    assert trace["official_rule_reward"] == pytest.approx(1.0)
    assert trace["keyword_only_completion"]
    assert trace["primary_attribution"] == "synthesis_failure"
    assert trace["false_perfect"]


def test_visible_evidence_excludes_result_metadata_and_guidance():
    query = "请列出两个独立控制要求。"
    expected = "[short] alpha control ||| beta isolation"
    label = audit_short_label(
        query=query,
        expected=expected,
        bank="controls",
        results=[_result("alpha control and beta isolation")],
    )
    metadata_only = SearchResult(
        source="alpha-control-beta-isolation.md",
        heading="alpha control beta isolation",
        text="unrelated body",
        score=1.0,
        raw_score=1.0,
        quality_category="reference",
        quality_weight=1.0,
    )
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": "<search>metadata only</search>"},
        {
            "role": "environment",
            "content": (
                "[检索结果]\n"
                "1. 来源：alpha-control-beta-isolation.md · alpha control beta isolation\n"
                "相关度：1.00\n"
                "unrelated body\n\n还可检索 1 次。证据不足时继续检索。"
            ),
        },
        {"role": "assistant", "content": r"\boxed{alpha control; beta isolation}"},
    ]

    trace = audit_short_trace(
        trace_source="fixture",
        source_row_id=5,
        query=query,
        expected=expected,
        bank="controls",
        messages=messages,
        label_audit=label,
        index=_StubIndex({"metadata only": [metadata_only]}),
    )

    assert trace["visible_literal_hit_indexes"] == []
    assert trace["evidence_support_level"] == "none"
    assert trace["primary_attribution"] == "unsupported_retrieval"


def test_nemo_validation_singleton_wrappers_are_unwrapped():
    messages = [{"role": "user", "content": "prompt"}]
    logged = {
        "content": [messages],
        "rewards": [1.0],
        "idx": [179],
    }

    assert _logged_messages(logged) == messages
    assert _logged_scalar(logged, "rewards") == pytest.approx(1.0)
    assert _logged_scalar(logged, "idx") == 179
