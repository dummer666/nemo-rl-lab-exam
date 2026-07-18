from __future__ import annotations

import pytest

from common.retrieval.markdown_bm25 import SearchResult
from common.retrieval.qa_target_rebuild import (
    align_literal_quote,
    assign_group_splits,
    bind_visible_evidence,
    build_evidence_spans,
    evidence_quote_hits,
    extract_json_object,
    materialize_span_references,
    non_text_task_reason,
    question_query_candidates,
    rebuilt_expected_answer,
    trusted_visible_quote_hits,
    validate_generated_target,
    verifier_accepts,
)


def _evidence(text: str) -> dict:
    return {
        "source": "trusted.md",
        "heading": "技术说明",
        "quality_category": "reference",
        "text": text,
    }


def test_generated_target_requires_exact_grounded_quotes():
    evidence = {
        "E01": _evidence("CCP 使用平行板电容耦合产生等离子体，适合高密度工艺。"),
        "E02": _evidence("ICP 使用感应线圈耦合，并可独立控制离子能量。"),
    }
    payload = {
        "decision": "answerable",
        "answer_points": [
            {
                "statement": "CCP 通过平行板电容耦合产生等离子体。",
                "evidence_id": "E01",
                "quote": "CCP 使用平行板电容耦合产生等离子体",
            },
            {
                "statement": "ICP 采用感应线圈耦合并可独立控制离子能量。",
                "evidence_id": "E02",
                "quote": "ICP 使用感应线圈耦合，并可独立控制离子能量",
            },
        ],
    }

    points, reason = validate_generated_target(
        payload,
        question="Dry Etch chamber 的两种类型及特点是什么？",
        evidence_by_id=evidence,
    )

    assert reason == "accepted"
    assert points is not None
    assert rebuilt_expected_answer(points).startswith("[short] CCP")


def test_generated_target_rejects_fabricated_quote_and_entity():
    payload = {
        "decision": "answerable",
        "answer_points": [
            {
                "statement": "CCP 的功率固定为 900W。",
                "evidence_id": "E01",
                "quote": "CCP 的功率固定为 900W",
            },
            {
                "statement": "ICP 使用线圈产生等离子体。",
                "evidence_id": "E01",
                "quote": "ICP 使用线圈产生等离子体",
            },
        ],
    }

    points, reason = validate_generated_target(
        payload,
        question="两类 chamber 有何特点？",
        evidence_by_id={"E01": _evidence("CCP 与 ICP 是两类 chamber。")},
    )

    assert points is None
    assert reason == "point_1:quote_not_exact"


def test_generated_target_requires_literal_quote_and_safe_target_syntax():
    evidence = {
        "E01": _evidence("ICP 使用感应线圈耦合，并可独立控制离子能量。"),
        "E02": _evidence("CCP 使用平行板电容耦合，并适用于刻蚀工艺。"),
    }
    punctuation_changed = {
        "decision": "answerable",
        "answer_points": [
            {
                "statement": "ICP 使用感应线圈耦合并可独立控制离子能量。",
                "evidence_id": "E01",
                "quote": "ICP 使用感应线圈耦合并可独立控制离子能量",
            },
            {
                "statement": "CCP 使用平行板电容耦合并适用于刻蚀工艺。",
                "evidence_id": "E02",
                "quote": "CCP 使用平行板电容耦合，并适用于刻蚀工艺",
            },
        ],
    }
    points, reason = validate_generated_target(
        punctuation_changed,
        question="两种 chamber 的特点是什么？",
        evidence_by_id=evidence,
    )
    assert reason == "accepted"
    assert points is not None
    assert points[0]["quote"] == "ICP 使用感应线圈耦合，并可独立控制离子能量"
    assert points[0]["quote_alignment"] == "punctuation_repaired"

    delimiter_injected = {
        **punctuation_changed,
        "answer_points": [
            {
                "statement": "ICP 使用感应线圈耦合 ||| 并可控制离子能量。",
                "evidence_id": "E01",
                "quote": "ICP 使用感应线圈耦合，并可独立控制离子能量",
            },
            punctuation_changed["answer_points"][1],
        ],
    }
    points, reason = validate_generated_target(
        delimiter_injected,
        question="两种 chamber 的特点是什么？",
        evidence_by_id=evidence,
    )
    assert points is None
    assert reason == "point_1:reserved_statement_syntax"


def test_quote_alignment_requires_one_unambiguous_contiguous_source_span():
    source = "第一处：连续，原文证据。第二处：连续，原文证据。"
    assert align_literal_quote("唯一证据：连续，原文。", "连续原文") == (
        "连续，原文"
    )
    assert align_literal_quote(source, "连续原文证据") is None
    assert (
        align_literal_quote(
            "前半段连续原文与后半段证据均完整保留。",
            "前半段连续原文...后半段证据",
        )
        is None
    )


def test_question_query_candidates_only_use_literal_question_clauses():
    question = "题目：请比较 CCP 与 ICP 的耦合方式；说明各自适用场景。"
    candidates = question_query_candidates(question)
    assert candidates == [
        "请比较 CCP 与 ICP 的耦合方式；说明各自适用场景。",
        "比较 CCP 与 ICP 的耦合方式",
        "各自适用场景",
    ]
    assert all("答案" not in candidate for candidate in candidates)


def test_span_selection_materializes_exact_source_across_ocr_metadata():
    source = (
        "建立评审体系并保持维 *[Image OCR] There is no text. [End OCR]* "
        "护和持续改进。下一项职责是批准变更。"
    )
    evidence = {"E09": _evidence(source)}
    spans = build_evidence_spans(evidence)
    matching = [
        span
        for span in spans
        if "建立评审体系" in span["quote"] and "持续改进" in span["quote"]
    ]
    assert len(matching) == 1
    assert matching[0]["quote"] in source

    payload, reason = materialize_span_references(
        {
            "decision": "answerable",
            "answer_points": [
                {
                    "statement": "委员会建立评审体系并负责持续改进。",
                    "span_id": matching[0]["span_id"].lower().replace("s0", "s"),
                }
            ],
        },
        {span["span_id"]: span for span in spans},
    )
    assert reason is None
    assert payload is not None
    point = payload["answer_points"][0]
    assert point["evidence_id"] == "E09"
    assert point["quote"] == matching[0]["quote"]


def test_evidence_spans_remove_only_synthetic_edge_ellipsis():
    evidence = {
        "E01": _evidence("…这是完整且连续的有效证据句子。"),
        "E02": _evidence("前半段证据…后半段证据不能拼接。"),
    }
    spans = build_evidence_spans(evidence)
    quotes = [span["quote"] for span in spans]
    assert "这是完整且连续的有效证据句子。" in quotes
    assert all("…" not in quote for quote in quotes)


def test_span_selection_rejects_unknown_teacher_reference():
    payload, reason = materialize_span_references(
        {
            "decision": "answerable",
            "answer_points": [
                {"statement": "完整事实句。", "span_id": "S999"}
            ],
        },
        {},
    )
    assert payload is None
    assert reason == "point_1:unknown_span"


def test_verifier_and_visible_quote_gates_are_strict():
    verifier = {
        "decision": "accept",
        "complete": True,
        "point_checks": [
            {"index": 1, "supported": True, "relevant": True},
            {"index": 2, "supported": True, "relevant": True},
        ],
    }
    points = [
        {"index": 1, "quote": "第一条连续证据文本"},
        {"index": 2, "quote": "第二条连续证据文本"},
    ]

    assert verifier_accepts(verifier, 2)
    assert not verifier_accepts(
        {**verifier, "point_checks": verifier["point_checks"][:1]},
        2,
    )
    assert evidence_quote_hits(
        "[检索结果]\n第一条连续证据文本；其余无关。",
        points,
    ) == {0}


def test_quote_hits_require_same_visible_trusted_result():
    points = [{"index": 1, "quote": "第一条连续证据文本"}]
    results = [
        SearchResult(
            source="questions.md",
            heading="考试题",
            text="第一条连续证据文本",
            score=2.0,
            quality_category="question-only",
        ),
        SearchResult(
            source="reference.md",
            heading="参考资料",
            text="第一条连续证据文本",
            score=1.0,
            quality_category="reference",
        ),
    ]

    assert trusted_visible_quote_hits(
        results,
        ["第一条连续证据文本", ""],
        points,
    ) == set()
    assert trusted_visible_quote_hits(
        results,
        ["第一条连续证据文本", "第一条连续证据文本"],
        points,
    ) == {0}


def test_json_parser_and_non_text_filters_do_not_fallback_to_dirty_targets():
    assert extract_json_object('说明\n```json\n{"decision":"reject"}\n```') == {
        "decision": "reject"
    }
    assert extract_json_object("无法解析") is None
    assert non_text_task_reason("请完成系统配置并截图上传。") == "screenshot"
    assert non_text_task_reason("请说明系统配置的两个关键约束。") is None


def test_group_split_requires_unique_questions_and_is_disjoint():
    records = [
        {
            "source_row_id": index,
            "question_fingerprint": f"fp-{index}",
            "search_turns": 1 if index < 10 else 2,
        }
        for index in range(20)
    ]
    first = assign_group_splits(records, seed=9)
    second = assign_group_splits(records, seed=9)

    assert first == second
    by_split = {
        split: {
            row["question_fingerprint"]
            for row in first
            if row["split"] == split
        }
        for split in ("train", "validation", "rl_holdout")
    }
    assert all(by_split.values())
    assert by_split["train"].isdisjoint(by_split["validation"])
    assert by_split["train"].isdisjoint(by_split["rl_holdout"])
    assert by_split["validation"].isdisjoint(by_split["rl_holdout"])

    duplicate = [*records, {**records[0], "source_row_id": 99}]
    with pytest.raises(ValueError, match="deduplicated"):
        assign_group_splits(duplicate)


def test_points_bind_to_the_ranked_source_actually_visible_to_agent():
    points = [
        {
            "index": 1,
            "statement": "第一项完整事实。",
            "quote": "第一条连续原文证据",
            "source": "candidate-copy.md",
        },
        {
            "index": 2,
            "statement": "第二项完整事实。",
            "quote": "第二条连续原文证据",
            "source": "candidate-copy.md",
        },
    ]
    hops = [
        {
            "hop": 1,
            "observation": (
                "[检索结果]\n第一条连续原文证据\n第二条连续原文证据"
            ),
            "top_k_results": [
                {
                    "rank": 1,
                    "source": "visible.md",
                    "heading": "答案段",
                    "quality_category": "reference",
                    "text": "第一条连续原文证据；第二条连续原文证据",
                }
            ],
        }
    ]

    bound = bind_visible_evidence(points, hops)

    assert bound[0]["source"] == "candidate-copy.md"
    assert bound[0]["visible_supports"] == [
        {
            "hop": 1,
            "rank": 1,
            "source": "visible.md",
            "heading": "答案段",
            "quality_category": "reference",
        }
    ]

    with pytest.raises(ValueError, match="no exact visible source binding"):
        bind_visible_evidence(
            [{**points[0], "quote": "未显示的原文证据"}],
            hops,
        )

    untrusted_hops = [
        {
            **hops[0],
            "top_k_results": [
                {
                    **hops[0]["top_k_results"][0],
                    "quality_category": "question-only",
                }
            ],
        }
    ]
    with pytest.raises(ValueError, match="no exact visible source binding"):
        bind_visible_evidence(points, untrusted_hops)
