from __future__ import annotations

import pytest

from common.retrieval.markdown_bm25 import (
    MarkdownBM25Index,
    SearchResult,
    build_retrieval_query,
    classify_document_quality,
    extract_search_query,
    format_search_results,
    question_context,
)
from common.retrieval.qa_agent import QARetrievalRunner
from common.retrieval.semantic_reranker import (
    reciprocal_rank_fusion,
    rerank_by_semantic,
)
from common.rewards.qa_reward import FORMAT_PENALTY, qa_rule_reward_fn


def _build_index(tmp_path):
    (tmp_path / "implant.md").write_text(
        "# 离子注入系统\n离子注入机由离子源、分析磁场、加速器、聚焦扫描系统、法拉第杯和反应室组成。\n",
        encoding="utf-8",
    )
    (tmp_path / "str.md").write_text(
        "# STR wafer\nSTR 作业 recipe 跑货前需确认 offline monitor 和 pi-run 结果，并按 PRS 要求收集 inline 数据。\n",
        encoding="utf-8",
    )
    return MarkdownBM25Index.from_directory(tmp_path, chunk_chars=240, overlap_chars=20)


def test_chinese_retrieval_ranks_matching_document(tmp_path):
    index = _build_index(tmp_path)
    results = index.search("离子注入系统由哪些部分组成", top_k=2)
    assert index.num_documents == 2
    assert results[0].source == "implant.md"
    assert "分析磁场" in results[0].text


def test_mixed_english_retrieval(tmp_path):
    index = _build_index(tmp_path)
    results = index.search("STR recipe offline monitor", top_k=1)
    assert results[0].source == "str.md"


def test_gb18030_markdown_is_indexed(tmp_path):
    content = "# 化学品安全\n侦测器预报警后应通知区域工程师和安全工程师。"
    (tmp_path / "safety.markdown").write_bytes(content.encode("gb18030"))
    index = MarkdownBM25Index.from_directory(
        tmp_path,
        chunk_chars=240,
        overlap_chars=20,
    )
    results = index.search("侦测器预报警 安全工程师", top_k=1)
    assert results[0].source == "safety.markdown"


def test_search_action_uses_last_tag_and_accepts_missing_close():
    assert extract_search_query("<search>旧词</search>\n<search>新关键词</search>") == "新关键词"
    assert extract_search_query("思考后 <search>离子注入 组成") == "离子注入 组成"
    assert extract_search_query("直接作答") is None


def test_question_context_and_query_metadata():
    prompt = "请分析。\n题目：SERVER ROOM 通过什么连接\n\n选项：\nA. SQL"
    assert question_context(prompt) == "SERVER ROOM 通过什么连接"
    combined = build_retrieval_query("server room", prompt, "MFG OJT")
    assert combined.splitlines() == [
        "server room",
        "SERVER ROOM 通过什么连接",
        "MFG OJT",
    ]


def test_result_format_is_bounded(tmp_path):
    index = _build_index(tmp_path)
    results = index.search("离子注入", top_k=2)
    rendered = format_search_results(
        results,
        "离子注入",
        max_chars=180,
        per_result_chars=80,
    )
    assert rendered.startswith("[检索结果]")
    assert "来源：" in rendered
    assert len(rendered) <= 180


def test_document_quality_classification():
    answer = classify_document_quality("培训试题&答案.md", "答案", "正确答案：A")
    question = classify_document_quality("设备培训试卷.md", "填空题", "1. SERVER ROOM 通过____连接")
    reference = classify_document_quality("设备手册.md", "连接架构", "SERVER ROOM 通过 OPC server 连接。")
    noise = classify_document_quality("ocr.md", "Page 1", "\ufffd\ufffd\ufffd")

    assert answer.category == "answer-bearing"
    assert question.category == "question-only"
    assert reference.category == "reference"
    assert noise.category == "noise"
    assert answer.weight > reference.weight > question.weight > noise.weight


def test_quality_rerank_demotes_question_only_exam(tmp_path):
    (tmp_path / "exact-question-exam.md").write_text(
        "# SERVER ROOM 试题\nSERVER ROOM 通过什么与 Clean room 连接？\n",
        encoding="utf-8",
    )
    (tmp_path / "system-manual.md").write_text(
        "# SERVER ROOM 连接架构\n设备手册规定 SERVER ROOM 通过 OPC server 与 Clean room 连接。\n",
        encoding="utf-8",
    )
    index = MarkdownBM25Index.from_directory(tmp_path, chunk_chars=240, overlap_chars=20)

    baseline = index.search("SERVER ROOM 通过什么与 Clean room 连接", top_k=1)
    reranked = index.search(
        "SERVER ROOM 通过什么与 Clean room 连接",
        top_k=1,
        candidate_k=2,
        quality_rerank=True,
    )

    assert baseline[0].source == "exact-question-exam.md"
    assert reranked[0].source == "system-manual.md"
    assert reranked[0].quality_category == "reference"
    assert index.quality_category_counts["question-only"] == 1


def test_hybrid_rerank_combines_semantics_and_quality(tmp_path):
    (tmp_path / "device-exam.md").write_text(
        "# Device Exam\nHow many monitor items does ICS8000 have?\n",
        encoding="utf-8",
    )
    (tmp_path / "device-manual.md").write_text(
        "# Device Manual\nThe ICS8000 concentration monitor has twelve monitored items.\n",
        encoding="utf-8",
    )
    index = MarkdownBM25Index.from_directory(tmp_path, chunk_chars=240, overlap_chars=20)
    candidates = index.search("ICS8000 monitor items", top_k=2)
    semantic_scores = [0.95 if result.source == "device-exam.md" else 0.85 for result in candidates]

    semantic = rerank_by_semantic(candidates, semantic_scores)
    hybrid = reciprocal_rank_fusion(candidates, semantic_scores)

    assert semantic[0].source == "device-exam.md"
    assert hybrid[0].source == "device-manual.md"


def test_agent_search_then_submit_answer(tmp_path):
    runner = QARetrievalRunner(_build_index(tmp_path), qa_rule_reward_fn)
    metadata = {
        "query": "题目：离子注入系统包括什么？",
        "expected_answer": "[single] A",
        "bank": "IMP 培训",
        "search_count": 0,
        "search_queries": [],
        "invalid_count": 0,
    }

    search_turn = runner.process("<search>离子注入 系统组成</search>", metadata)
    assert search_turn.terminated is False
    assert search_turn.reward == 0.0
    assert search_turn.metadata["search_count"] == 1
    assert "implant.md" in search_turn.observation

    final_turn = runner.process(r"根据资料，答案为 \boxed{A}", search_turn.metadata)
    assert final_turn.terminated is True
    assert final_turn.reward == 1.0
    assert final_turn.answer == "[single] A"


def test_agent_penalizes_search_over_limit(tmp_path):
    runner = QARetrievalRunner(
        _build_index(tmp_path),
        qa_rule_reward_fn,
        max_searches=1,
    )
    metadata = {
        "query": "题目：离子注入系统包括什么？",
        "expected_answer": "[single] A",
        "search_count": 1,
    }
    turn = runner.process("<search>再次检索</search>", metadata)
    assert turn.terminated is True
    assert turn.reward == FORMAT_PENALTY


def test_agent_allows_one_format_retry_then_penalizes(tmp_path):
    runner = QARetrievalRunner(_build_index(tmp_path), qa_rule_reward_fn)
    metadata = {
        "query": "题目：测试",
        "expected_answer": "[single] A",
        "invalid_count": 0,
    }
    first = runner.process("只有分析，没有动作", metadata)
    assert first.terminated is False
    assert first.metadata["invalid_count"] == 1

    second = runner.process("仍然没有动作", first.metadata)
    assert second.terminated is True
    assert second.reward == FORMAT_PENALTY


def test_agent_rejects_mixed_search_and_final_answer(tmp_path):
    runner = QARetrievalRunner(_build_index(tmp_path), qa_rule_reward_fn)
    metadata = {
        "query": "题目：离子注入系统包括什么？",
        "expected_answer": "[single] A",
        "search_count": 0,
        "search_queries": [],
        "invalid_count": 0,
    }

    turn = runner.process(
        r"<search>离子注入 系统组成</search> \boxed{A}",
        metadata,
    )

    assert not turn.terminated
    assert turn.reward == 0.0
    assert turn.metadata["invalid_count"] == 1
    assert turn.metadata["search_count"] == 0
    assert "不能同时检索并提交" in turn.observation


def test_agent_terminates_invalid_answer_after_searches_exhausted(tmp_path):
    runner = QARetrievalRunner(
        _build_index(tmp_path),
        qa_rule_reward_fn,
        max_searches=1,
    )
    metadata = {
        "query": "题目：测试",
        "expected_answer": "[single] A",
        "search_count": 1,
        "invalid_count": 0,
    }
    turn = runner.process("检索后仍未提交答案", metadata)
    assert turn.terminated is True
    assert turn.reward == FORMAT_PENALTY


def test_agent_forces_warmup_search_before_open_answer(tmp_path):
    runner = QARetrievalRunner(_build_index(tmp_path), qa_rule_reward_fn)
    metadata = {
        "query": "题目：离子注入系统包括什么？",
        "expected_answer": "[short] 离子源 ||| 分析磁场",
        "force_search": True,
        "search_count": 0,
        "search_queries": [],
    }

    rejected = runner.process(r"\boxed{离子源; 分析磁场}", metadata)
    assert rejected.terminated is False
    assert rejected.reward == 0.0
    assert "必须先检索" in rejected.observation

    searched = runner.process("<search>离子注入 系统组成</search>", rejected.metadata)
    submitted = runner.process(r"\boxed{离子源; 分析磁场}", searched.metadata)
    assert submitted.terminated is True
    assert submitted.reward == 1.0


def test_agent_enforces_two_distinct_retrieval_turns_when_required(tmp_path):
    runner = QARetrievalRunner(_build_index(tmp_path), qa_rule_reward_fn)
    metadata = {
        "query": "题目：离子注入系统包括什么？",
        "expected_answer": "[short] 离子源 ||| 分析磁场",
        "minimum_searches": 2,
        "search_count": 0,
        "search_queries": [],
    }

    first = runner.process("<search>离子注入 系统组成</search>", metadata)
    assert first.terminated is False
    assert "还需检索 1 次" in first.observation

    premature = runner.process(r"\boxed{离子源; 分析磁场}", first.metadata)
    assert premature.terminated is False
    assert "必须先检索 2 次" in premature.observation

    second = runner.process("<search>分析磁场 工作原理</search>", premature.metadata)
    submitted = runner.process(r"\boxed{离子源; 分析磁场}", second.metadata)
    assert submitted.terminated is True
    assert submitted.reward == 1.0


def test_agent_rewards_only_incremental_evidence_and_penalizes_duplicate(tmp_path):
    runner = QARetrievalRunner(
        _build_index(tmp_path),
        qa_rule_reward_fn,
        evidence_reward_scale=0.1,
        search_cost=0.01,
        duplicate_query_penalty=0.02,
    )
    metadata = {
        "query": "题目：离子注入系统包括什么？",
        "expected_answer": "[short] 离子源 ||| 分析磁场 ||| 真空系统",
        "search_count": 0,
        "search_queries": [],
        "evidence_hits": [],
    }

    first = runner.process("<search>离子注入 系统组成</search>", metadata)
    assert first.reward == pytest.approx(0.1 * (2 / 3) - 0.01)
    assert first.metadata["evidence_coverage"] == pytest.approx(2 / 3)

    duplicate = runner.process("<search>离子注入 系统组成</search>", first.metadata)
    assert duplicate.reward == pytest.approx(-0.03)
    assert duplicate.metadata["evidence_coverage"] == pytest.approx(2 / 3)


def test_agent_does_not_reward_keypoint_hidden_outside_visible_snippet(tmp_path):
    (tmp_path / "hidden.md").write_text(
        "# Hidden evidence\n"
        + "queryanchor "
        + ("中间内容" * 40)
        + " 秘密答案",
        encoding="utf-8",
    )
    index = MarkdownBM25Index.from_directory(
        tmp_path,
        chunk_chars=400,
        overlap_chars=20,
    )
    runner = QARetrievalRunner(
        index,
        qa_rule_reward_fn,
        top_k=1,
        candidate_k=1,
        max_result_chars=200,
        per_result_chars=60,
        evidence_reward_scale=0.1,
    )
    metadata = {
        "query": "题目：请根据 queryanchor 资料作答。",
        "expected_answer": "[short] 秘密答案",
        "search_count": 0,
        "search_queries": [],
        "evidence_hits": [],
    }

    turn = runner.process("<search>queryanchor</search>", metadata)

    assert "秘密答案" not in turn.observation
    assert turn.metadata["evidence_hits"] == []
    assert turn.metadata["evidence_coverage"] == 0.0
    assert turn.reward == 0.0


def test_agent_requires_visible_keypoint_in_same_trusted_result():
    class StubIndex:
        def search(
            self,
            _query: str,
            *,
            top_k: int,
            candidate_k: int,
            quality_rerank: bool,
        ) -> list[SearchResult]:
            del top_k, candidate_k, quality_rerank
            return [
                SearchResult(
                    source="questions.md",
                    heading="考试题",
                    text="queryanchor alpha",
                    score=2.0,
                    quality_category="question-only",
                ),
                SearchResult(
                    source="reference.md",
                    heading="参考资料",
                    text="queryanchor alpha",
                    score=1.0,
                    quality_category="reference",
                ),
            ]

    runner = QARetrievalRunner(
        StubIndex(),  # type: ignore[arg-type]
        qa_rule_reward_fn,
        max_result_chars=75,
        per_result_chars=100,
        evidence_reward_scale=0.1,
    )
    turn = runner.process(
        "<search>queryanchor</search>",
        {
            "expected_answer": "[short] alpha",
            "query": "测试问题",
            "search_count": 0,
        },
    )

    assert turn.reward == 0.0
    assert turn.metadata is not None
    assert turn.metadata["evidence_hits"] == []
    assert turn.metadata["evidence_coverage"] == 0.0


def test_agent_validation_shaping_defaults_to_disabled(tmp_path):
    runner = QARetrievalRunner(_build_index(tmp_path), qa_rule_reward_fn)
    metadata = {
        "query": "题目：离子注入系统包括什么？",
        "expected_answer": "[short] 离子源 ||| 分析磁场",
        "search_count": 0,
        "search_queries": [],
    }

    first = runner.process("<search>离子注入 系统组成</search>", metadata)
    duplicate = runner.process("<search>离子注入 系统组成</search>", first.metadata)
    assert first.reward == 0.0
    assert duplicate.reward == 0.0
