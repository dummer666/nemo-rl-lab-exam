from __future__ import annotations

from common.retrieval.evidence import text_keypoint_hits
from common.retrieval.markdown_bm25 import SearchResult
from common.retrieval.qa_sft import visible_retrieval_text
from experiments.qa_sft_data_select_wanghaonan.run import (
    _assign_primary_splits,
    _candidate_record,
    _greedy_support_results,
    _selection_status,
)


def _result(
    source: str,
    text: str,
    *,
    category: str = "reference",
    score: float = 1.0,
) -> SearchResult:
    return SearchResult(
        source=source,
        heading=source,
        text=text,
        score=score,
        raw_score=score,
        quality_category=category,
        quality_weight=1.0,
    )


class _StubIndex:
    def __init__(self, results: list[SearchResult]):
        self.results = results

    def search(self, _query, top_k, *, candidate_k, quality_rerank):
        assert candidate_k == 50
        assert quality_rerank
        return self.results[:top_k]


def test_text_keypoint_hits_only_returns_visible_points():
    keypoints = [["sqlserver"], ["cleanroom", "洁净室"]]

    assert text_keypoint_hits("SQL Server connects to the fab", keypoints) == {0}
    assert text_keypoint_hits("洁净室", keypoints) == {1}


def test_selection_status_separates_direct_and_rewrite_candidates():
    assert _selection_status(1.0, 1.0) == "ready_one_search"
    assert _selection_status(0.5, 1.0) == "needs_query_rewrite"
    assert _selection_status(0.0, 0.5) == "partial_review"
    assert _selection_status(0.0, 0.0) == "excluded_unsupported"


def test_visible_retrieval_text_excludes_ranks_and_scores():
    rendered = "[检索结果]\n1. 来源：doc.md\n相关度：12.50\n没有数字答案"

    visible = visible_retrieval_text(rendered)

    assert "1. 来源" not in visible
    assert "12.50" not in visible
    assert text_keypoint_hits(visible, [["1"]]) == set()


def test_greedy_support_results_prefers_result_with_more_new_keypoints():
    results = [
        _result("one.md", "alpha"),
        _result("both.md", "alpha beta", score=0.5),
        _result("question.md", "alpha beta", category="question-only", score=2.0),
    ]

    selected = _greedy_support_results(results, [["alpha"], ["beta"]])

    assert [record["source"] for record in selected] == ["both.md"]
    assert selected[0]["keypoint_hits"] == [0, 1]


def test_candidate_record_uses_rendered_top_four_for_readiness():
    results = [
        _result("answer.md", "alpha beta"),
        _result("reference.md", "unrelated"),
    ]
    row = {
        "query": "下面是一道填空题。\n题目：alpha 和 beta",
        "expected_answer": "[fill] alpha ||| beta",
        "meta": {"bank": "demo"},
        "_clean": {"row_id": 7, "support_level": "full"},
    }

    record = _candidate_record(row, _StubIndex(results))

    assert record["row_id"] == 7
    assert record["selection_status"] == "ready_one_search"
    assert record["dataset_role"] == "primary"
    assert record["first_observation_coverage"] == 1.0
    assert record["top20_pool_coverage"] == 1.0


def test_primary_split_is_deterministic_stratified_and_disjoint():
    records = [
        {
            "row_id": row_id,
            "query": f"question-{row_id}",
            "question_type": question_type,
        }
        for row_id, question_type in [
            *((index, "fill") for index in range(10)),
            *((100 + index, "short") for index in range(20)),
        ]
    ]

    first = _assign_primary_splits(records, validation_fraction=0.1, seed=9)
    second = _assign_primary_splits(records, validation_fraction=0.1, seed=9)

    assert first == second
    validation = [record for record in first if record["split"] == "validation"]
    assert sum(record["question_type"] == "fill" for record in validation) == 1
    assert sum(record["question_type"] == "short" for record in validation) == 2
    assert {record["row_id"] for record in validation}.isdisjoint(
        record["row_id"] for record in first if record["split"] == "train"
    )
