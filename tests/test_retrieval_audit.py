from common.retrieval.markdown_bm25 import SearchResult
from experiments.retrieval_qa_audit_wanghaonan.run import (
    _evidence_coverage,
    _gold_keypoints,
    _normalize,
)


def test_gold_keypoints_normalize_fill_alternatives():
    question_type, points = _gold_keypoints("[fill] 3 V/3V ||| OPC Server")

    assert question_type == "fill"
    assert points == [["3v"], ["opcserver"]]
    assert _normalize("３ μm") == "3um"


def test_evidence_coverage_ignores_question_only_documents():
    question = SearchResult(
        source="exam.md",
        heading="Exam",
        text="The expected answer phrase appears as an option.",
        score=1.0,
        quality_category="question-only",
    )
    manual = SearchResult(
        source="manual.md",
        heading="Manual",
        text="The equipment uses an OPC server.",
        score=0.9,
        quality_category="reference",
    )
    _, points = _gold_keypoints("[short] expected answer phrase ||| OPC server")

    assert _evidence_coverage([question, manual], points, top_k=2) == 0.5
