from experiments.qa_grounded_cloze_review_wanghaonan import run


def _row(
    answer="ILD",
    sentence="层间介质通常称为 ILD，并广泛用于半导体器件不同金属层之间的电气绝缘。",
):
    return {
        "query": "题目：层间介质通常称为【1】并用于器件之间的绝缘。",
        "expected_answer": f"[fill] {answer}",
        "search_turns": 1,
        "source_candidates": [
            {
                "answer": answer,
                "answer_kind": "acronym",
                "sentence": sentence,
                "masked_sentence": "层间介质通常称为【1】并用于器件之间的绝缘。",
                "model_query": "层间介质 待填 器件 绝缘",
            }
        ],
    }


def test_review_accepts_grounded_masked_row():
    assert run.review_issues(_row()) == []


def test_review_rejects_generic_english_token():
    row = _row(
        answer="THE",
        sentence="THE process description contains enough surrounding technical context.",
    )

    assert "generic_english_token" in run.review_issues(row)
