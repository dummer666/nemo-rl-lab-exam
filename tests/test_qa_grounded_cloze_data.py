from experiments.qa_grounded_cloze_data_wanghaonan import run


def test_answer_candidates_keep_groundable_terms_and_units():
    sentence = "ILD 厚度控制在 120nm，RMS 系统记录量测结果。"

    candidates = run._answer_candidates(sentence)

    assert ("numeric_unit", "120nm") in candidates
    assert ("acronym", "ILD") in candidates
    assert ("acronym", "RMS") in candidates


def test_fill_prompt_numbers_independent_masked_sentences():
    prompt = run._fill_prompt(
        [
            "ILD 厚度为【1】。",
            "RMS 的全称是【1】。",
        ]
    )

    assert "1. ILD 厚度为【1】" in prompt
    assert "2. RMS 的全称是【2】" in prompt


def test_source_split_is_deterministic():
    source = "module/reference.md"

    assert run._source_split(source) == run._source_split(source)


def test_pair_rejects_answer_cross_leakage():
    first = {
        "source": "a.md",
        "split": "train",
        "answer": "ILD",
        "masked_sentence": "层间介质称为【1】。",
    }
    second = {
        "source": "b.md",
        "split": "train",
        "answer": "RMS",
        "masked_sentence": "ILD 系统通过【1】控制。",
    }

    assert run._can_pair(first, second) is False
