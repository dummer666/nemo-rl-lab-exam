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


def test_candidate_quality_rejects_slide_and_table_fragments():
    assert "boilerplate_or_slide_noise" in run.candidate_quality_issues(
        r"-- Slide number: 1 -->\nEVG MBDS manual SOP",
        "MBDS",
        "acronym",
    )
    assert "table_fragment" in run.candidate_quality_issues(
        "| 1 | E3000399 | PAC | 卡 |",
        "E3000399",
        "acronym",
    )
    assert "slide_bullet_fragment" in run.candidate_quality_issues(
        "–挖掘FDC数据中的有效信息来解决设备问题并提升整体效能。",
        "FDC",
        "acronym",
    )
    assert "missing_sentence_terminator" in run.candidate_quality_issues(
        "Universal-300 Dual 是由天津华海清科机电科技",
        "Dual",
        "acronym",
    )
    assert "button_or_operation_fragment" in run.candidate_quality_issues(
        "后依次点击FOSB FLAG，再点击ADD，点击Confirm输入密码确认。",
        "FLAG",
        "acronym",
    )
    assert "context_dependent_fragment" in run.candidate_quality_issues(
        "上表中ICPMS测试机台的更新由工程二部进一步跟进定义。",
        "ICPMS",
        "acronym",
    )
    assert "english_predicate_fragment" in run.candidate_quality_issues(
        "Measures the intensity of the DUV light at wafer level.",
        "DUV",
        "acronym",
    )
    assert "numbered_instruction_fragment" in run.candidate_quality_issues(
        "18Check that the High Pressure Exhaust Valve (HPEX) is CLOSED.",
        "HPEX",
        "acronym",
    )
    assert "english_title_fragment" in run.candidate_quality_issues(
        "ASCAL Wafer-by-Wafer LH ILIAS Advanced Lens Control (ALC)",
        "ALC",
        "acronym",
    )
    assert "slide_bullet_fragment" in run.candidate_quality_issues(
        "► 匹配规则包含4个过滤器。",
        "4个",
        "numeric_unit",
    )


def test_candidate_quality_keeps_complete_technical_statements():
    assert run.candidate_quality_issues(
        "系统通过挖掘 FDC 数据中的有效信息来解决设备问题并提升整体效能。",
        "FDC",
        "acronym",
    ) == []
    assert run.candidate_quality_issues(
        "二次电子产生深度通常小于 10nm，因此主要反映试样表面信息。",
        "10nm",
        "numeric_unit",
    ) == []


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


def test_validated_pool_accepts_explicit_larger_targets(monkeypatch):
    candidates = [
        {"candidate_id": str(index), "split": "train"}
        for index in range(4)
    ]
    monkeypatch.setattr(
        run,
        "_build_one_hop",
        lambda _index, candidate, _tokenizer: {"id": candidate["candidate_id"]},
    )

    pools, reasons = run._validated_pools(
        object(),
        candidates,
        object(),
        pool_targets={"train": 3},
    )

    assert len(pools["train"]) == 3
    assert not reasons
