from experiments.qa_training_clean_wanghaonan.run import (
    _analyze_structure,
    _apply_duplicate_policy,
    _sample_weight,
)


def test_structure_validation_flags_invalid_objective_answer():
    record = _analyze_structure(
        {
            "query": "题目：测试\n选项：\nA. 对\nB. 错",
            "expected_answer": "[single] C",
        },
        0,
    )

    assert "answer_letter_out_of_range" in record["fatal_issues"]
    assert _sample_weight(record) == 0.0


def test_fill_mismatch_is_warning_not_fatal():
    record = _analyze_structure(
        {
            "query": "题目：填写【1】和【2】",
            "expected_answer": "[fill] 一个答案",
        },
        0,
    )

    assert record["fatal_issues"] == []
    assert record["issues"] == ["fill_blank_count_mismatch"]


def test_duplicate_policy_excludes_repeated_rows_and_conflicts():
    duplicate_rows = [
        _analyze_structure(
            {"query": "题目：A?\n选项：\nA. 是\nB. 否", "expected_answer": "[single] A"},
            row_id,
        )
        for row_id in range(2)
    ]
    _apply_duplicate_policy(duplicate_rows)

    assert duplicate_rows[0]["duplicate_of"] is None
    assert duplicate_rows[1]["duplicate_of"] == 0

    conflict_rows = [
        _analyze_structure(
            {
                "query": "题目：B?\n选项：\nA. 是\nB. 否",
                "expected_answer": f"[single] {answer}",
            },
            row_id,
        )
        for row_id, answer in enumerate(("A", "B"))
    ]
    _apply_duplicate_policy(conflict_rows)

    assert all("duplicate_answer_conflict" in row["fatal_issues"] for row in conflict_rows)


def test_open_answer_sampling_weights():
    record = _analyze_structure(
        {"query": "题目：说明作用", "expected_answer": "[short] 要点一 ||| 要点二"},
        0,
    )

    record["support_level"] = "full"
    assert _sample_weight(record) == 3.0
    record["support_level"] = "partial"
    assert _sample_weight(record) == 2.0
    record["support_level"] = "none"
    assert _sample_weight(record) == 0.25
