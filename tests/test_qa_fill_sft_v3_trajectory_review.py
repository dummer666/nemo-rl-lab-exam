from experiments.qa_fill_sft_v3_trajectory_review_wanghaonan import run


def _row(row_index, step, diagnosis):
    return {
        "row_index": row_index,
        "step": step,
        "primary_diagnosis": diagnosis,
    }


def test_review_selection_keeps_all_versions_of_critical_rows():
    rows = [
        _row(1, 20, "off_target_second_search"),
        _row(1, 40, "reward_regression"),
        _row(2, 40, "incremental_evidence_not_used"),
        _row(3, 40, "off_target_second_search"),
        _row(4, 40, "redundant_second_search"),
    ]

    selected = run.select_review_rows(rows)

    assert [[row["step"] for row in versions] for versions in selected[:2]] == [
        [20, 40],
        [40],
    ]
    assert {
        versions[0]["row_index"]
        for versions in selected
    } == {1, 2, 3, 4}
