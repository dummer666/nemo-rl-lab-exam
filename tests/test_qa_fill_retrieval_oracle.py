from common.retrieval.evidence import fragile_keypoint_indexes
from experiments.qa_fill_retrieval_oracle_wanghaonan import run


def test_blank_context_queries_are_answer_free_and_blank_specific():
    query = (
        "下面是一道填空题。\n"
        "题目：在版图中常用的分地方式是【1】分地，"
        "这种方式利用PN结【2】偏实现隔离。"
    )

    variants = run.blank_context_queries(query)

    assert any("待填" in variant for variant in variants)
    assert any("PN" in variant and "定义" in variant for variant in variants)
    assert all("DNW" not in variant and "反向" not in variant for variant in variants)


def test_fragile_keypoints_are_not_used_for_oracle_gate():
    assert fragile_keypoint_indexes([["dnw"], ["反"], ["5"]]) == {1, 2}
