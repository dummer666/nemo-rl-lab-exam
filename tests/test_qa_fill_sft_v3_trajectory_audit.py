from experiments.qa_fill_sft_v3_trajectory_audit_wanghaonan import run


def test_sources_strip_headings_and_deduplicate():
    rendered = (
        "[检索结果]\n"
        "1. 来源：/data/docs/a.md · Heading A\n相关度：4.0\ntext\n\n"
        "2. 来源：/data/docs/a.md · Heading B\n相关度：3.0\ntext\n\n"
        "3. 来源：/data/docs/b.md\n相关度：2.0\ntext"
    )

    assert run._sources(rendered) == ["/data/docs/a.md", "/data/docs/b.md"]


def test_query_similarity_distinguishes_repeat_from_rewrite():
    assert run.query_similarity("湿法刻蚀 清洗", "湿法刻蚀，清洗") == 1.0
    assert run.query_similarity("湿法刻蚀 清洗", "扩散炉 温度") < 0.2


def test_fragile_keypoints_exclude_incidental_single_character_hits():
    keypoints = [["dnw"], ["反"], ["5"], ["层间介质", "ild"]]

    assert run.fragile_keypoint_indexes(keypoints) == {1, 2}


def test_classification_separates_evidence_and_synthesis_failures():
    base = {
        "reward_delta": 0.0,
        "candidate": {"perfect": False},
        "evidence": {
            "full_after_two_hops": False,
            "first_hop_full": False,
            "incremental_keypoint_hits": [1],
            "new_sources": ["new.md"],
        },
    }
    assert run.classify_two_hop(base) == "incremental_evidence_not_used"

    full = {
        **base,
        "evidence": {
            **base["evidence"],
            "full_after_two_hops": True,
        },
    }
    assert run.classify_two_hop(full) == "synthesis_failure_after_full_evidence"

    gain = {**base, "reward_delta": 0.5}
    assert run.classify_two_hop(gain) == "useful_score_gain"
