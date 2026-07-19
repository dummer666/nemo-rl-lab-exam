from __future__ import annotations

import pytest

from experiments.qa_fill_sft_v3_data_wanghaonan.run import (
    build_oversampled_pack,
)


def _record(
    index: int,
    *,
    split: str,
    question_type: str,
    search_turns: int = 0,
) -> dict:
    return {
        "source_row_id": index,
        "question_fingerprint": f"{split}-{question_type}-{index}",
        "question_type": question_type,
        "split": split,
        "search_turns": search_turns,
        "messages": [{"role": "user", "content": str(index)}],
    }


def _fills(*, train_two_hop: int = 7) -> list[dict]:
    rows = [
        _record(
            index,
            split="train",
            question_type="fill",
            search_turns=2 if index < train_two_hop else 1,
        )
        for index in range(38)
    ]
    rows.extend(
        _record(
            100 + index,
            split="validation",
            question_type="fill",
            search_turns=2 if index == 0 else 1,
        )
        for index in range(5)
    )
    rows.extend(
        _record(
            200 + index,
            split="rl_holdout",
            question_type="fill",
            search_turns=2 if index == 0 else 1,
        )
        for index in range(7)
    )
    return rows


def _objectives() -> list[dict]:
    rows = []
    index = 1000
    for split, per_type in (("train", 24), ("validation", 4)):
        for question_type in ("single", "multiple", "bool"):
            for _ in range(per_type):
                rows.append(
                    _record(
                        index,
                        split=split,
                        question_type=question_type,
                    )
                )
                index += 1
    return rows


def test_v3_pack_raises_two_hop_exposure_and_preserves_replay():
    pack, summary = build_oversampled_pack(_fills(), _objectives())

    assert summary["profiles"]["train"] == {
        "fill_one_hop": 31,
        "fill_two_hop": 28,
        "objective": 24,
    }
    assert summary["profiles"]["validation"] == {
        "fill_one_hop": 4,
        "fill_two_hop": 1,
        "objective": 6,
    }
    assert len(pack["train"]) == 83
    assert len(pack["validation"]) == 11
    assert len(pack["rl_holdout"]) == 7
    assert summary["objective_train_per_type"] == {
        "bool": 8,
        "multiple": 8,
        "single": 8,
    }
    assert summary["objective_replay_fraction"] == 24 / 83


def test_v3_pack_requires_exact_audited_two_hop_profile():
    with pytest.raises(ValueError, match="unexpected train fill profile"):
        build_oversampled_pack(
            _fills(train_two_hop=6),
            _objectives(),
        )
