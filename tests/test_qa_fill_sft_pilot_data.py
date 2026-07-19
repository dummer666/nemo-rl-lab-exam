from __future__ import annotations

import pytest

from experiments.qa_fill_sft_pilot_data_wanghaonan.run import (
    build_pilot_manifests,
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


def _fill_records(*, train_two_hop: bool = True) -> list[dict]:
    rows = [
        _record(
            index,
            split="train",
            question_type="fill",
            search_turns=2 if train_two_hop and index == 0 else 1,
        )
        for index in range(38)
    ]
    rows.extend(
        _record(
            100 + index,
            split="validation",
            question_type="fill",
            search_turns=1,
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


def _objective_records() -> list[dict]:
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


def test_pilot_pack_preserves_fill_splits_and_balances_replay():
    manifests, summary = build_pilot_manifests(
        _fill_records(),
        _objective_records(),
    )

    assert summary["fill_split_counts"] == {
        "train": 38,
        "validation": 5,
        "rl_holdout": 7,
    }
    assert summary["output_split_counts"] == {
        "train": 53,
        "validation": 11,
        "rl_holdout": 7,
    }
    assert summary["objective_train_counts"] == {
        "bool": 5,
        "multiple": 5,
        "single": 5,
    }
    assert summary["objective_validation_counts"] == {
        "bool": 2,
        "multiple": 2,
        "single": 2,
    }
    assert summary["objective_replay_fraction"] == 15 / 53
    assert sum(
        record["question_type"] == "fill"
        for record in manifests["train"]
    ) == 38


def test_pilot_pack_requires_a_training_two_hop():
    with pytest.raises(
        ValueError,
        match="training split has no two-hop",
    ):
        build_pilot_manifests(
            _fill_records(train_two_hop=False),
            _objective_records(),
        )
