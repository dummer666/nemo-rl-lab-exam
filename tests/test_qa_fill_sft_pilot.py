from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

RUN_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "sft_qwen3.5-9b_qa-fill-v2_wanghaonan"
    / "run.py"
)
SPEC = importlib.util.spec_from_file_location("qa_fill_sft_pilot_run", RUN_PATH)
assert SPEC is not None and SPEC.loader is not None
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


def _row(environment_turns: int) -> dict:
    messages = [{"role": "user", "content": "question"}]
    for index in range(environment_turns):
        messages.extend(
            [
                {"role": "assistant", "content": f"<search>{index}</search>"},
                {"role": "environment", "content": f"result {index}"},
            ]
        )
    messages.append({"role": "assistant", "content": r"\boxed{answer}"})
    return {"messages": messages}


def test_dataset_profile_and_preflight_selection_cover_all_routes():
    rows = [
        *[_row(0) for _ in range(15)],
        *[_row(1) for _ in range(31)],
        *[_row(2) for _ in range(7)],
    ]

    assert run.dataset_profile(rows) == {0: 15, 1: 31, 2: 7}
    selected = run.select_preflight_samples(rows)
    assert run.dataset_profile(selected) == {0: 1, 1: 2, 2: 2}


def test_dataset_profile_rejects_invalid_runtime_roles():
    with pytest.raises(ValueError, match="invalid runtime role sequence"):
        run.dataset_profile(
            [
                {
                    "messages": [
                        {"role": "user", "content": "question"},
                        {"role": "environment", "content": "bad"},
                        {"role": "assistant", "content": "answer"},
                    ]
                }
            ]
        )


def test_expected_profiles_can_be_overridden(monkeypatch):
    monkeypatch.setenv(
        "QA_FILL_SFT_EXPECTED_PROFILES",
        '{"train":{"0":24,"1":31,"2":28},'
        '"validation":{"0":6,"1":4,"2":1}}',
    )

    assert run.expected_profiles() == {
        "train": {0: 24, 1: 31, 2: 28},
        "validation": {0: 6, 1: 4, 2: 1},
    }
