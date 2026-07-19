from __future__ import annotations

from common.retrieval.qa_target_rebuild import question_fingerprint
from experiments.qa_fill_only_audit_wanghaonan import run as fill_audit

_machine_gate = fill_audit._machine_gate
_review_samples = fill_audit._review_samples
_unique_fill_sources = fill_audit._unique_fill_sources


def _source(row_id: int, *, split: str, query: str, expected: str = "[fill] x"):
    return {
        "row_id": row_id,
        "question_type": "fill",
        "query": query,
        "expected_answer": expected,
        "split": split,
        "_audit": {
            "first_query": f"{query} first",
            "second_query": f"{query} second",
        },
    }


def _accepted(index: int, *, split: str, search_turns: int = 1):
    query = f"fill question {index}"
    return {
        "source_row_id": index,
        "question_fingerprint": question_fingerprint(query),
        "question_type": "fill",
        "query": query,
        "split": split,
        "search_turns": search_turns,
        "search_hops": [
            {
                "hop": hop,
                "model_search_query": f"fill query {index} hop {hop}",
            }
            for hop in range(1, search_turns + 1)
        ],
        "human_reviewed": False,
        "_audit": {
            "trusted_visible_coverage": 1.0,
            "incremental_two_hop": search_turns == 2,
            "query_leakage_check": True,
            "official_validation_fingerprint_overlap": False,
            "token_length": 100 + index,
            "runtime_raw_chunk_alignment": True,
        },
    }


def test_unique_fill_sources_deduplicates_and_rejects_unsafe_groups():
    rows = [
        _source(1, split="train", query="same"),
        _source(2, split="train", query="same"),
        _source(3, split="train", query="cross"),
        _source(4, split="validation", query="cross"),
        _source(5, split="train", query="conflict", expected="[fill] a"),
        _source(6, split="train", query="conflict", expected="[fill] b"),
    ]

    selected, rejected, stats = _unique_fill_sources(rows)

    assert [row["row_id"] for row in selected] == [1]
    assert selected[0]["_fill_audit_source"]["source_row_ids"] == [1, 2]
    assert {row["decision"] for row in rejected} == {
        "source_question_cross_split",
        "source_question_conflicting_answers",
    }
    assert stats == {
        "raw_fill_rows": 6,
        "unique_source_questions": 3,
        "same_question_duplicate_rows": 3,
        "eligible_unique_questions": 1,
        "source_prefilter_rejections": 2,
    }


def test_review_samples_include_every_two_hop_and_twenty_others():
    accepted = [
        *[
            _accepted(index, split="train", search_turns=2)
            for index in range(3)
        ],
        *[
            _accepted(index, split="train")
            for index in range(3, 28)
        ],
    ]

    first = _review_samples(accepted)
    second = _review_samples(accepted)

    assert first == second
    assert len(first) == 23
    assert sum(row["review_selection"] == "all_two_hop" for row in first) == 3
    assert sum(
        row["review_selection"] == "deterministic_random"
        for row in first
    ) == 20
    assert all(row["human_reviewed"] is False for row in first)


def test_fill_audit_reuses_shared_builder_and_records_leak_rejections(
    monkeypatch,
):
    rows = [
        _source(1, split="train", query="accepted fill"),
        _source(2, split="validation", query="leaky fill"),
    ]
    calls = []

    def fake_fill_trajectory(source, index, tokenizer, official_fingerprints):
        calls.append(
            (
                source["row_id"],
                index,
                tokenizer,
                official_fingerprints,
            )
        )
        if source["row_id"] == 2:
            return None, "first_query_answer_leak"
        record = _accepted(1, split="train")
        record["query"] = source["query"]
        record["question_fingerprint"] = question_fingerprint(source["query"])
        return record, "accepted"

    monkeypatch.setattr(fill_audit, "_fill_trajectory", fake_fill_trajectory)
    index = object()
    tokenizer = object()
    official = {"official"}

    accepted, audit_rows, _stats = fill_audit._audit_fill_sources(
        rows,
        index,
        tokenizer,
        official,
    )

    assert len(accepted) == 1
    assert [call[0] for call in calls] == [1, 2]
    assert all(call[1:] == (index, tokenizer, official) for call in calls)
    rejection = next(row for row in audit_rows if not row["accepted"])
    assert rejection["decision"] == "first_query_answer_leak"
    assert rejection["answer_leak_detected"]


def test_fill_machine_gate_requires_all_isolated_split_thresholds():
    accepted = [
        *[_accepted(index, split="train") for index in range(40)],
        *[
            _accepted(100 + index, split="validation")
            for index in range(8)
        ],
        *[
            _accepted(200 + index, split="rl_holdout", search_turns=2)
            for index in range(8)
        ],
    ]
    audit_rows = [
        {
            "decision": "accepted",
            "accepted": True,
        }
        for _record in accepted
    ]

    gate = _machine_gate(accepted, audit_rows, set())

    assert gate["passed"]
    assert gate["training_ready_count"] == 0
    assert gate["human_review_still_required"]
    assert not gate["auto_training_allowed"]

    gate = _machine_gate(accepted[:-1], audit_rows[:-1], set())
    assert not gate["passed"]
    assert not gate["split_threshold_checks"]["rl_holdout"]
