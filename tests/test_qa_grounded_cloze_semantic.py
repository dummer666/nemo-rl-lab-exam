import json

from experiments.qa_grounded_cloze_semantic_wanghaonan import run


def _verdict(**overrides):
    payload = {
        "decision": "accept",
        "confidence": 0.96,
        "complete_natural_statement": True,
        "meaningful_answer": True,
        "noise": {field: False for field in run.CRITIC_NOISE_FIELDS},
        "reason": "完整技术陈述",
    }
    payload.update(overrides)
    return payload


def test_strict_verdict_accepts_clean_schema():
    payload, error = run._validate_verdict(
        json.dumps(_verdict(), ensure_ascii=False)
    )

    assert error is None
    assert payload["decision"] == "accept"


def test_strict_verdict_rejects_missing_field_and_extra_text():
    missing = _verdict()
    del missing["meaningful_answer"]

    assert run._validate_verdict(json.dumps(missing))[1] == "schema_fields"
    assert run._validate_verdict("result: " + json.dumps(_verdict()))[1] == (
        "invalid_json"
    )


def test_strict_verdict_rejects_contradictory_accept():
    payload = _verdict(
        noise={
            **_verdict()["noise"],
            "truncated_fragment": True,
        }
    )

    assert run._validate_verdict(json.dumps(payload))[1] == (
        "schema_contradiction"
    )


def test_strict_verdict_allows_consistent_reject():
    payload = _verdict(
        decision="reject",
        confidence=0.99,
        complete_natural_statement=False,
        reason="句子截断",
        noise={
            **_verdict()["noise"],
            "truncated_fragment": True,
        },
    )

    parsed, error = run._validate_verdict(json.dumps(payload))

    assert error is None
    assert parsed["decision"] == "reject"
