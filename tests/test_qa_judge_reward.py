from __future__ import annotations

import json
import urllib.error

from common.rewards import qa_judge_reward


class _Response:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def getcode(self) -> int:
        return self.status

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_probe_requires_explicit_injected_environment():
    called = False

    def opener(request, timeout):
        nonlocal called
        called = True
        raise AssertionError("probe must not call a default local endpoint")

    report = qa_judge_reward.probe_judge_endpoint({}, opener=opener)

    assert not called
    assert report["available"] is False
    assert report["fallback_root_cause"] == (
        "missing_env:JUDGE_BASE_URL,JUDGE_MODEL,JUDGE_API_KEY"
    )
    assert report["base_url_present"] is False
    assert report["api_key_present"] is False


def test_probe_uses_bearer_auth_without_serializing_secrets():
    base_url = "https://private-judge.invalid/v1"
    api_key = "top-secret-judge-key"
    seen_authorization = []

    def opener(request, timeout):
        assert timeout == 3.0
        seen_authorization.append(request.get_header("Authorization"))
        if request.full_url.endswith("/models"):
            return _Response(
                {
                    "data": [
                        {"id": "judge-model"},
                        {"id": "other-model"},
                    ]
                }
            )
        body = json.loads(request.data)
        assert body["model"] == "judge-model"
        student_answer = body["messages"][1]["content"]
        score = (
            1.0
            if "alert developers" in student_answer
            else 0.0
        )
        return _Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"score": score, "reason": "ok"}
                            )
                        }
                    }
                ]
            }
        )

    report = qa_judge_reward.probe_judge_endpoint(
        {
            "JUDGE_BASE_URL": base_url,
            "JUDGE_MODEL": "judge-model",
            "JUDGE_API_KEY": api_key,
            "JUDGE_TIMEOUT": "3",
        },
        opener=opener,
    )

    assert report["available"] is True
    assert report["models"]["http_status"] == 200
    assert report["models"]["data_ids"] == [
        "judge-model",
        "other-model",
    ]
    assert report["synthetic_scores"]["semantic_match"]["score"] == 1.0
    assert report["synthetic_scores"]["keyword_trap"]["score"] == 0.0
    assert report["synthetic_scores"]["margin"] == 1.0
    assert report["synthetic_scores"]["discriminates"] is True
    assert len(seen_authorization) == 3
    assert seen_authorization == [
        f"Bearer {api_key}",
        f"Bearer {api_key}",
        f"Bearer {api_key}",
    ]
    serialized = json.dumps(report)
    assert base_url not in serialized
    assert api_key not in serialized


def test_probe_http_failures_are_sanitized():
    base_url = "https://private-judge.invalid/v1"
    api_key = "top-secret-judge-key"

    def opener(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "unauthorized",
            None,
            None,
        )

    report = qa_judge_reward.probe_judge_endpoint(
        {
            "JUDGE_BASE_URL": base_url,
            "JUDGE_MODEL": "judge-model",
            "JUDGE_API_KEY": api_key,
        },
        opener=opener,
    )

    assert report["available"] is False
    assert report["fallback_root_cause"] == (
        "models_request_failed:http_401"
    )
    serialized = json.dumps(report)
    assert base_url not in serialized
    assert api_key not in serialized


def test_probe_requires_semantic_discrimination():
    def opener(request, timeout):
        if request.full_url.endswith("/models"):
            return _Response({"data": [{"id": "judge-model"}]})
        return _Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"score": 0.8, "reason": "same"}'
                        }
                    }
                ]
            }
        )

    report = qa_judge_reward.probe_judge_endpoint(
        {
            "JUDGE_BASE_URL": "https://private-judge.invalid/v1",
            "JUDGE_MODEL": "judge-model",
            "JUDGE_API_KEY": "fake-key",
        },
        opener=opener,
    )

    assert report["available"] is False
    assert report["synthetic_scores"]["discriminates"] is False
    assert report["fallback_root_cause"] == (
        "synthetic_scores_do_not_discriminate"
    )


def test_probe_malformed_url_is_sanitized_and_does_not_raise():
    base_url = "not a valid judge url"
    api_key = "top-secret-judge-key"

    report = qa_judge_reward.probe_judge_endpoint(
        {
            "JUDGE_BASE_URL": base_url,
            "JUDGE_MODEL": "judge-model",
            "JUDGE_API_KEY": api_key,
        }
    )

    assert report["available"] is False
    assert report["fallback_root_cause"] == (
        "models_request_failed:invalid_url"
    )
    serialized = json.dumps(report)
    assert base_url not in serialized
    assert api_key not in serialized


def test_invalid_runtime_timeout_returns_no_score(monkeypatch):
    called = False

    def opener(request, timeout):
        nonlocal called
        called = True
        raise AssertionError("invalid timeout must prevent the request")

    monkeypatch.setattr(qa_judge_reward, "JUDGE_TIMEOUT", "invalid")
    monkeypatch.setattr(
        qa_judge_reward.urllib.request,
        "urlopen",
        opener,
    )

    score = qa_judge_reward.judge_short_answer_score(
        "题目：说明原因。",
        r"部分正确。\boxed{要点一}",
        "[short] 要点一 ||| 要点二",
    )

    assert score is None
    assert not called


def test_invalid_runtime_concurrency_falls_back_to_keyword(monkeypatch):
    monkeypatch.setattr(
        qa_judge_reward,
        "JUDGE_CONCURRENCY",
        "invalid",
    )
    monkeypatch.setattr(
        qa_judge_reward,
        "_call_judge",
        lambda prompt: (_ for _ in ()).throw(
            AssertionError("judge request must not run")
        ),
    )

    rewards = qa_judge_reward.qa_judge_reward_fn(
        ["题目：说明优点。"],
        [r"答案。\boxed{低温}"],
        ["[short] 低温 ||| 纯度高"],
    )

    assert rewards == [0.5]


def test_semantic_judge_call_uses_configured_bearer_key(monkeypatch):
    api_key = "runtime-secret-key"
    seen_authorization = []

    def opener(request, timeout):
        seen_authorization.append(request.get_header("Authorization"))
        return _Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"score": 0.75, "reason": "partial"}'
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(
        qa_judge_reward,
        "JUDGE_BASE_URL",
        "https://private-judge.invalid/v1",
    )
    monkeypatch.setattr(qa_judge_reward, "JUDGE_MODEL", "judge-model")
    monkeypatch.setattr(qa_judge_reward, "JUDGE_API_KEY", api_key)
    monkeypatch.setattr(
        qa_judge_reward.urllib.request,
        "urlopen",
        opener,
    )

    score = qa_judge_reward.judge_short_answer_score(
        "题目：说明原因。",
        r"部分正确。\boxed{要点一}",
        "[short] 要点一 ||| 要点二",
    )

    assert score == 0.75
    assert seen_authorization == [f"Bearer {api_key}"]
