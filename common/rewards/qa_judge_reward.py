#!/usr/bin/env python3
r"""简答题 LLM-as-judge 奖励（混合判分）。

定位：简答题没有唯一答案，关键词覆盖率（qa_reward.py 的 [short]）只是廉价代理，
会漏判同义表达、也奖励“堆关键词”。本模块用一个裁判 LLM 给简答打 0~1 分，质量更高。

混合策略（推荐）：
    - single/bool/multiple/fill -> 直接走 qa_reward 规则判分（快、客观、零成本）
    - short                     -> 调裁判 LLM 打分；失败则回退到 qa_reward 的关键词覆盖率

Lab 作业优先使用中心服务注入的 OpenAI 兼容裁判端点；仅本地开发时自行启动 vLLM：
    vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001
然后设环境变量：
    JUDGE_BASE_URL=http://127.0.0.1:8001/v1
    JUDGE_MODEL=Qwen/Qwen2.5-7B-Instruct
    JUDGE_API_KEY=EMPTY
    JUDGE_CONCURRENCY=16
    JUDGE_TIMEOUT=30

接口与 qa_reward 一致：qa_judge_reward_fn(queries, completions, expected_answers)。
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

try:
    from common.rewards import qa_reward
except ImportError:
    import qa_reward  # type: ignore

JUDGE_BASE_URL = os.environ.get(
    "JUDGE_BASE_URL",
    "http://127.0.0.1:8001/v1",
)
JUDGE_MODEL = os.environ.get(
    "JUDGE_MODEL",
    "Qwen/Qwen2.5-7B-Instruct",
)
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "EMPTY")
JUDGE_CONCURRENCY = os.environ.get("JUDGE_CONCURRENCY", "16")
JUDGE_TIMEOUT = os.environ.get("JUDGE_TIMEOUT", "30")

_JUDGE_SYS = (
    "你是严格、公正的阅卷老师。根据【参考要点】判断【学生作答】对题目的覆盖与正确程度，"
    "只按事实打分，不被冗长或堆砌关键词迷惑。"
    '输出严格的 JSON：{"score": x, "reason": "..."}，x 为 0 到 1 的小数。'
    "覆盖全部要点且正确=1.0；完全错误或答非所问=0.0；部分正确按比例。"
)


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def _request_json(
    url: str,
    *,
    api_key: str,
    timeout: float,
    body: bytes | None = None,
    opener=None,
) -> tuple[int | None, dict[str, Any] | None, float, str | None]:
    open_request = opener or urllib.request.urlopen
    started = time.perf_counter()
    try:
        request = urllib.request.Request(
            url,
            data=body,
            headers=_headers(api_key),
        )
    except (TypeError, ValueError, UnicodeError):
        return None, None, time.perf_counter() - started, "invalid_url"
    try:
        with open_request(request, timeout=timeout) as response:
            status = int(
                getattr(response, "status", None)
                or response.getcode()
            )
            payload = json.loads(response.read())
        if not isinstance(payload, dict):
            return (
                status,
                None,
                time.perf_counter() - started,
                "invalid_json_shape",
            )
        return status, payload, time.perf_counter() - started, None
    except urllib.error.HTTPError as error:
        return (
            int(error.code),
            None,
            time.perf_counter() - started,
            f"http_{int(error.code)}",
        )
    except urllib.error.URLError:
        return None, None, time.perf_counter() - started, "url_error"
    except TimeoutError:
        return None, None, time.perf_counter() - started, "timeout"
    except OSError:
        return (
            None,
            None,
            time.perf_counter() - started,
            "connection_error",
        )
    except json.JSONDecodeError:
        return None, None, time.perf_counter() - started, "invalid_json"
    except (TypeError, ValueError, UnicodeError):
        return (
            None,
            None,
            time.perf_counter() - started,
            "invalid_response",
        )


def _score_from_payload(payload: Mapping[str, Any]) -> float | None:
    try:
        text = str(payload["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError):
        return None
    match = re.search(r'"score"\s*:\s*([01](?:\.\d+)?)', text)
    if not match:
        match = re.search(r"(?<![\d.])([01](?:\.\d+)?)(?![\d.])", text)
    if not match:
        return None
    score = float(match.group(1))
    return score if 0.0 <= score <= 1.0 else None


def probe_judge_endpoint(
    environ: Mapping[str, str] | None = None,
    *,
    opener=None,
) -> dict[str, Any]:
    """Probe an injected judge endpoint without returning secrets or its URL."""
    env = os.environ if environ is None else environ
    base_url = str(env.get("JUDGE_BASE_URL", "")).strip()
    model = str(env.get("JUDGE_MODEL", "")).strip()
    api_key = str(env.get("JUDGE_API_KEY", "")).strip()
    report: dict[str, Any] = {
        "base_url_present": bool(base_url),
        "model_present": bool(model),
        "model_id": model or None,
        "api_key_present": bool(api_key),
        "models": {
            "http_status": None,
            "latency_ms": None,
            "data_ids": [],
            "configured_model_listed": False,
            "error": None,
        },
        "synthetic_scores": {
            "semantic_match": {
                "http_status": None,
                "latency_ms": None,
                "valid": False,
                "score": None,
                "error": None,
            },
            "keyword_trap": {
                "http_status": None,
                "latency_ms": None,
                "valid": False,
                "score": None,
                "error": None,
            },
            "margin": None,
            "discriminates": False,
        },
        "available": False,
        "fallback_root_cause": None,
    }
    missing = [
        name
        for name, value in (
            ("JUDGE_BASE_URL", base_url),
            ("JUDGE_MODEL", model),
            ("JUDGE_API_KEY", api_key),
        )
        if not value
    ]
    if missing:
        report["fallback_root_cause"] = (
            "missing_env:" + ",".join(missing)
        )
        return report

    try:
        timeout = float(env.get("JUDGE_TIMEOUT", "30"))
    except ValueError:
        report["fallback_root_cause"] = "invalid_timeout"
        return report
    if timeout <= 0:
        report["fallback_root_cause"] = "invalid_timeout"
        return report

    models_status, models_payload, models_seconds, models_error = _request_json(
        f"{base_url.rstrip('/')}/models",
        api_key=api_key,
        timeout=timeout,
        opener=opener,
    )
    model_ids = []
    if models_payload is not None and isinstance(
        models_payload.get("data"),
        list,
    ):
        model_ids = sorted(
            {
                str(item["id"])
                for item in models_payload["data"]
                if isinstance(item, Mapping) and item.get("id")
            }
        )
    report["models"] = {
        "http_status": models_status,
        "latency_ms": round(models_seconds * 1000, 3),
        "data_ids": model_ids,
        "configured_model_listed": model in model_ids,
        "error": models_error,
    }

    def synthetic_score(student_answer: str) -> dict[str, Any]:
        body = json.dumps(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": _JUDGE_SYS},
                    {
                        "role": "user",
                        "content": (
                            "【题目】What is one benefit of unit tests?\n"
                            "【参考要点】They catch regressions.\n"
                            f"【学生作答】{student_answer}\n"
                            "Return JSON only."
                        ),
                    },
                ],
                "temperature": 0.0,
                "max_tokens": 64,
            }
        ).encode("utf-8")
        status, payload, seconds, error = _request_json(
            f"{base_url.rstrip('/')}/chat/completions",
            api_key=api_key,
            timeout=timeout,
            body=body,
            opener=opener,
        )
        score = _score_from_payload(payload) if payload is not None else None
        return {
            "http_status": status,
            "latency_ms": round(seconds * 1000, 3),
            "valid": score is not None,
            "score": score,
            "error": error or (
                None if score is not None else "invalid_score"
            ),
        }

    semantic_match = synthetic_score(
        "They alert developers when a change breaks behavior that worked before."
    )
    keyword_trap = synthetic_score(
        "They catch regressions by approving broken behavior and hiding failures."
    )
    semantic_value = semantic_match["score"]
    trap_value = keyword_trap["score"]
    margin = (
        float(semantic_value) - float(trap_value)
        if semantic_value is not None and trap_value is not None
        else None
    )
    discriminates = bool(
        margin is not None
        and float(semantic_value) >= 0.75
        and float(trap_value) <= 0.25
        and margin >= 0.5
    )
    report["synthetic_scores"] = {
        "semantic_match": semantic_match,
        "keyword_trap": keyword_trap,
        "margin": margin,
        "discriminates": discriminates,
    }

    if models_error:
        root_cause = "models_request_failed:" + models_error
    elif model not in model_ids:
        root_cause = "configured_model_not_listed"
    elif semantic_match["error"]:
        root_cause = (
            "semantic_match_request_failed:"
            + str(semantic_match["error"])
        )
    elif keyword_trap["error"]:
        root_cause = (
            "keyword_trap_request_failed:"
            + str(keyword_trap["error"])
        )
    elif not discriminates:
        root_cause = "synthetic_scores_do_not_discriminate"
    else:
        root_cause = None
    report["available"] = root_cause is None
    report["fallback_root_cause"] = root_cause
    return report


def _question_from_query(query: str) -> str:
    match = re.search(r"题目：(.*)", query, flags=re.DOTALL)
    return match.group(1).strip() if match else query.strip()


def _build_judge_prompt(
    query: str,
    completion: str,
    key_points: list[str],
) -> str:
    boxed = qa_reward.extract_boxed(completion)
    answer = completion.strip()
    if boxed:
        answer += f"\n（学生标注要点：{boxed}）"
    points = "\n".join(f"- {point}" for point in key_points)
    return (
        f"【题目】\n{_question_from_query(query)}\n\n"
        f"【参考要点】\n{points}\n\n"
        f"【学生作答】\n{answer}\n\n"
        "请给出 JSON 评分。"
    )


def _call_judge(prompt: str) -> float | None:
    body = json.dumps(
        {
            "model": JUDGE_MODEL,
            "messages": [
                {"role": "system", "content": _JUDGE_SYS},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 256,
        }
    ).encode("utf-8")
    try:
        timeout = float(JUDGE_TIMEOUT)
    except (TypeError, ValueError):
        return None
    if timeout <= 0:
        return None
    status, payload, _seconds, error = _request_json(
        f"{JUDGE_BASE_URL.rstrip('/')}/chat/completions",
        api_key=JUDGE_API_KEY,
        timeout=timeout,
        body=body,
    )
    if error or status is None or not 200 <= status < 300 or payload is None:
        return None
    return _score_from_payload(payload)


def judge_short_answer_score(
    query: str,
    completion: str,
    expected_answer: str,
) -> float | None:
    """Return a semantic short-answer score without keyword fallback."""
    if not str(expected_answer).lstrip().startswith("[short]"):
        return None
    key_points = [
        point.strip()
        for point in str(expected_answer).split("]", 1)[1].split("|||")
        if point.strip()
    ]
    if not key_points:
        return None
    return _call_judge(
        _build_judge_prompt(query, completion, key_points)
    )


def qa_judge_reward_fn(
    queries,
    completions,
    expected_answers,
    **kwargs,
):
    """Judge short answers semantically and grade other types by rules."""
    groups = qa_reward._load_synonyms()
    rewards: list[float | None] = [None] * len(completions)
    judge_jobs: list[tuple[int, str]] = []

    for index, (query, completion, expected) in enumerate(
        zip(queries, completions, expected_answers, strict=False)
    ):
        if not str(expected).lstrip().startswith("[short]"):
            rewards[index] = qa_reward._grade_one(
                expected,
                completion,
                groups,
            )
            continue
        if qa_reward.extract_boxed(completion) is None:
            rewards[index] = qa_reward.FORMAT_PENALTY
            continue
        key_points = [
            point.strip()
            for point in expected.split("]", 1)[1].split("|||")
            if point.strip()
        ]
        judge_jobs.append(
            (
                index,
                _build_judge_prompt(query, completion, key_points),
            )
        )

    if judge_jobs:
        try:
            concurrency = int(JUDGE_CONCURRENCY)
        except (TypeError, ValueError):
            concurrency = 0
        if concurrency > 0:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                scores = list(
                    executor.map(
                        lambda job: _call_judge(job[1]),
                        judge_jobs,
                    )
                )
        else:
            scores = [None] * len(judge_jobs)
        for (index, _prompt), score in zip(
            judge_jobs,
            scores,
            strict=False,
        ):
            rewards[index] = (
                score
                if score is not None
                else qa_reward._grade_one(
                    expected_answers[index],
                    completions[index],
                    groups,
                )
            )

    return [float(reward) for reward in rewards]


if __name__ == "__main__":
    questions = ["题目：1+1=?", "题目：简述离子注入优点。"]
    completions = [
        r"\boxed{B}",
        r"低温、纯度高、可精确控制浓度。\boxed{低温; 纯度高; 精确控制}",
    ]
    expected = [
        "[single] B",
        "[short] 低温/掺杂 ||| 纯度高 ||| 精确控制",
    ]
    print(
        "rewards:",
        qa_judge_reward_fn(questions, completions, expected),
    )
