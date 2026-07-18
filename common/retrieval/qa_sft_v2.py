"""Strict selection and isolation helpers for the targeted QA SFT v2 dataset."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from common.retrieval.evidence import normalize_evidence_text
from common.retrieval.markdown_bm25 import extract_search_query, tokenize
from common.retrieval.qa_sft import (
    grounded_points_response,
    observation_with_guidance,
)
from common.retrieval.qa_target_rebuild import question_fingerprint

OBJECTIVE_TYPES = ("single", "multiple", "bool")
SPLITS = ("train", "validation", "rl_holdout")
_TRUSTED_CATEGORIES = {"answer-bearing", "reference"}
_EXPECTED = re.compile(r"^\s*\[(\w+)\]\s*(.*)", re.DOTALL)


def question_keypoint_leak(
    search_query: str,
    original_question: str,
    keypoints: Sequence[Sequence[str]],
) -> bool:
    """Return whether a search query introduces an answer absent from the question."""
    normalized_query = normalize_evidence_text(search_query)
    normalized_question = normalize_evidence_text(original_question)
    return any(
        alternative in normalized_query and alternative not in normalized_question
        for alternatives in keypoints
        for alternative in alternatives
        if alternative
    )


def grounded_answer_term_leak(
    search_query: str,
    visible_context: str,
    points: Sequence[Mapping[str, Any]],
) -> bool:
    """Detect grounded answer terms introduced before they are visible."""
    normalized_query = normalize_evidence_text(search_query)
    normalized_context = normalize_evidence_text(visible_context)
    query_terms = {term for term in tokenize(search_query) if len(term) >= 2}
    context_terms = {term for term in tokenize(visible_context) if len(term) >= 2}
    answer_terms: set[str] = set()
    for point in points:
        statement = str(point.get("statement", ""))
        quote = str(point.get("quote", ""))
        for value in (statement, quote):
            normalized = normalize_evidence_text(value)
            if normalized and normalized in normalized_query and normalized not in normalized_context:
                return True
        answer_terms.update(
            term
            for term in set(tokenize(statement)) & set(tokenize(quote))
            if len(term) >= 2
        )
    return bool((query_terms & answer_terms) - context_terms)


def open_answer_leak_points(expected: str) -> list[dict[str, str]]:
    """Preserve raw open-answer terms for search-query leakage checks."""
    match = _EXPECTED.match(str(expected))
    if not match or match.group(1).lower() not in {"fill", "short"}:
        return []
    points = []
    for raw_point in match.group(2).split("|||"):
        for alternative in re.split(r"[/／]", raw_point):
            value = alternative.strip()
            if value:
                points.append({"statement": value, "quote": value})
    return points


def source_question_answer_term_leak(
    question: str,
    points: Sequence[Mapping[str, Any]],
) -> bool:
    """Reject prompts that expose a full answer or multiple grounded answer terms."""
    normalized_question = normalize_evidence_text(question)
    question_terms = {term for term in tokenize(question) if len(term) >= 2}
    for point in points:
        value = str(point.get("statement", ""))
        normalized_value = normalize_evidence_text(value)
        if normalized_value and normalized_value in normalized_question:
            return True
        overlap = question_terms & {
            term for term in tokenize(value) if len(term) >= 2
        }
        if len(overlap) >= 2 or any(len(term) >= 4 for term in overlap):
            return True
    return False


def rendered_observation_from_records(
    results: Sequence[Mapping[str, Any]],
    *,
    max_chars: int = 1800,
) -> str:
    """Reconstruct the exact bounded observation from audited ranked records."""
    blocks = ["[检索结果]"]
    for expected_rank, result in enumerate(results, start=1):
        if int(result.get("rank", -1)) != expected_rank:
            raise ValueError("ranked results must be contiguous and ordered")
        heading = f" · {result.get('heading')}" if result.get("heading") else ""
        display_score = result.get("raw_score")
        if display_score is None:
            display_score = result.get("score")
        if display_score is None:
            raise ValueError("ranked result is missing its display score")
        blocks.append(
            f"{expected_rank}. 来源：{result.get('source', '')}{heading}\n"
            f"相关度：{float(display_score):.2f}\n"
            f"{result.get('text', '')}"
        )
    return "\n\n".join(blocks)[:max_chars]


def short_target_issues(
    record: Mapping[str, Any],
    *,
    expected_initial_prompt: str | None = None,
) -> list[str]:
    """Validate a machine-rebuilt short target before it can enter SFT v2."""
    issues: list[str] = []
    if record.get("question_type") != "short":
        issues.append("wrong_question_type")
    if record.get("machine_verified") is not True:
        issues.append("not_machine_verified")
    if record.get("split") not in SPLITS:
        issues.append("invalid_split")
    fingerprint = str(record.get("question_fingerprint", ""))
    query = str(record.get("query", "")).strip()
    if not fingerprint:
        issues.append("missing_question_fingerprint")
    if not query:
        issues.append("missing_query")
    elif fingerprint != question_fingerprint(query):
        issues.append("question_fingerprint_mismatch")

    points = record.get("answer_points")
    if not isinstance(points, list) or not 2 <= len(points) <= 6:
        issues.append("invalid_answer_point_count")
        points = []
    hops = record.get("search_hops")
    if not isinstance(hops, list) or len(hops) != int(record.get("search_turns", -1)):
        issues.append("search_hop_count_mismatch")
        hops = []

    result_by_hop_rank: dict[tuple[int, int], Mapping[str, Any]] = {}
    observation_by_hop: dict[int, str] = {}
    results_by_hop: dict[int, list[Mapping[str, Any]]] = {}
    query_by_hop: dict[int, str] = {}
    recorded_hits_by_hop: dict[int, set[int]] = {}
    recorded_new_hits_by_hop: dict[int, set[int]] = {}
    for hop in hops:
        if not isinstance(hop, Mapping):
            issues.append("invalid_search_hop")
            continue
        try:
            hop_index = int(hop["hop"])
        except (KeyError, TypeError, ValueError):
            issues.append("invalid_search_hop")
            continue
        observation_by_hop[hop_index] = str(hop.get("observation", ""))
        query_by_hop[hop_index] = str(hop.get("model_search_query", "")).strip()
        recorded_hits_by_hop[hop_index] = {
            int(value)
            for value in hop.get("answer_point_hit_indexes", [])
            if isinstance(value, int)
        }
        recorded_new_hits_by_hop[hop_index] = {
            int(value)
            for value in hop.get("new_answer_point_hit_indexes", [])
            if isinstance(value, int)
        }
        ranked_results = hop.get("top_k_results", [])
        if not isinstance(ranked_results, list):
            issues.append("invalid_ranked_result")
            ranked_results = []
        results_by_hop[hop_index] = [
            result
            for result in ranked_results
            if isinstance(result, Mapping)
        ]
        for result in ranked_results:
            if not isinstance(result, Mapping):
                issues.append("invalid_ranked_result")
                continue
            try:
                rank = int(result["rank"])
            except (KeyError, TypeError, ValueError):
                issues.append("invalid_ranked_result")
                continue
            result_by_hop_rank[(hop_index, rank)] = result
        try:
            reconstructed = rendered_observation_from_records(
                results_by_hop[hop_index]
            )
        except (TypeError, ValueError):
            issues.append("observation_reconstruction_failure")
        else:
            if reconstructed != observation_by_hop[hop_index]:
                issues.append("observation_ranked_results_mismatch")

    statements = []
    point_quotes: dict[int, str] = {}
    for point in points:
        if not isinstance(point, Mapping):
            issues.append("invalid_answer_point")
            continue
        statement = str(point.get("statement", "")).strip()
        quote = str(point.get("quote", "")).strip()
        statements.append(statement)
        try:
            point_index = int(point["index"]) - 1
        except (KeyError, TypeError, ValueError):
            issues.append("invalid_answer_point")
            continue
        point_quotes[point_index] = quote
        supports = point.get("visible_supports")
        if not statement or not quote:
            issues.append("empty_statement_or_quote")
        if not isinstance(supports, list) or not supports:
            issues.append("missing_visible_support")
            continue
        point_supported = False
        for support in supports:
            if not isinstance(support, Mapping):
                continue
            try:
                key = (int(support["hop"]), int(support["rank"]))
            except (KeyError, TypeError, ValueError):
                continue
            result = result_by_hop_rank.get(key)
            if not result:
                continue
            category = str(result.get("quality_category", ""))
            if category not in _TRUSTED_CATEGORIES:
                continue
            if category != str(support.get("quality_category", "")):
                continue
            if quote not in str(result.get("text", "")):
                continue
            if quote not in observation_by_hop.get(key[0], ""):
                continue
            point_supported = True
            break
        if not point_supported:
            issues.append("unbound_visible_support")

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 4:
        issues.append("invalid_runtime_messages")
    else:
        roles = [message.get("role") for message in messages if isinstance(message, Mapping)]
        expected_roles = [
            "user",
            *[
                "assistant" if index % 2 else "environment"
                for index in range(1, len(messages))
            ],
        ]
        if len(roles) != len(messages) or roles != expected_roles:
            issues.append("invalid_runtime_roles")
        if any(
            not isinstance(message.get("content"), str)
            for message in messages
            if isinstance(message, Mapping)
        ):
            issues.append("non_string_message_content")
        if (
            expected_initial_prompt is not None
            and str(messages[0].get("content", "")) != expected_initial_prompt
        ):
            issues.append("initial_runtime_prompt_mismatch")
        final = (
            str(messages[-1].get("content", ""))
            if isinstance(messages[-1], Mapping)
            else ""
        )
        try:
            expected_final = grounded_points_response(statements)
        except ValueError:
            expected_final = ""
        if final != expected_final:
            issues.append("incomplete_final_answer")
        expected_message_count = 2 * len(hops) + 2
        if len(messages) != expected_message_count:
            issues.append("runtime_trace_length_mismatch")
        else:
            visible_context = str(record.get("query", ""))
            for hop_offset, hop_index in enumerate(range(1, len(hops) + 1)):
                assistant = messages[1 + 2 * hop_offset]
                environment = messages[2 + 2 * hop_offset]
                assistant_content = str(assistant.get("content", ""))
                message_query = extract_search_query(assistant_content)
                if (
                    message_query != query_by_hop.get(hop_index)
                    or assistant_content
                    != f"<search>{query_by_hop.get(hop_index, '')}</search>"
                ):
                    issues.append("runtime_search_query_mismatch")
                observation = observation_by_hop.get(hop_index, "")
                expected_observation = observation_with_guidance(
                    observation,
                    searches_remaining=1 if hop_index == 1 else 0,
                )
                if str(environment.get("content", "")) != expected_observation:
                    issues.append("runtime_observation_mismatch")
                if grounded_answer_term_leak(
                    query_by_hop.get(hop_index, ""),
                    visible_context,
                    points,
                ):
                    issues.append("search_query_answer_leak")
                visible_context += "\n" + observation
            if extract_search_query(final) is not None:
                issues.append("mixed_final_action")

    recomputed_hits_by_hop: dict[int, set[int]] = {}
    for hop_index in range(1, len(hops) + 1):
        hits = set()
        observation = observation_by_hop.get(hop_index, "")
        for point_index, quote in point_quotes.items():
            if not quote or quote not in observation:
                continue
            for (result_hop, _rank), result in result_by_hop_rank.items():
                if result_hop != hop_index:
                    continue
                if result.get("quality_category") not in _TRUSTED_CATEGORIES:
                    continue
                if quote in str(result.get("text", "")):
                    hits.add(point_index)
                    break
        recomputed_hits_by_hop[hop_index] = hits
        if hits != recorded_hits_by_hop.get(hop_index, set()):
            issues.append("recorded_hop_hits_mismatch")

    all_point_indexes = set(point_quotes)
    if len(hops) == 1:
        if recomputed_hits_by_hop.get(1, set()) != all_point_indexes:
            issues.append("incomplete_one_hop_evidence")
    elif len(hops) == 2:
        first_hits = recomputed_hits_by_hop.get(1, set())
        second_hits = recomputed_hits_by_hop.get(2, set())
        new_hits = second_hits - first_hits
        if first_hits == all_point_indexes:
            issues.append("first_hop_already_complete")
        if not new_hits:
            issues.append("nonincremental_second_hop")
        if first_hits | second_hits != all_point_indexes:
            issues.append("incomplete_two_hop_evidence")
        if recorded_new_hits_by_hop.get(2, set()) != new_hits:
            issues.append("recorded_new_hits_mismatch")
    else:
        issues.append("unsupported_search_turn_count")
    expected = "[short] " + " ||| ".join(statements)
    if record.get("expected_answer") != expected:
        issues.append("rebuilt_expected_answer_mismatch")

    return sorted(set(issues))


def _ordered(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    namespace: str,
) -> list[dict[str, Any]]:
    return sorted(
        (dict(record) for record in records),
        key=lambda record: hashlib.sha256(
            (
                f"{seed}:{namespace}:"
                f"{record.get('question_fingerprint', '')}:"
                f"{record.get('source_row_id', record.get('row_id', ''))}"
            ).encode("utf-8")
        ).hexdigest(),
    )


def select_balanced_objective_replay(
    records: Sequence[Mapping[str, Any]],
    *,
    open_train_count: int,
    target_fraction: float = 0.30,
    minimum_fraction: float = 0.25,
    maximum_fraction: float = 0.35,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select equal objective counts while keeping replay within the gate."""
    if open_train_count < 1:
        raise ValueError("open_train_count must be positive")
    if not 0 < minimum_fraction <= target_fraction <= maximum_fraction < 1:
        raise ValueError("invalid objective replay fractions")

    by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        question_type = str(record.get("question_type", ""))
        if question_type in OBJECTIVE_TYPES:
            by_type[question_type].append(record)
    if any(not by_type[question_type] for question_type in OBJECTIVE_TYPES):
        raise ValueError("objective replay is missing a question type")

    max_per_type = min(len(by_type[question_type]) for question_type in OBJECTIVE_TYPES)
    candidates = []
    for per_type in range(1, max_per_type + 1):
        objective_count = per_type * len(OBJECTIVE_TYPES)
        fraction = objective_count / (open_train_count + objective_count)
        if minimum_fraction <= fraction <= maximum_fraction:
            candidates.append((abs(fraction - target_fraction), per_type, fraction))
    if not candidates:
        raise ValueError("cannot satisfy objective replay fraction with balanced types")
    _distance, per_type, _fraction = min(candidates)

    selected = []
    for question_type in OBJECTIVE_TYPES:
        selected.extend(
            _ordered(
                by_type[question_type],
                seed=seed,
                namespace=f"objective-train:{question_type}",
            )[:per_type]
        )
    return selected


def select_objective_validation(
    records: Sequence[Mapping[str, Any]],
    *,
    per_type: int = 2,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select a deterministic, balanced objective validation replay."""
    if per_type < 1:
        raise ValueError("per_type must be positive")
    selected = []
    for question_type in OBJECTIVE_TYPES:
        group = [
            record
            for record in records
            if record.get("question_type") == question_type
        ]
        if len(group) < per_type:
            raise ValueError(f"not enough validation replay for {question_type}")
        selected.extend(
            _ordered(
                group,
                seed=seed,
                namespace=f"objective-validation:{question_type}",
            )[:per_type]
        )
    return selected


def assert_question_split_isolation(records: Sequence[Mapping[str, Any]]) -> None:
    """Require every source-question fingerprint to occur exactly once."""
    split_by_fingerprint: dict[str, str] = {}
    for record in records:
        fingerprint = str(record.get("question_fingerprint", ""))
        split = str(record.get("split", ""))
        if not fingerprint or split not in SPLITS:
            raise ValueError("every record needs a fingerprint and valid split")
        previous = split_by_fingerprint.get(fingerprint)
        if previous is not None:
            if previous == split:
                raise ValueError(f"duplicate question fingerprint in {split}: {fingerprint}")
            raise ValueError(
                f"question fingerprint overlaps {previous} and {split}: {fingerprint}"
            )
        split_by_fingerprint[fingerprint] = split


def objective_replay_fraction(records: Sequence[Mapping[str, Any]]) -> float:
    train = [record for record in records if record.get("split") == "train"]
    if not train:
        return 0.0
    objective_count = sum(
        record.get("question_type") in OBJECTIVE_TYPES
        for record in train
    )
    return objective_count / len(train)
