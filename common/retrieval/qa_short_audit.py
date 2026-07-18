"""Auditable short-answer label, evidence, trajectory, and reward diagnostics."""

from __future__ import annotations

import re
from collections.abc import Sequence
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any

from common.retrieval.evidence import normalize_evidence_text
from common.retrieval.markdown_bm25 import (
    SearchResult,
    build_retrieval_query,
    extract_search_query,
)
from common.retrieval.qa_sft import visible_retrieval_text
from common.rewards.qa_reward import (
    _alts_for_blank,
    _load_synonyms,
    _norm,
    extract_boxed,
    qa_rule_reward_fn,
)

_EXPECTED = re.compile(r"^\s*\[(\w+)\]\s*(.*)", re.DOTALL)
_MALFORMED_DELIMITER = re.compile(r"(?<!\|)\|{1,2}(?!\|)|\|{4,}")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_ALNUM = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_TEMPLATE_TERMS = (
    "依据检索证据",
    "根据检索证据",
    "依据证据",
    "根据证据",
    "最终答案",
    "答案是",
    "答案为",
    "分析后",
    "因此",
    "所以",
    "可知",
    "答案",
)

GENERIC_SHORT_TERMS = {
    "代码",
    "设备",
    "流程",
    "系统",
    "问题",
    "方法",
    "操作",
    "功能",
    "数据",
    "信息",
    "程序",
    "工艺",
    "原因",
    "结果",
    "内容",
    "步骤",
    "参数",
    "标准",
    "规范",
    "技术",
    "软件",
    "硬件",
    "模块",
    "平台",
    "工具",
    "产品",
    "材料",
    "文件",
    "文档",
    "测试",
    "方案",
    "code",
    "device",
    "equipment",
    "process",
    "system",
    "method",
    "operation",
    "procedure",
    "data",
    "program",
}


@lru_cache(maxsize=1)
def _synonym_groups() -> list[set[str]]:
    return _load_synonyms()


def _support_level(hits: set[int], point_count: int) -> str:
    if not point_count or not hits:
        return "none"
    return "full" if len(hits) == point_count else "partial"


def _is_very_short(value: str) -> bool:
    if not value:
        return True
    if _CJK.search(value):
        return len(value) <= 2
    compact = "".join(_ALNUM.findall(value))
    return len(compact) <= 3


def _overlap_ratio(value: str, context: str) -> float:
    if not value or not context:
        return 0.0
    if value in context:
        return 1.0
    match = SequenceMatcher(None, value, context).find_longest_match()
    return match.size / len(value)


def _near_duplicate(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    if min(len(left), len(right)) >= 3 and (left in right or right in left):
        return True
    return min(len(left), len(right)) >= 4 and SequenceMatcher(None, left, right).ratio() >= 0.88


def parse_short_gold(expected: str, *, query: str = "", bank: str = "") -> dict[str, Any]:
    """Parse short gold without hiding malformed delimiters or duplicate points."""
    match = _EXPECTED.match(str(expected))
    question_type = match.group(1).lower() if match else "unknown"
    answer = match.group(2).strip() if match else ""
    delimiter_issues: list[str] = []
    if question_type != "short":
        delimiter_issues.append("not_short_answer")
    if _MALFORMED_DELIMITER.search(answer):
        delimiter_issues.append("malformed_keypoint_delimiter")

    raw_points = answer.split("|||") if answer else []
    if any(not point.strip() for point in raw_points):
        delimiter_issues.append("empty_keypoint_segment")

    question_norm = normalize_evidence_text(query)
    bank_norm = normalize_evidence_text(bank)
    points: list[dict[str, Any]] = []
    for index, raw_point in enumerate(raw_points):
        raw = raw_point.strip()
        alternatives = [part.strip() for part in re.split(r"[/／]", raw) if part.strip()]
        normalized_alternatives = list(
            dict.fromkeys(
                normalize_evidence_text(alternative)
                for alternative in alternatives
                if normalize_evidence_text(alternative)
            )
        )
        point_issues: list[str] = []
        if not normalized_alternatives:
            point_issues.append("empty_keypoint")
        generic = bool(normalized_alternatives) and all(
            alternative in GENERIC_SHORT_TERMS for alternative in normalized_alternatives
        )
        very_short = bool(normalized_alternatives) and all(
            _is_very_short(alternative) for alternative in normalized_alternatives
        )
        overlap = max(
            (_overlap_ratio(alternative, question_norm) for alternative in normalized_alternatives),
            default=0.0,
        )
        topic_answer = bool(normalized_alternatives) and all(
            alternative in GENERIC_SHORT_TERMS
            or (
                bank_norm
                and len(alternative) >= 2
                and (alternative == bank_norm or (len(alternative) <= 8 and alternative in bank_norm))
            )
            for alternative in normalized_alternatives
        )
        if generic:
            point_issues.append("generic_keypoint")
        if very_short:
            point_issues.append("very_short_keypoint")
        if overlap >= 0.8:
            point_issues.append("question_high_overlap")
        if topic_answer:
            point_issues.append("topic_as_answer")
        points.append(
            {
                "index": index,
                "raw": raw,
                "alternatives": alternatives,
                "normalized_alternatives": normalized_alternatives,
                "question_overlap_ratio": overlap,
                "issues": point_issues,
            }
        )

    duplicate_pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(points):
        for right in points[left_index + 1 :]:
            best_pair = None
            for left_value in left["normalized_alternatives"]:
                for right_value in right["normalized_alternatives"]:
                    if _near_duplicate(left_value, right_value):
                        best_pair = {
                            "left_index": left["index"],
                            "right_index": right["index"],
                            "left": left_value,
                            "right": right_value,
                        }
                        break
                if best_pair:
                    break
            if best_pair:
                duplicate_pairs.append(best_pair)

    issue_codes = list(delimiter_issues)
    if len(points) == 1:
        issue_codes.append("singleton_keypoint")
    if not 2 <= len(points) <= 6:
        issue_codes.append("target_point_count_out_of_range")
    if duplicate_pairs:
        issue_codes.append("near_duplicate_keypoints")
    issue_codes.extend(
        f"{issue}:{point['index']}"
        for point in points
        for issue in point["issues"]
    )

    overlap_points = sum("question_high_overlap" in point["issues"] for point in points)
    defect_reasons = list(delimiter_issues)
    if not points:
        defect_reasons.append("empty_short_keypoints")
    if len(points) > 6:
        defect_reasons.append("too_many_keypoints")
    if duplicate_pairs:
        defect_reasons.append("near_duplicate_keypoints")
    if len(points) == 1:
        singleton_issues = set(points[0]["issues"])
        for issue in ("generic_keypoint", "very_short_keypoint", "question_high_overlap", "topic_as_answer"):
            if issue in singleton_issues:
                defect_reasons.append(f"singleton_{issue}")
    elif points and overlap_points / len(points) >= 0.5:
        defect_reasons.append("majority_keypoints_overlap_question")
    if points and all(
        "topic_as_answer" in point["issues"]
        for point in points
    ):
        defect_reasons.append("all_keypoints_are_topics")

    defective_indexes = sorted(
        point["index"]
        for point in points
        if {
            "generic_keypoint",
            "very_short_keypoint",
            "question_high_overlap",
            "topic_as_answer",
        }
        & set(point["issues"])
    )
    return {
        "question_type": question_type,
        "raw_answer": answer,
        "full_gold_keypoints": points,
        "keypoint_count": len(points),
        "label_issue_codes": list(dict.fromkeys(issue_codes)),
        "label_defect_reasons": list(dict.fromkeys(defect_reasons)),
        "duplicate_keypoint_pairs": duplicate_pairs,
        "defective_keypoint_indexes": defective_indexes,
    }


def _result_matches(result: SearchResult, points: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence = normalize_evidence_text(result.text)
    matches = []
    for point in points:
        matched = [
            alternative
            for alternative in point["normalized_alternatives"]
            if alternative and alternative in evidence
        ]
        if matched:
            matches.append(
                {
                    "keypoint_index": int(point["index"]),
                    "matched_alternatives": matched,
                }
            )
    return matches


def audit_search_results(
    results: Sequence[SearchResult],
    parsed_gold: dict[str, Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    """Preserve Top-k text and distinguish literal hits from trusted evidence."""
    points = parsed_gold["full_gold_keypoints"]
    defective_indexes = set(parsed_gold["defective_keypoint_indexes"])
    result_records = []
    literal_hits: set[int] = set()
    trusted_hits: set[int] = set()
    evidence_map: dict[int, list[dict[str, Any]]] = {
        int(point["index"]): [] for point in points
    }
    for rank, result in enumerate(results[:top_k], start=1):
        eligible = result.quality_category not in {"question-only", "noise"}
        matches = _result_matches(result, points) if eligible else []
        hit_indexes = {int(match["keypoint_index"]) for match in matches}
        literal_hits.update(hit_indexes)
        trusted_hits.update(hit_indexes - defective_indexes)
        record = {
            "rank": rank,
            "source": result.source,
            "heading": result.heading,
            "quality_category": result.quality_category,
            "quality_weight": float(result.quality_weight),
            "raw_score": float(result.raw_score or 0.0),
            "eligible_answer_evidence": eligible,
            "keypoint_matches": matches,
            "text": result.text,
        }
        result_records.append(record)
        for hit_index in hit_indexes:
            evidence_map[hit_index].append(
                {
                    "rank": rank,
                    "source": result.source,
                    "heading": result.heading,
                    "quality_category": result.quality_category,
                    "text": result.text,
                }
            )
    point_count = int(parsed_gold["keypoint_count"])
    return {
        "top_k": top_k,
        "top_k_results": result_records,
        "literal_hit_indexes": sorted(literal_hits),
        "trusted_hit_indexes": sorted(trusted_hits),
        "literal_support_level": _support_level(literal_hits, point_count),
        "trusted_support_level": _support_level(trusted_hits, point_count),
        "answer_bearing_evidence_by_keypoint": [
            {
                "keypoint_index": point_index,
                "evidence": evidence_map[point_index],
            }
            for point_index in sorted(evidence_map)
        ],
    }


def rule_short_match_details(expected: str, completion: str) -> dict[str, Any]:
    """Explain the unchanged official rule reward without replacing it."""
    match = _EXPECTED.match(str(expected))
    gold = match.group(2).strip() if match and match.group(1).lower() == "short" else ""
    keywords = [keyword for keyword in gold.split("|||") if keyword.strip()]
    boxed = extract_boxed(str(completion))
    answer_norm = _norm(boxed or "") + "|" + _norm(completion)
    matched = []
    for index, keyword in enumerate(keywords):
        alternatives = sorted(_alts_for_blank(keyword, _synonym_groups()))
        matched_alternatives = [
            alternative
            for alternative in alternatives
            if alternative and alternative in answer_norm
        ]
        if matched_alternatives:
            matched.append(
                {
                    "keypoint_index": index,
                    "raw_keypoint": keyword,
                    "matched_alternatives": matched_alternatives,
                }
            )
    reward = float(qa_rule_reward_fn([""], [str(completion)], [str(expected)])[0])
    return {
        "boxed": boxed,
        "official_rule_reward": reward,
        "rule_matched_keypoints": matched,
        "rule_matched_keypoint_indexes": [
            int(item["keypoint_index"]) for item in matched
        ],
    }


def is_keyword_only_completion(expected: str, completion: str) -> bool:
    boxed = extract_boxed(str(completion))
    if boxed is None:
        return False
    details = rule_short_match_details(expected, completion)
    if details["official_rule_reward"] < 1.0:
        return False
    residual = _norm(completion)
    for term in _TEMPLATE_TERMS:
        residual = residual.replace(_norm(term), "")
    match = _EXPECTED.match(str(expected))
    gold = match.group(2).strip() if match else ""
    alternatives = []
    for keyword in gold.split("|||"):
        alternatives.extend(_alts_for_blank(keyword, _synonym_groups()))
    for alternative in sorted(alternatives, key=len, reverse=True):
        if alternative:
            residual = residual.replace(alternative, "")
    residual = residual.replace("boxed", "")
    return len(residual) < 8


def reward_attack_audit(query: str, expected: str) -> list[dict[str, Any]]:
    parsed = parse_short_gold(expected, query=query)
    canonical_points = [
        point["alternatives"][0]
        for point in parsed["full_gold_keypoints"]
        if point["alternatives"]
    ]
    all_alternatives = [
        alternative
        for point in parsed["full_gold_keypoints"]
        for alternative in point["alternatives"]
    ]
    attacks = {
        "keyword_stuffing": rf"\boxed{{{'; '.join(canonical_points)}}}",
        "shortest_keypoint": (
            rf"\boxed{{{min(all_alternatives, key=lambda item: len(normalize_evidence_text(item)))}}}"
            if all_alternatives
            else r"\boxed{}"
        ),
        "question_copy": rf"\boxed{{{query}}}",
    }
    if len(canonical_points) == 1:
        attacks["bare_singleton"] = rf"\boxed{{{canonical_points[0]}}}"
    return [
        {
            "attack": name,
            "completion": completion,
            **rule_short_match_details(expected, completion),
        }
        for name, completion in attacks.items()
    ]


def audit_short_label(
    *,
    query: str,
    expected: str,
    bank: str,
    results: Sequence[SearchResult],
    top_k: int = 20,
) -> dict[str, Any]:
    parsed = parse_short_gold(expected, query=query, bank=bank)
    evidence = audit_search_results(results, parsed, top_k=top_k)
    static_defect_reasons = list(parsed["label_defect_reasons"])
    evidence_mapping_failure = bool(
        parsed["keypoint_count"]
        and not evidence["literal_hit_indexes"]
    )
    parsed["static_label_defect_reasons"] = static_defect_reasons
    parsed["label_defect_reasons"] = static_defect_reasons
    parsed["label_defect"] = bool(static_defect_reasons)
    attacks = reward_attack_audit(query, expected)
    return {
        **parsed,
        "evidence": evidence,
        "support_level": evidence["trusted_support_level"],
        "evidence_mapping_failure": evidence_mapping_failure,
        "evidence_issue_codes": (
            ["no_answer_bearing_evidence_mapping"]
            if evidence_mapping_failure
            else []
        ),
        "reward_attacks": attacks,
        "attack_full_reward": any(
            float(attack["official_rule_reward"]) >= 1.0
            for attack in attacks
        ),
        "false_perfect_attack": any(
            float(attack["official_rule_reward"]) >= 1.0
            and (
                parsed["label_defect"]
                or attack["attack"] in {"question_copy", "bare_singleton", "keyword_stuffing"}
            )
            for attack in attacks
        ),
    }


def _visible_hits(
    text: str,
    parsed_gold: dict[str, Any],
) -> tuple[set[int], set[int]]:
    visible = visible_retrieval_text(text)
    for marker in (
        "\n\n还可检索",
        "\n\n检索次数已用完",
        "\n[检索反馈]",
        "\n[检索约束]",
    ):
        visible = visible.split(marker, 1)[0]
    evidence = normalize_evidence_text(visible)
    literal = {
        int(point["index"])
        for point in parsed_gold["full_gold_keypoints"]
        if any(
            alternative and alternative in evidence
            for alternative in point["normalized_alternatives"]
        )
    }
    trusted = literal - set(parsed_gold["defective_keypoint_indexes"])
    return literal, trusted


def _query_leaks(
    search_query: str,
    original_query: str,
    parsed_gold: dict[str, Any],
) -> list[dict[str, Any]]:
    normalized_query = normalize_evidence_text(search_query)
    normalized_original = normalize_evidence_text(original_query)
    leaks = []
    for point in parsed_gold["full_gold_keypoints"]:
        matched = [
            alternative
            for alternative in point["normalized_alternatives"]
            if alternative
            and alternative in normalized_query
            and alternative not in normalized_original
        ]
        if matched:
            leaks.append(
                {
                    "keypoint_index": int(point["index"]),
                    "matched_alternatives": matched,
                }
            )
    return leaks


def _protocol_audit(messages: Sequence[dict[str, Any]]) -> dict[str, Any]:
    assistant_responses = [
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "assistant"
    ]
    search_queries = []
    issues = []
    final_completion = None
    for response in assistant_responses:
        search_query = extract_search_query(response)
        boxed = extract_boxed(response)
        if search_query is not None:
            if not search_query:
                issues.append("empty_search_query")
            else:
                search_queries.append(search_query)
        if boxed is not None:
            final_completion = response
        if search_query is not None and boxed is not None:
            issues.append("search_and_answer_in_same_turn")
        if search_query is None and boxed is None:
            issues.append("invalid_assistant_action")
    if not assistant_responses:
        issues.append("missing_assistant_response")
    if final_completion is None:
        issues.append("missing_boxed_completion")
    if len(search_queries) > 2:
        issues.append("too_many_searches")
    normalized_queries = [normalize_evidence_text(query) for query in search_queries]
    if len(set(normalized_queries)) != len(normalized_queries):
        issues.append("duplicate_search_query")
    return {
        "assistant_responses": assistant_responses,
        "search_queries": search_queries,
        "model_completion": final_completion,
        "protocol_issues": list(dict.fromkeys(issues)),
    }


def audit_short_trace(
    *,
    trace_source: str,
    source_row_id: int,
    query: str,
    expected: str,
    bank: str,
    messages: Sequence[dict[str, Any]],
    label_audit: dict[str, Any],
    index,
    logged_reward: float | None = None,
    sample_number: int | None = None,
    checkpoint_step: int | None = None,
    top_k: int = 20,
) -> dict[str, Any]:
    """Audit one real or supervised trajectory and rerun each search against BM25."""
    protocol = _protocol_audit(messages)
    parsed_gold = {
        key: label_audit[key]
        for key in (
            "full_gold_keypoints",
            "keypoint_count",
            "defective_keypoint_indexes",
        )
    }
    search_hops = []
    cumulative_literal: set[int] = set()
    cumulative_trusted: set[int] = set()
    assistant_positions = [
        index_position
        for index_position, message in enumerate(messages)
        if message.get("role") == "assistant"
    ]
    search_position = 0
    for message_index in assistant_positions:
        response = str(messages[message_index].get("content", ""))
        model_search_query = extract_search_query(response)
        if model_search_query is None or not model_search_query:
            continue
        search_position += 1
        observation = ""
        if message_index + 1 < len(messages) and messages[message_index + 1].get("role") == "environment":
            observation = str(messages[message_index + 1].get("content", ""))
        retrieval_query = build_retrieval_query(model_search_query, query, bank)
        results = index.search(
            retrieval_query,
            top_k=top_k,
            candidate_k=50,
            quality_rerank=True,
        )
        result_audit = audit_search_results(results, parsed_gold, top_k=top_k)
        displayed_literal, displayed_trusted = _visible_hits(observation, parsed_gold)
        eligible_display_audit = audit_search_results(
            results[:4],
            parsed_gold,
            top_k=min(4, len(results)),
        )
        eligible_display_hits = set(eligible_display_audit["literal_hit_indexes"])
        displayed_literal.intersection_update(eligible_display_hits)
        displayed_trusted.intersection_update(eligible_display_hits)
        new_literal = displayed_literal - cumulative_literal
        new_trusted = displayed_trusted - cumulative_trusted
        cumulative_literal.update(displayed_literal)
        cumulative_trusted.update(displayed_trusted)
        search_hops.append(
            {
                "hop": search_position,
                "model_search_query": model_search_query,
                "retrieval_query": retrieval_query,
                "query_answer_leaks": _query_leaks(model_search_query, query, parsed_gold),
                "observation": observation,
                "displayed_literal_hit_indexes": sorted(displayed_literal),
                "displayed_trusted_hit_indexes": sorted(displayed_trusted),
                "new_literal_hit_indexes": sorted(new_literal),
                "new_trusted_hit_indexes": sorted(new_trusted),
                "cumulative_literal_hit_indexes": sorted(cumulative_literal),
                "cumulative_trusted_hit_indexes": sorted(cumulative_trusted),
                **result_audit,
            }
        )

    point_count = int(parsed_gold["keypoint_count"])
    visible_support = _support_level(cumulative_trusted, point_count)
    model_completion = protocol["model_completion"]
    if model_completion is None:
        rule_details = {
            "boxed": None,
            "official_rule_reward": -0.5,
            "rule_matched_keypoints": [],
            "rule_matched_keypoint_indexes": [],
        }
        keyword_only = False
    else:
        rule_details = rule_short_match_details(expected, model_completion)
        keyword_only = is_keyword_only_completion(expected, model_completion)
    official_reward = float(rule_details["official_rule_reward"])
    logged_matches = (
        None
        if logged_reward is None
        else abs(float(logged_reward) - official_reward) <= 1e-6
    )
    false_perfect = official_reward >= 1.0 and (
        bool(label_audit["label_defect"])
        or visible_support != "full"
        or keyword_only
    )

    failure_labels = []
    if label_audit["label_defect"]:
        failure_labels.append("label_defect")
    if protocol["protocol_issues"]:
        failure_labels.append("protocol_failure")
    if visible_support == "none":
        failure_labels.append("unsupported_retrieval")
    elif visible_support == "partial":
        failure_labels.append("partial_evidence")
    elif official_reward < 1.0 or keyword_only:
        failure_labels.append("synthesis_failure")

    if label_audit["label_defect"]:
        primary = "label_defect"
    elif protocol["protocol_issues"]:
        primary = "protocol_failure"
    elif visible_support == "none":
        primary = "unsupported_retrieval"
    elif visible_support == "partial":
        primary = "partial_evidence"
    elif official_reward < 1.0 or keyword_only:
        primary = "synthesis_failure"
    else:
        primary = "success"

    return {
        "trace_source": trace_source,
        "source_row_id": int(source_row_id),
        "sample_number": sample_number,
        "checkpoint_step": checkpoint_step,
        "query": query,
        "expected_answer": expected,
        "full_gold_keypoints": label_audit["full_gold_keypoints"],
        "bank": bank,
        "search_query": protocol["search_queries"][0] if protocol["search_queries"] else None,
        "search_hops": search_hops,
        "model_completion": model_completion,
        **rule_details,
        "logged_reward": logged_reward,
        "logged_reward_matches_official_rule": logged_matches,
        "keyword_only_completion": keyword_only,
        "protocol_issues": protocol["protocol_issues"],
        "assistant_responses": protocol["assistant_responses"],
        "visible_literal_hit_indexes": sorted(cumulative_literal),
        "visible_trusted_hit_indexes": sorted(cumulative_trusted),
        "evidence_support_level": visible_support,
        "label_defect": bool(label_audit["label_defect"]),
        "label_defect_reasons": list(label_audit["label_defect_reasons"]),
        "failure_labels": failure_labels,
        "primary_attribution": primary,
        "false_perfect": false_perfect,
        "reward_hacking": false_perfect,
    }
