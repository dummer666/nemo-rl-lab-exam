"""Strict helpers for rebuilding short-answer targets from visible evidence."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from common.retrieval.evidence import normalize_evidence_text
from common.retrieval.markdown_bm25 import SearchResult, question_context, tokenize

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_TRUNCATION_MARKER = re.compile(r"(?:\.{2,}|。{2,}|…|⋯)")
_INLINE_METADATA = re.compile(
    r"\*?\[Image OCR\].*?\[End OCR\]\*?|<!--.*?-->",
    re.IGNORECASE | re.DOTALL,
)
_SPAN_BOUNDARY = re.compile(r"[^。！？!?；;\n]+(?:[。！？!?；;]+|$)")
_ENTITY = re.compile(
    r"[A-Za-z][A-Za-z0-9+._/-]*|\d+(?:\.\d+)?(?:%|℃|°C|V|A|Pa|nm|um|μm)?",
    re.IGNORECASE,
)
_NON_TEXT_TASKS = (
    ("screenshot", re.compile(r"截图|截屏|screen\s*shot", re.IGNORECASE)),
    ("upload", re.compile(r"上传|upload", re.IGNORECASE)),
    ("draw", re.compile(r"画出|绘制|补充.*图|示意图|乌龟图", re.IGNORECASE)),
    (
        "write_code",
        re.compile(r"编写.*代码|写出.*代码|代码实现|实现以下.*(?:程序|函数)", re.IGNORECASE),
    ),
)
_GENERIC_POINT_NORMALIZED = {
    normalize_evidence_text(value)
    for value in (
        "代码",
        "设备",
        "流程",
        "操作",
        "方法",
        "步骤",
        "参数",
        "事件类型",
        "略",
        "无",
        "NA",
        "N/A",
        "/",
    )
}
_GENERIC_TERMS = {
    term
    for term in tokenize(
        "答案 要点 内容 相关 主要 可以 需要 进行 包括 通过 使用 方面 情况 "
        "answer point content related main include use"
    )
    if len(term) >= 2
}


def question_fingerprint(query: str) -> str:
    """Return a template-insensitive source-question fingerprint."""
    normalized = normalize_evidence_text(question_context(query))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def non_text_task_reason(query: str) -> str | None:
    """Identify tasks whose required artifact cannot be represented as text QA."""
    context = question_context(query)
    for reason, pattern in _NON_TEXT_TASKS:
        if pattern.search(context):
            return reason
    return None


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse one model-produced JSON object without accepting prose fallbacks."""
    candidate = str(text).strip()
    if "</think>" in candidate:
        candidate = candidate.rsplit("</think>", 1)[-1].strip()
    fenced = _JSON_FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escaped = False
        end = None
        for index, character in enumerate(candidate[start:], start=start):
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            return None
        try:
            value = json.loads(candidate[start:end])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _content_terms(text: str) -> set[str]:
    return {
        term
        for term in tokenize(text)
        if len(term) >= 2 and term not in _GENERIC_TERMS
    }


def _entity_set(text: str) -> set[str]:
    return {match.group(0).lower() for match in _ENTITY.finditer(str(text))}


def align_literal_quote(evidence_text: str, generated_quote: str) -> str | None:
    """Return one exact source span when punctuation-only drift is unambiguous."""
    source = str(evidence_text)
    quote = str(generated_quote).strip()
    if not quote or _TRUNCATION_MARKER.search(quote):
        return None
    if quote in source:
        return quote

    normalized_quote = normalize_evidence_text(quote)
    if not normalized_quote:
        return None
    normalized_chars = []
    source_indexes = []
    for source_index, character in enumerate(source):
        for normalized_character in normalize_evidence_text(character):
            normalized_chars.append(normalized_character)
            source_indexes.append(source_index)
    normalized_source = "".join(normalized_chars)
    start = normalized_source.find(normalized_quote)
    if start < 0 or normalized_source.find(normalized_quote, start + 1) >= 0:
        return None
    end = start + len(normalized_quote) - 1
    return source[source_indexes[start] : source_indexes[end] + 1]


def question_query_candidates(question: str, max_candidates: int = 8) -> list[str]:
    """Build leak-free BM25 query variants using only literal question clauses."""
    context = question_context(question)[:256].strip()
    if not context:
        return []
    candidates = [context]
    seen = {" ".join(context.lower().split())}
    clauses = re.split(r"[\n\r，,。；;：:、！？!?（）()]+", context)
    for clause in clauses:
        candidate = re.sub(
            r"^(?:请|简述|说明|分析|列举|比较|指出|回答)\s*",
            "",
            clause.strip(),
        ).strip()
        normalized = " ".join(candidate.lower().split())
        if len(normalize_evidence_text(candidate)) < 4 or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(candidate[:256])
        if len(candidates) >= max_candidates:
            break
    return candidates


def build_evidence_spans(
    evidence_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Pre-cut exact source spans so the teacher selects instead of transcribes."""
    spans: list[dict[str, Any]] = []
    seen_quotes = set()
    for evidence_id, evidence in evidence_by_id.items():
        text = str(evidence.get("text", ""))
        masked = list(text)
        for match in _INLINE_METADATA.finditer(text):
            for index in range(match.start(), match.end()):
                if masked[index] in "。！？!?；;\n":
                    masked[index] = " "
        for match in _SPAN_BOUNDARY.finditer("".join(masked)):
            start, end = match.span()
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if start >= end:
                continue
            raw_span = text[start:end]
            raw_chunks = [raw_span]
            if len(normalize_evidence_text(raw_span)) > 220:
                raw_chunks = []
                chunk_start = 0
                while chunk_start < len(raw_span):
                    chunk_end = min(len(raw_span), chunk_start + 220)
                    chunk = raw_span[chunk_start:chunk_end].strip()
                    if chunk:
                        raw_chunks.append(chunk)
                    if chunk_end >= len(raw_span):
                        break
                    chunk_start = max(chunk_start + 1, chunk_end - 40)
            for quote in raw_chunks:
                quote = quote.strip("…").strip()
                normalized_length = len(normalize_evidence_text(quote))
                if (
                    _TRUNCATION_MARKER.search(quote)
                    or not 12 <= normalized_length <= 220
                    or quote in seen_quotes
                ):
                    continue
                seen_quotes.add(quote)
                spans.append(
                    {
                        "span_id": f"S{len(spans) + 1:03d}",
                        "evidence_id": str(evidence_id),
                        "quote": quote,
                        "source": str(evidence.get("source", "")),
                        "heading": str(evidence.get("heading", "")),
                        "quality_category": str(
                            evidence.get("quality_category", "")
                        ),
                    }
                )
    return spans


def materialize_span_references(
    payload: Mapping[str, Any],
    span_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    """Replace teacher-selected span IDs with their immutable exact source text."""
    if payload.get("decision") != "answerable":
        return dict(payload), None
    raw_points = payload.get("answer_points")
    if not isinstance(raw_points, list):
        return dict(payload), None

    materialized_points = []
    for index, raw_point in enumerate(raw_points, start=1):
        if not isinstance(raw_point, Mapping):
            return None, f"point_{index}:invalid_shape"
        generated_span_id = str(raw_point.get("span_id", "")).strip()
        match = re.fullmatch(r"[Ss]0*(\d+)", generated_span_id)
        canonical_span_id = (
            f"S{int(match.group(1)):03d}"
            if match
            else generated_span_id
        )
        span = span_by_id.get(canonical_span_id)
        if span is None:
            return None, f"point_{index}:unknown_span"
        materialized_points.append(
            {
                **dict(raw_point),
                "span_id": canonical_span_id,
                "evidence_id": str(span["evidence_id"]),
                "quote": str(span["quote"]),
                "teacher_supplied_quote": raw_point.get("quote"),
            }
        )
    return {
        **dict(payload),
        "answer_points": materialized_points,
    }, None


def _canonical_evidence_id(
    evidence_id: str,
    evidence_by_id: Mapping[str, Mapping[str, Any]],
) -> str | None:
    if evidence_id in evidence_by_id:
        return evidence_id
    match = re.fullmatch(r"[Ee]0*(\d+)", evidence_id)
    if not match:
        return None
    canonical = f"E{int(match.group(1)):02d}"
    return canonical if canonical in evidence_by_id else None


def _point_validation_reason(
    statement: str,
    quote: str,
    *,
    question: str,
    evidence_text: str,
) -> str | None:
    normalized_statement = normalize_evidence_text(statement)
    normalized_quote = normalize_evidence_text(quote)
    if not 8 <= len(normalized_statement) <= 240:
        return "statement_length"
    if normalized_statement in _GENERIC_POINT_NORMALIZED:
        return "generic_statement"
    if any(
        reserved in statement
        for reserved in ("|||", r"\boxed", "<search", "</search>", "{", "}")
    ):
        return "reserved_statement_syntax"
    if not 12 <= len(normalized_quote) <= 220:
        return "quote_length"
    if "…" in quote or quote not in evidence_text:
        return "quote_not_exact"

    statement_terms = _content_terms(statement)
    quote_terms = _content_terms(quote)
    shared = statement_terms & quote_terms
    if not statement_terms or len(shared) < 2:
        return "insufficient_statement_quote_overlap"
    if len(shared) / len(statement_terms) < 0.35:
        return "low_statement_quote_overlap"

    question_terms = _content_terms(question)
    if not statement_terms - question_terms:
        return "question_restatement"

    quote_entities = _entity_set(quote) | _entity_set(question)
    unsupported_entities = _entity_set(statement) - quote_entities
    if unsupported_entities:
        return "unsupported_statement_entities"
    return None


def validate_generated_target(
    payload: Mapping[str, Any],
    *,
    question: str,
    evidence_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]] | None, str]:
    """Validate a teacher target against exact evidence and deterministic gates."""
    if payload.get("decision") != "answerable":
        return None, "teacher_rejected"
    raw_points = payload.get("answer_points")
    if not isinstance(raw_points, list) or not 2 <= len(raw_points) <= 6:
        return None, "point_count"

    points: list[dict[str, Any]] = []
    for index, raw_point in enumerate(raw_points, start=1):
        if not isinstance(raw_point, Mapping):
            return None, f"point_{index}:invalid_shape"
        statement = str(raw_point.get("statement", "")).strip()
        generated_evidence_id = str(raw_point.get("evidence_id", "")).strip()
        evidence_id = _canonical_evidence_id(
            generated_evidence_id,
            evidence_by_id,
        )
        generated_quote = str(raw_point.get("quote", "")).strip()
        if evidence_id is None:
            return None, f"point_{index}:unknown_evidence"
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:  # pragma: no cover - canonical lookup guarantees this
            raise AssertionError("canonical evidence ID is missing")
        if evidence.get("quality_category") in {"question-only", "noise"}:
            return None, f"point_{index}:untrusted_evidence"
        quote = align_literal_quote(
            str(evidence.get("text", "")),
            generated_quote,
        )
        if quote is None:
            return None, f"point_{index}:quote_not_exact"
        reason = _point_validation_reason(
            statement,
            quote,
            question=question,
            evidence_text=str(evidence.get("text", "")),
        )
        if reason:
            return None, f"point_{index}:{reason}"
        points.append(
            {
                "index": index,
                "statement": statement,
                "evidence_id": evidence_id,
                "quote": quote,
                "quote_alignment": (
                    "selected_span"
                    if raw_point.get("span_id")
                    else (
                        "exact"
                        if quote == generated_quote
                        else "punctuation_repaired"
                    )
                ),
                "generated_quote": (
                    raw_point.get("teacher_supplied_quote")
                    if raw_point.get("span_id")
                    else generated_quote
                ),
                "span_id": str(raw_point.get("span_id", "")) or None,
                "source": str(evidence.get("source", "")),
                "heading": str(evidence.get("heading", "")),
                "quality_category": str(evidence.get("quality_category", "")),
            }
        )

    term_sets = [_content_terms(point["statement"]) for point in points]
    for left_index, left in enumerate(term_sets):
        for right in term_sets[left_index + 1 :]:
            union = left | right
            if union and len(left & right) / len(union) >= 0.8:
                return None, "near_duplicate_points"
    return points, "accepted"


def verifier_accepts(payload: Mapping[str, Any], point_count: int) -> bool:
    """Require explicit support and relevance decisions for every answer point."""
    if payload.get("decision") != "accept" or payload.get("complete") is not True:
        return False
    checks = payload.get("point_checks")
    if not isinstance(checks, list) or len(checks) != point_count:
        return False
    seen = set()
    for check in checks:
        if not isinstance(check, Mapping):
            return False
        try:
            index = int(check.get("index"))
        except (TypeError, ValueError):
            return False
        if check.get("supported") is not True or check.get("relevant") is not True:
            return False
        seen.add(index)
    return seen == set(range(1, point_count + 1))


def evidence_quote_hits(text: str, points: Sequence[Mapping[str, Any]]) -> set[int]:
    """Return indexes of exact evidence quotes visible to the deployed agent."""
    return {
        int(point["index"]) - 1
        for point in points
        if str(point.get("quote", "")) in str(text)
    }


def trusted_visible_quote_hits(
    results: Sequence[SearchResult],
    visible_snippets: Sequence[str],
    points: Sequence[Mapping[str, Any]],
) -> set[int]:
    """Return quote hits visible within the same trusted ranked result."""
    hits: set[int] = set()
    for result, snippet in zip(results, visible_snippets, strict=True):
        if result.quality_category not in {"question-only", "noise"}:
            hits.update(evidence_quote_hits(snippet, points))
    return hits


def rebuilt_expected_answer(points: Sequence[Mapping[str, Any]]) -> str:
    statements = [str(point["statement"]).strip() for point in points]
    if not 2 <= len(statements) <= 6 or any(not statement for statement in statements):
        raise ValueError("rebuilt expected answer requires 2-6 complete statements")
    return "[short] " + " ||| ".join(statements)


def bind_visible_evidence(
    points: Sequence[Mapping[str, Any]],
    search_hops: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Bind each point to the exact ranked result the agent actually saw."""
    bound = []
    for point in points:
        quote = str(point.get("quote", ""))
        supports = []
        for hop in search_hops:
            observation = str(hop.get("observation", ""))
            if not quote or quote not in observation:
                continue
            for result in hop.get("top_k_results", []):
                if not isinstance(result, Mapping):
                    continue
                if result.get("quality_category") in {"question-only", "noise"}:
                    continue
                if quote not in str(result.get("text", "")):
                    continue
                supports.append(
                    {
                        "hop": int(hop["hop"]),
                        "rank": int(result["rank"]),
                        "source": str(result.get("source", "")),
                        "heading": str(result.get("heading", "")),
                        "quality_category": str(
                            result.get("quality_category", "")
                        ),
                    }
                )
        if not supports:
            raise ValueError(
                f"point {point.get('index')} has no exact visible source binding"
            )
        bound.append({**dict(point), "visible_supports": supports})
    return bound


def assign_group_splits(
    records: Sequence[Mapping[str, Any]],
    *,
    validation_fraction: float = 0.15,
    rl_fraction: float = 0.15,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Split unique source questions deterministically, stratified by hop count."""
    if min(validation_fraction, rl_fraction) < 0:
        raise ValueError("split fractions must be non-negative")
    if validation_fraction + rl_fraction >= 1:
        raise ValueError("validation and RL fractions must sum to less than one")

    fingerprints = [str(record["question_fingerprint"]) for record in records]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("source questions must be deduplicated before splitting")

    by_turns: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_turns.setdefault(int(record["search_turns"]), []).append(dict(record))

    assigned: list[dict[str, Any]] = []
    for turns, group in sorted(by_turns.items()):
        ordered = sorted(
            group,
            key=lambda record: hashlib.sha256(
                f"{seed}:{turns}:{record['question_fingerprint']}".encode("utf-8")
            ).hexdigest(),
        )
        validation_count = round(len(ordered) * validation_fraction)
        rl_count = round(len(ordered) * rl_fraction)
        if len(ordered) >= 7:
            validation_count = max(1, validation_count)
            rl_count = max(1, rl_count)
        while validation_count + rl_count >= len(ordered) and rl_count:
            rl_count -= 1
        while validation_count + rl_count >= len(ordered) and validation_count:
            validation_count -= 1
        for index, record in enumerate(ordered):
            if index < validation_count:
                split = "validation"
            elif index < validation_count + rl_count:
                split = "rl_holdout"
            else:
                split = "train"
            record["split"] = split
            assigned.append(record)
    return sorted(assigned, key=lambda record: int(record["source_row_id"]))
