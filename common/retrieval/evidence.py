"""Deterministic answer-keypoint coverage helpers for retrieval diagnostics."""

from __future__ import annotations

import re
import unicodedata
from typing import Sequence

from common.retrieval.markdown_bm25 import SearchResult

_EXPECTED = re.compile(r"^\s*\[(\w+)\]\s*(.*)", re.DOTALL)
_PUNCT = re.compile(r"[\s，,。．.、；;：:！!？?\"'`（）()【】\[\]{}<>]+")


def normalize_evidence_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    normalized = normalized.replace("μ", "u").replace("µ", "u")
    return _PUNCT.sub("", normalized)


def expected_keypoints(expected: str) -> tuple[str, list[list[str]]]:
    match = _EXPECTED.match(str(expected))
    if not match:
        return "unknown", []
    question_type, answer = match.group(1).lower(), match.group(2)
    if question_type not in {"fill", "short"}:
        return question_type, []
    keypoints = []
    for raw_point in answer.split("|||"):
        parts = re.split(r"[/／]", raw_point)
        alternatives = {normalize_evidence_text(part) for part in parts}
        if len(parts) == 1:
            alternatives.add(normalize_evidence_text(raw_point))
        alternatives.discard("")
        if alternatives:
            keypoints.append(sorted(alternatives, key=len, reverse=True))
    return question_type, keypoints


def fragile_keypoint_indexes(
    keypoints: Sequence[Sequence[str]],
) -> set[int]:
    """Flag one-character or one-digit keypoints unsafe for substring evidence."""
    return {
        index
        for index, alternatives in enumerate(keypoints)
        if max((len(alternative) for alternative in alternatives), default=0) <= 1
    }


def evidence_coverage(
    results: Sequence[SearchResult],
    keypoints: Sequence[Sequence[str]],
    *,
    top_k: int,
) -> float:
    if not keypoints:
        return 0.0
    return len(evidence_keypoint_hits(results, keypoints, top_k=top_k)) / len(keypoints)


def text_keypoint_hits(
    text: str,
    keypoints: Sequence[Sequence[str]],
) -> set[int]:
    """Return gold keypoint indexes visible in a rendered text observation."""
    if not keypoints:
        return set()
    evidence = normalize_evidence_text(text)
    return {
        index
        for index, alternatives in enumerate(keypoints)
        if any(alternative in evidence for alternative in alternatives)
    }


def evidence_keypoint_hits(
    results: Sequence[SearchResult],
    keypoints: Sequence[Sequence[str]],
    *,
    top_k: int,
) -> set[int]:
    """Return gold keypoint indexes supported by answer-bearing retrieval results."""
    if not keypoints:
        return set()
    searchable = [
        result
        for result in results[:top_k]
        if result.quality_category not in {"question-only", "noise"}
    ]
    return text_keypoint_hits(
        "\n".join(result.text for result in searchable),
        keypoints,
    )


def visible_evidence_keypoint_hits(
    results: Sequence[SearchResult],
    visible_snippets: Sequence[str],
    keypoints: Sequence[Sequence[str]],
) -> set[int]:
    """Return keypoint hits visible within the same trusted ranked result."""
    hits: set[int] = set()
    for result, snippet in zip(results, visible_snippets, strict=True):
        if result.quality_category not in {"question-only", "noise"}:
            hits.update(text_keypoint_hits(snippet, keypoints))
    return hits
