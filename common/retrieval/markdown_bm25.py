"""Dependency-free BM25 retrieval over a directory of Markdown documents."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

_ALNUM = re.compile(r"[a-z0-9]+(?:[._+#/-][a-z0-9]+)*", re.IGNORECASE)
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
_QUESTION = re.compile(
    r"题目\s*[：:]\s*(.*?)(?=\n\s*选项\s*[：:]|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_SEARCH_OPEN = re.compile(r"<search(?:\s[^>]*)?>", re.IGNORECASE)
_SEARCH_CLOSE = re.compile(r"</search\s*>", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_ANSWER_MARKERS = re.compile(
    r"答案|答题解析|标准解答|参考解答|answer\s*key|answers?\b|solution",
    re.IGNORECASE,
)
_ANSWER_TEXT_MARKERS = re.compile(
    r"(?:正确|标准|参考)?答案\s*[：:]|(?:正确|标准|参考)?答案(?:是|为)",
    re.IGNORECASE,
)
_EXAM_MARKERS = re.compile(
    r"试题|试卷|考题|题库|考试|练习题|测验|(?:^|[\W_])(?:quiz|exam)(?:$|[\W_])",
    re.IGNORECASE,
)

DEFAULT_QUALITY_WEIGHTS = {
    "answer-bearing": 1.15,
    "reference": 1.0,
    "question-only": 0.45,
    "noise": 0.1,
}

_STOP_TERMS = {
    "一道",
    "下面",
    "作答",
    "分析",
    "最终",
    "答案",
    "字母",
    "选项",
    "题目",
    "填入",
    "多个",
    "所有",
    "please",
    "answer",
    "question",
}


@dataclass(frozen=True)
class _Chunk:
    source: str
    heading: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    """A ranked Markdown chunk."""

    source: str
    heading: str
    text: str
    score: float
    raw_score: float | None = None
    quality_category: str = "reference"
    quality_weight: float = 1.0


@dataclass(frozen=True)
class DocumentQuality:
    """Retrieval prior derived from document structure without deleting source data."""

    category: str
    weight: float


def classify_document_quality(
    source: str,
    heading: str,
    text: str,
    *,
    weights: Mapping[str, float] | None = None,
) -> DocumentQuality:
    """Classify answer-bearing, reference, question-only, and corrupted chunks."""
    quality_weights = {**DEFAULT_QUALITY_WEIGHTS, **dict(weights or {})}
    visible = _WHITESPACE.sub("", str(text))
    replacement_ratio = visible.count("\ufffd") / max(1, len(visible))
    if not visible or replacement_ratio >= 0.05:
        category = "noise"
    else:
        label = f"{source}\n{heading}"
        has_answer = bool(_ANSWER_MARKERS.search(label) or _ANSWER_TEXT_MARKERS.search(text))
        is_exam = bool(_EXAM_MARKERS.search(label))
        if has_answer:
            category = "answer-bearing"
        elif is_exam:
            category = "question-only"
        else:
            category = "reference"
    return DocumentQuality(category=category, weight=float(quality_weights[category]))


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text without an external segmenter."""
    lowered = str(text).lower()
    terms = [token for token in _ALNUM.findall(lowered) if len(token) > 1 and token not in _STOP_TERMS]
    for run in _CJK_RUN.findall(lowered):
        terms.extend(ch for ch in run if ch not in _STOP_TERMS)
        terms.extend(run[i : i + 2] for i in range(len(run) - 1) if run[i : i + 2] not in _STOP_TERMS)
        if 2 <= len(run) <= 6 and run not in _STOP_TERMS:
            terms.append(run)
    return terms


def extract_search_query(text: str, max_chars: int = 256) -> str | None:
    """Extract the last complete, or trailing incomplete, ``<search>`` action."""
    matches = list(_SEARCH_OPEN.finditer(str(text)))
    if not matches:
        return None
    start = matches[-1].end()
    close = _SEARCH_CLOSE.search(str(text), start)
    end = close.start() if close else len(str(text))
    query = _WHITESPACE.sub(" ", str(text)[start:end]).strip()
    return query[:max_chars] if query else ""


def question_context(query: str) -> str:
    """Keep the actual question and drop generic answer-format instructions."""
    match = _QUESTION.search(str(query))
    if match:
        return _WHITESPACE.sub(" ", match.group(1)).strip()
    return _WHITESPACE.sub(" ", str(query)).strip()


def build_retrieval_query(search_query: str, original_query: str, bank: str = "") -> str:
    """Combine the model query with stable question metadata for recall."""
    parts: list[str] = []
    for part in (search_query, question_context(original_query), bank):
        normalized = _WHITESPACE.sub(" ", str(part)).strip()
        if normalized and normalized not in parts:
            parts.append(normalized)
    return "\n".join(parts)


def _split_text(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        if end < len(text):
            lower_bound = start + chunk_chars // 2
            boundaries = [text.rfind(mark, lower_bound, end) for mark in ("\n", "。", "；", ";")]
            boundary = max(boundaries)
            if boundary >= lower_bound:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap_chars)
    return chunks


def _chunks_from_file(
    path: Path,
    root: Path,
    chunk_chars: int,
    overlap_chars: int,
) -> list[_Chunk]:
    encoded = path.read_bytes()
    raw = ""
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            raw = encoded.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if not raw:
        raw = encoded.decode("utf-8", errors="replace")
    source = path.relative_to(root).as_posix()
    current_heading = path.stem
    section_lines: list[str] = []
    sections: list[tuple[str, str]] = []

    def flush() -> None:
        text = "\n".join(section_lines).strip()
        if text:
            sections.append((current_heading, text))
        section_lines.clear()

    for line in raw.splitlines():
        heading = _HEADING.match(line)
        if heading:
            flush()
            current_heading = heading.group(1).strip()
        else:
            section_lines.append(line)
    flush()

    chunks: list[_Chunk] = []
    for heading, section in sections:
        for text in _split_text(section, chunk_chars, overlap_chars):
            chunks.append(_Chunk(source=source, heading=heading, text=text))
    return chunks


class MarkdownBM25Index:
    """In-memory sparse index built once per retrieval environment actor."""

    def __init__(
        self,
        chunks: list[_Chunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        quality_weights: Mapping[str, float] | None = None,
    ):
        if not chunks:
            raise ValueError("Cannot build a retrieval index without Markdown content")
        self._chunks = chunks
        self.k1 = float(k1)
        self.b = float(b)
        self.quality_weights = {**DEFAULT_QUALITY_WEIGHTS, **dict(quality_weights or {})}
        self._lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._qualities = [
            classify_document_quality(
                chunk.source,
                chunk.heading,
                chunk.text,
                weights=self.quality_weights,
            )
            for chunk in chunks
        ]

        for doc_id, chunk in enumerate(chunks):
            weighted = f"{chunk.source} {chunk.source} {chunk.heading} {chunk.heading} {chunk.text}"
            counts = Counter(tokenize(weighted))
            self._lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self._postings[term].append((doc_id, frequency))

        self._average_length = sum(self._lengths) / len(self._lengths)
        self._idf = {
            term: math.log(1.0 + (len(chunks) - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self._postings.items()
        }

    @property
    def num_documents(self) -> int:
        return len(self._chunks)

    @property
    def quality_category_counts(self) -> dict[str, int]:
        counts = Counter(quality.category for quality in self._qualities)
        return {category: counts.get(category, 0) for category in DEFAULT_QUALITY_WEIGHTS}

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        *,
        chunk_chars: int = 1200,
        overlap_chars: int = 160,
        k1: float = 1.5,
        b: float = 0.75,
        quality_weights: Mapping[str, float] | None = None,
    ) -> "MarkdownBM25Index":
        root_path = Path(root)
        if not root_path.is_dir():
            raise FileNotFoundError(f"Markdown document directory does not exist: {root_path}")
        if chunk_chars < 200:
            raise ValueError("chunk_chars must be at least 200")
        if overlap_chars < 0 or overlap_chars >= chunk_chars:
            raise ValueError("overlap_chars must be in [0, chunk_chars)")

        files = sorted(
            path for path in root_path.rglob("*") if path.is_file() and path.suffix.lower() in {".md", ".markdown"}
        )
        if not files:
            raise FileNotFoundError(f"No Markdown files found under: {root_path}")

        chunks: list[_Chunk] = []
        for path in files:
            chunks.extend(_chunks_from_file(path, root_path, chunk_chars, overlap_chars))
        return cls(chunks, k1=k1, b=b, quality_weights=quality_weights)

    def search(
        self,
        query: str,
        top_k: int = 3,
        *,
        candidate_k: int | None = None,
        quality_rerank: bool = False,
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []
        candidate_limit = max(top_k, int(candidate_k or top_k))
        query_counts = Counter(tokenize(query))
        scores: dict[int, float] = defaultdict(float)
        for term, query_frequency in query_counts.items():
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            query_weight = 1.0 + math.log(query_frequency)
            for doc_id, term_frequency in postings:
                length_norm = 1.0 - self.b + self.b * (self._lengths[doc_id] / self._average_length)
                term_score = term_frequency * (self.k1 + 1.0) / (term_frequency + self.k1 * length_norm)
                scores[doc_id] += idf * term_score * query_weight

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        selected: list[SearchResult] = []
        source_counts: Counter[str] = Counter()
        for doc_id, score in ranked:
            chunk = self._chunks[doc_id]
            if source_counts[chunk.source] >= 2:
                continue
            quality = self._qualities[doc_id]
            candidate_rank = len(selected) + 1
            adjusted_score = quality.weight / (60.0 + candidate_rank) if quality_rerank else score
            selected.append(
                SearchResult(
                    source=chunk.source,
                    heading=chunk.heading,
                    text=chunk.text,
                    score=adjusted_score,
                    raw_score=score,
                    quality_category=quality.category,
                    quality_weight=quality.weight,
                )
            )
            source_counts[chunk.source] += 1
            if len(selected) >= candidate_limit:
                break
        if quality_rerank:
            selected.sort(key=lambda result: (-result.score, -float(result.raw_score or 0.0), result.source))
        return selected[:top_k]


def _best_snippet(text: str, query: str, limit: int) -> str:
    compact = _WHITESPACE.sub(" ", text).strip()
    if len(compact) <= limit:
        return compact
    terms = sorted({term for term in tokenize(query) if len(term) >= 2}, key=len, reverse=True)
    lowered = compact.lower()
    positions = [lowered.find(term.lower()) for term in terms]
    positions = [position for position in positions if position >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - limit // 3)
    end = min(len(compact), start + limit)
    start = max(0, end - limit)
    prefix = "…" if start else ""
    suffix = "…" if end < len(compact) else ""
    return prefix + compact[start:end].strip() + suffix


def format_search_results(
    results: list[SearchResult],
    query: str,
    *,
    max_chars: int = 1800,
    per_result_chars: int = 520,
) -> str:
    """Format ranked snippets for safe, bounded environment feedback."""
    if not results:
        return "[检索结果]\n未找到匹配文档。请改用更具体的设备、流程或规范关键词。"

    blocks = ["[检索结果]"]
    for index, result in enumerate(results, start=1):
        snippet = _best_snippet(result.text, query, per_result_chars)
        snippet = snippet.replace("<search", "＜search").replace("</search>", "＜/search＞")
        heading = f" · {result.heading}" if result.heading else ""
        display_score = result.raw_score if result.raw_score is not None else result.score
        blocks.append(f"{index}. 来源：{result.source}{heading}\n相关度：{display_score:.2f}\n{snippet}")
    return "\n\n".join(blocks)[:max_chars]
