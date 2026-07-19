#!/usr/bin/env python
"""Build clean document-grounded one-hop/two-hop fill SFT trajectories."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.retrieval.evidence import (  # noqa: E402
    expected_keypoints,
    normalize_evidence_text,
    text_keypoint_hits,
    visible_evidence_keypoint_hits,
)
from common.retrieval.markdown_bm25 import (  # noqa: E402
    MarkdownBM25Index,
    SearchResult,
    build_retrieval_query,
    format_search_results_with_visible_snippets,
)
from common.retrieval.qa_sft import (  # noqa: E402
    build_search_messages,
    validate_messages,
)
from common.retrieval.qa_target_rebuild import question_fingerprint  # noqa: E402
from experiments.qa_sft_v2_data_build_wanghaonan.run import (  # noqa: E402
    DEFAULT_V1_MANIFEST,
    MAX_TOKENS,
    _load_tokenizer,
    _objective_record,
    _runtime_messages,
    _token_length,
)

DOCS_ROOT = Path("/data/docs")
OFFICIAL_VALIDATION = Path("/data/datasets/qa_rl/val.jsonl")
TARGETS = {
    "train": {"one_hop": 160, "two_hop": 120},
    "validation": {"one_hop": 20, "two_hop": 20},
}
OBJECTIVE_TRAIN_PER_TYPE = 23
OBJECTIVE_REPEAT_PER_TYPE = 9
OBJECTIVE_VALIDATION_PER_TYPE = 2
POOL_TARGETS = {"train": 500, "validation": 100}
MAX_RAW_CANDIDATES = 16000
MAX_CANDIDATES_PER_SOURCE = 3
TRAIN_MAX_TOKENS = 3072
OBJECTIVE_TYPES = ("single", "multiple", "bool")
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])|\n+")
_NUMERIC = re.compile(
    r"(?<![A-Za-z0-9])"
    r"\d+(?:\.\d+)?\s*"
    r"(?:nm|um|μm|µm|mm|cm|℃|°C|mV|V|mA|A|W|kW|%|ppm|颗|次|小时|分钟|秒|片|个|层)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_ACRONYM = re.compile(
    r"(?<![A-Za-z0-9])[A-Z][A-Z0-9+./-]{2,14}(?![A-Za-z0-9])"
)
_QUOTED = re.compile(r"[“\"]([\u3400-\u4dbf\u4e00-\u9fff]{2,12})[”\"]")
_SPACE = re.compile(r"\s+")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_ENGLISH_WORD = re.compile(r"\b[A-Za-z]{2,}\b")
_SEMANTIC_PREDICATE = re.compile(
    r"是|为|指|称为|叫做|用于|用来|通过|控制|表示|检测|测量|"
    r"计算|定义|包括|小于|大于|不超过|不少于|范围|厚度|温度|"
    r"精度|浓度|时间|距离|产生|形成|实现|负责|管理|记录|解决|"
    r"\b(?:is|are|means|stands\s+for|used|controls?|measures?|"
    r"calculates?|loads?|contains?|consists?|defined|requires?|records?)\b",
    re.IGNORECASE,
)
_BULLET_PREFIX = re.compile(r"^\s*[–—•●▪✓]\s*")
_COMPLETE_SENTENCE_END = re.compile(r"[。！？!?.)）\]】]$")
_DISCOURSE_FRAGMENT = re.compile(r"^(?:其次|上表中|如下|上述)")
_OPERATION_FRAGMENT = re.compile(
    r"依次点击|再点击|点击.*(?:确认|密码)|"
    r"\b(?:MAIN\s+MENU|MENU\s+screen|button|0/1\s+switch)\b",
    re.IGNORECASE,
)
_SHIFT_LOG_FRAGMENT = re.compile(
    r"\b(?:suffer|follow|rework)\b|值班|写case",
    re.IGNORECASE,
)
_ENGLISH_PREDICATE_FRAGMENT = re.compile(
    r"^(?:measures?|controls?|calculates?|records?|loads?|uses?|provides?)\b",
    re.IGNORECASE,
)
_CONTENT_NOISE = re.compile(
    r"slide\s+number|###\s*notes|<!--|<html|copyright|all\s+rights|"
    r"prior\s+consent|own\s+risk|equivalent\s+to\s+rev|"
    r"new\s+york.*london|\\n",
    re.IGNORECASE,
)
_BANNED_SOURCE = re.compile(
    r"试题|试卷|考题|题库|考试|练习题|测验|答案|certify|quiz|exam",
    re.IGNORECASE,
)
_BANNED_ANSWERS = {
    "fab",
    "wafer",
    "lot",
    "job",
    "step",
    "item",
    "system",
    "server",
    "new",
    "note",
    "page",
    "pac",
    "pdf",
    "rev",
    "sop",
    "pe",
    "yes",
    "no",
}


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def _output_dir(overrides: Sequence[str]) -> Path:
    for override in overrides:
        if override.startswith("logger.log_dir="):
            output = Path(override.split("=", 1)[1]).parent / "grounded_cloze"
            output.mkdir(parents=True, exist_ok=True)
            return output
    output = THIS_DIR / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stable_hash(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _source_split(source: str) -> str:
    return (
        "validation"
        if int(_stable_hash(f"source:{source}")[:8], 16) % 8 == 0
        else "train"
    )


def _clean_text(value: str) -> str:
    return _SPACE.sub(" ", str(value)).strip()


def _answer_candidates(sentence: str) -> list[tuple[str, str]]:
    candidates = [
        *(("numeric_unit", match.group(0)) for match in _NUMERIC.finditer(sentence)),
        *(("acronym", match.group(0)) for match in _ACRONYM.finditer(sentence)),
        *(("quoted_term", match.group(1)) for match in _QUOTED.finditer(sentence)),
    ]
    accepted = []
    seen = set()
    lowered = sentence.lower()
    for kind, answer in candidates:
        answer = answer.strip()
        normalized = normalize_evidence_text(answer)
        if (
            not 2 <= len(normalized) <= 40
            or normalized in _BANNED_ANSWERS
            or normalized in seen
            or lowered.count(answer.lower()) != 1
        ):
            continue
        seen.add(normalized)
        accepted.append((kind, answer))
    return accepted


def candidate_quality_issues(
    sentence: str,
    answer: str,
    answer_kind: str,
) -> list[str]:
    issues = []
    if _CONTENT_NOISE.search(sentence):
        issues.append("boilerplate_or_slide_noise")
    if sentence.count("|") >= 2 or sentence.count("\t") >= 2:
        issues.append("table_fragment")
    if sentence.lstrip().startswith(("--", "<", "|")):
        issues.append("markup_fragment")
    if _BULLET_PREFIX.search(sentence):
        issues.append("slide_bullet_fragment")
    if not _COMPLETE_SENTENCE_END.search(sentence):
        issues.append("missing_sentence_terminator")
    if _DISCOURSE_FRAGMENT.search(sentence):
        issues.append("context_dependent_fragment")
    if _OPERATION_FRAGMENT.search(sentence):
        issues.append("button_or_operation_fragment")
    if _SHIFT_LOG_FRAGMENT.search(sentence):
        issues.append("shift_log_fragment")
    if _ENGLISH_PREDICATE_FRAGMENT.search(sentence):
        issues.append("english_predicate_fragment")
    if len(_CJK.findall(sentence)) < 10 and len(_ENGLISH_WORD.findall(sentence)) < 8:
        issues.append("insufficient_sentence_context")
    if not _SEMANTIC_PREDICATE.search(sentence):
        issues.append("missing_semantic_predicate")

    normalized_answer = normalize_evidence_text(answer)
    if answer_kind == "acronym":
        if not re.fullmatch(r"[A-Z]{3,10}", answer):
            issues.append("code_like_acronym")
        if normalized_answer in _BANNED_ANSWERS:
            issues.append("generic_acronym")
        position = sentence.lower().find(answer.lower())
        if position >= 0:
            adjacent = sentence[max(0, position - 1) : position + len(answer) + 1]
            if adjacent.startswith("-") or adjacent.endswith("-"):
                issues.append("hyphen_fragment")
    elif answer_kind == "numeric_unit":
        compact = answer.replace(" ", "")
        if re.match(r"^0[1-9]\d*(?!\.)", compact):
            issues.append("malformed_leading_zero")
    elif answer_kind == "quoted_term" and not re.search(
        r"是|为|指|称为|叫做|所谓|方法|工艺",
        sentence,
    ):
        issues.append("ungrounded_quoted_term")
    return sorted(set(issues))


def _replace_case_insensitive(text: str, answer: str, replacement: str) -> str:
    return re.sub(re.escape(answer), replacement, text, count=1, flags=re.IGNORECASE)


def _fill_prompt(masked_sentences: Sequence[str]) -> str:
    count = len(masked_sentences)
    numbered = "\n".join(
        f"{index}. {sentence.replace('【1】', f'【{index}】')}"
        for index, sentence in enumerate(masked_sentences, start=1)
    )
    return (
        f"下面是一道技术资料填空题，共 {count} 个空。"
        "请先检索资料，再按编号顺序填写。\n"
        "把最终答案放入 \\boxed{}，各空用“;”分隔。\n\n"
        f"题目：\n{numbered}"
    )


def _candidate_query(heading: str, masked_sentence: str, answer: str) -> str:
    clean_heading = _replace_case_insensitive(heading, answer, "")
    context = masked_sentence.replace("【1】", "待填")
    return _clean_text(f"{clean_heading} {context}")[:256]


def _extract_raw_candidates(
    index: MarkdownBM25Index,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records = []
    source_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    fingerprints = set()
    for chunk in index.iter_chunks(quality_categories={"reference"}):
        if len(records) >= MAX_RAW_CANDIDATES:
            break
        if _BANNED_SOURCE.search(chunk.source):
            reason_counts["banned_source"] += 1
            continue
        if source_counts[chunk.source] >= MAX_CANDIDATES_PER_SOURCE:
            continue
        for raw_sentence in _SENTENCE_SPLIT.split(chunk.text):
            sentence = _clean_text(raw_sentence)
            if (
                not 28 <= len(sentence) <= 220
                or "【" in sentence
                or "____" in sentence
                or "答案" in sentence
            ):
                continue
            for answer_kind, answer in _answer_candidates(sentence):
                quality_issues = candidate_quality_issues(
                    sentence,
                    answer,
                    answer_kind,
                )
                if quality_issues:
                    reason_counts.update(quality_issues)
                    continue
                masked = _replace_case_insensitive(sentence, answer, "【1】")
                if normalize_evidence_text(answer) in normalize_evidence_text(masked):
                    reason_counts["answer_still_visible"] += 1
                    continue
                query = _candidate_query(chunk.heading, masked, answer)
                if (
                    not query
                    or normalize_evidence_text(answer)
                    in normalize_evidence_text(query)
                ):
                    reason_counts["query_answer_leak"] += 1
                    continue
                prompt = _fill_prompt([masked])
                fingerprint = question_fingerprint(prompt)
                if fingerprint in fingerprints:
                    reason_counts["duplicate_question"] += 1
                    continue
                fingerprints.add(fingerprint)
                source_counts[chunk.source] += 1
                records.append(
                    {
                        "candidate_id": _stable_hash(
                            f"{chunk.source}:{chunk.heading}:{sentence}:{answer}"
                        ),
                        "source": chunk.source,
                        "heading": chunk.heading,
                        "sentence": sentence,
                        "masked_sentence": masked,
                        "answer": answer,
                        "answer_kind": answer_kind,
                        "model_query": query,
                        "query": prompt,
                        "expected_answer": f"[fill] {answer}",
                        "split": _source_split(chunk.source),
                    }
                )
                break
            if source_counts[chunk.source] >= MAX_CANDIDATES_PER_SOURCE:
                break
    records.sort(key=lambda record: record["candidate_id"])
    return records, dict(sorted(reason_counts.items()))


def _visible_records(
    results: Sequence[SearchResult],
    snippets: Sequence[str],
) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "source": result.source,
            "heading": result.heading,
            "quality_category": result.quality_category,
            "raw_score": result.raw_score,
            "text": snippet,
        }
        for rank, (result, snippet) in enumerate(
            zip(results, snippets, strict=True),
            start=1,
        )
    ]


def _search(
    index: MarkdownBM25Index,
    *,
    model_query: str,
    original_query: str,
    keypoints: Sequence[Sequence[str]],
    exclude_sources: set[str] | None = None,
) -> dict[str, Any]:
    retrieval_query = build_retrieval_query(model_query, original_query)
    results = index.search(
        retrieval_query,
        top_k=4,
        candidate_k=50,
        quality_rerank=True,
        exclude_sources=exclude_sources,
    )
    observation, snippets = format_search_results_with_visible_snippets(
        results,
        retrieval_query,
        max_chars=1800,
        per_result_chars=360,
    )
    return {
        "model_search_query": model_query,
        "retrieval_query": retrieval_query,
        "observation": observation,
        "hits": visible_evidence_keypoint_hits(results, snippets, keypoints),
        "results": results,
        "visible_snippets": snippets,
        "top_k_results": _visible_records(results, snippets),
    }


def _source_supports_answer(
    hop: Mapping[str, Any],
    source: str,
    keypoint_index: int,
    keypoints: Sequence[Sequence[str]],
) -> bool:
    return any(
        result.source == source
        and keypoint_index in text_keypoint_hits(snippet, keypoints)
        for result, snippet in zip(
            hop["results"],
            hop["visible_snippets"],
            strict=True,
        )
    )


def _base_fill_record(
    *,
    query: str,
    expected: str,
    split: str,
    search_hops: Sequence[Mapping[str, Any]],
    messages: Sequence[Mapping[str, str]],
    tokenizer,
    source_candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    issues = validate_messages(messages)
    if issues:
        return None
    runtime_messages = _runtime_messages(tokenizer, messages)
    token_length = _token_length(tokenizer, runtime_messages)
    if token_length > min(MAX_TOKENS, TRAIN_MAX_TOKENS):
        return None
    serialized_hops = []
    cumulative = set()
    for hop in search_hops:
        hits = set(hop["hits"])
        new_hits = hits - cumulative
        cumulative.update(hits)
        serialized_hops.append(
            {
                key: value
                for key, value in hop.items()
                if key not in {"results", "visible_snippets", "hits"}
            }
            | {
                "hits": sorted(hits),
                "new_hits": sorted(new_hits),
            }
        )
    return {
        "source_kind": "document_grounded_cloze",
        "question_fingerprint": question_fingerprint(query),
        "question_type": "fill",
        "query": query,
        "expected_answer": expected,
        "search_turns": len(search_hops),
        "search_hops": serialized_hops,
        "split": split,
        "messages": runtime_messages,
        "machine_verified": True,
        "human_reviewed": False,
        "source_candidates": [
            {
                key: value
                for key, value in candidate.items()
                if key != "_one_hop_record"
            }
            for candidate in source_candidates
        ],
        "_audit": {
            "trusted_visible_coverage": 1.0,
            "incremental_two_hop": len(search_hops) == 2,
            "query_leakage_check": True,
            "token_length": token_length,
            "runtime_raw_chunk_alignment": True,
        },
    }


def _build_one_hop(
    index: MarkdownBM25Index,
    candidate: Mapping[str, Any],
    tokenizer,
) -> dict[str, Any] | None:
    _question_type, keypoints = expected_keypoints(
        str(candidate["expected_answer"])
    )
    first = _search(
        index,
        model_query=str(candidate["model_query"]),
        original_query=str(candidate["query"]),
        keypoints=keypoints,
    )
    if set(first["hits"]) != {0} or not _source_supports_answer(
        first,
        str(candidate["source"]),
        0,
        keypoints,
    ):
        return None
    messages = build_search_messages(
        query=str(candidate["query"]),
        expected=str(candidate["expected_answer"]),
        first_query=str(candidate["model_query"]),
        first_observation=str(first["observation"]),
    )
    return _base_fill_record(
        query=str(candidate["query"]),
        expected=str(candidate["expected_answer"]),
        split=str(candidate["split"]),
        search_hops=[first],
        messages=messages,
        tokenizer=tokenizer,
        source_candidates=[candidate],
    )


def _pair_prompt(first: Mapping[str, Any], second: Mapping[str, Any]) -> str:
    return _fill_prompt(
        [
            str(first["masked_sentence"]),
            str(second["masked_sentence"]),
        ]
    )


def _can_pair(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> bool:
    if first["source"] == second["source"] or first["split"] != second["split"]:
        return False
    first_answer = normalize_evidence_text(str(first["answer"]))
    second_answer = normalize_evidence_text(str(second["answer"]))
    if not first_answer or not second_answer or first_answer == second_answer:
        return False
    prompt = _pair_prompt(first, second)
    normalized_prompt = normalize_evidence_text(prompt)
    first_query = normalize_evidence_text(str(first.get("model_query", "")))
    second_query = normalize_evidence_text(str(second.get("model_query", "")))
    return (
        first_answer not in normalized_prompt
        and second_answer not in normalized_prompt
        and first_answer not in first_query
        and second_answer not in first_query
        and first_answer not in second_query
        and second_answer not in second_query
    )


def _build_two_hop(
    index: MarkdownBM25Index,
    first_candidate: Mapping[str, Any],
    second_candidate: Mapping[str, Any],
    tokenizer,
) -> dict[str, Any] | None:
    if not _can_pair(first_candidate, second_candidate):
        return None
    query = _pair_prompt(first_candidate, second_candidate)
    expected = (
        f"[fill] {first_candidate['answer']} ||| {second_candidate['answer']}"
    )
    _question_type, keypoints = expected_keypoints(expected)
    first = _search(
        index,
        model_query=str(first_candidate["model_query"]),
        original_query=query,
        keypoints=keypoints,
    )
    if set(first["hits"]) != {0} or not _source_supports_answer(
        first,
        str(first_candidate["source"]),
        0,
        keypoints,
    ):
        return None
    first_sources = {result.source for result in first["results"]}
    second = _search(
        index,
        model_query=str(second_candidate["model_query"]),
        original_query=query,
        keypoints=keypoints,
        exclude_sources=first_sources,
    )
    if (
        1 not in set(second["hits"])
        or not _source_supports_answer(
            second,
            str(second_candidate["source"]),
            1,
            keypoints,
        )
        or set(first["hits"]) | set(second["hits"]) != {0, 1}
    ):
        return None
    messages = build_search_messages(
        query=query,
        expected=expected,
        first_query=str(first_candidate["model_query"]),
        first_observation=str(first["observation"]),
        second_query=str(second_candidate["model_query"]),
        second_observation=str(second["observation"]),
    )
    return _base_fill_record(
        query=query,
        expected=expected,
        split=str(first_candidate["split"]),
        search_hops=[first, second],
        messages=messages,
        tokenizer=tokenizer,
        source_candidates=[first_candidate, second_candidate],
    )


def _validated_pools(
    index: MarkdownBM25Index,
    raw_candidates: Sequence[Mapping[str, Any]],
    tokenizer,
    *,
    pool_targets: Mapping[str, int] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
    targets = dict(pool_targets or POOL_TARGETS)
    pools: dict[str, list[dict[str, Any]]] = {
        split: [] for split in targets
    }
    reasons: Counter[str] = Counter()
    for candidate in raw_candidates:
        split = str(candidate["split"])
        if split not in pools or len(pools[split]) >= targets[split]:
            continue
        record = _build_one_hop(index, candidate, tokenizer)
        if record is None:
            reasons["one_hop_retrieval_rejected"] += 1
            continue
        pools[split].append(
            {
                **dict(candidate),
                "_one_hop_record": record,
            }
        )
        if all(
            len(pools[current]) >= target
            for current, target in targets.items()
        ):
            break
    return pools, reasons


def _build_pairs(
    index: MarkdownBM25Index,
    candidates: Sequence[Mapping[str, Any]],
    tokenizer,
    target: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    pairs = []
    reasons: Counter[str] = Counter()
    used_pairs = set()
    for first_index, first in enumerate(candidates):
        if len(pairs) >= target:
            break
        for offset in range(1, min(100, len(candidates))):
            second_index = (first_index + offset) % len(candidates)
            second = candidates[second_index]
            pair_key = tuple(
                sorted((str(first["candidate_id"]), str(second["candidate_id"])))
            )
            if pair_key in used_pairs or not _can_pair(first, second):
                continue
            used_pairs.add(pair_key)
            record = _build_two_hop(index, first, second, tokenizer)
            if record is None:
                reasons["two_hop_retrieval_rejected"] += 1
                continue
            pairs.append(record)
            break
    return pairs, reasons


def _objective_replay(
    tokenizer,
    official_fingerprints: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records = _read_jsonl(DEFAULT_V1_MANIFEST)
    accepted = []
    seen = set()
    for source in records:
        record, _reason = _objective_record(
            source,
            official_fingerprints,
            tokenizer,
        )
        if record is None:
            continue
        fingerprint = str(record["question_fingerprint"])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        accepted.append(record)

    by_split_type: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in accepted:
        by_split_type[
            (str(record["split"]), str(record["question_type"]))
        ].append(record)
    for group in by_split_type.values():
        group.sort(key=lambda record: str(record["question_fingerprint"]))

    train = []
    validation = []
    available = {}
    for question_type in OBJECTIVE_TYPES:
        train_group = by_split_type[("train", question_type)]
        validation_group = by_split_type[("validation", question_type)]
        available[f"train:{question_type}"] = len(train_group)
        available[f"validation:{question_type}"] = len(validation_group)
        if (
            len(train_group) < OBJECTIVE_TRAIN_PER_TYPE
            or len(validation_group) < OBJECTIVE_VALIDATION_PER_TYPE
        ):
            raise RuntimeError(
                f"insufficient objective replay for {question_type}: "
                f"train={len(train_group)} validation={len(validation_group)}"
            )
        selected = train_group[:OBJECTIVE_TRAIN_PER_TYPE]
        for exposure, record in enumerate(
            [
                *selected,
                *selected[:OBJECTIVE_REPEAT_PER_TYPE],
            ],
            start=1,
        ):
            clone = copy.deepcopy(record)
            clone["source_kind"] = "objective_replay"
            clone["objective_exposure"] = exposure
            train.append(clone)
        validation.extend(
            copy.deepcopy(
                validation_group[:OBJECTIVE_VALIDATION_PER_TYPE]
            )
        )
    for record in validation:
        record["source_kind"] = "objective_replay"
    return train, validation, available


def _profile(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if row.get("question_type") in OBJECTIVE_TYPES:
            counts[f"objective:{row['question_type']}"] += 1
        else:
            counts[f"fill:{row['search_turns']}_hop"] += 1
    return dict(sorted(counts.items()))


def _split_source_overlap(
    train_fill: Sequence[Mapping[str, Any]],
    validation_fill: Sequence[Mapping[str, Any]],
) -> list[str]:
    def sources(rows):
        return {
            str(candidate["source"])
            for row in rows
            for candidate in row["source_candidates"]
        }

    return sorted(sources(train_fill) & sources(validation_fill))


def main() -> None:
    _, overrides = _parse_args()
    output_dir = _output_dir(overrides)
    required = [
        DOCS_ROOT,
        OFFICIAL_VALIDATION,
        DEFAULT_V1_MANIFEST,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing grounded cloze inputs: {missing}")

    official_rows = _read_jsonl(OFFICIAL_VALIDATION)
    official_fingerprints = {
        question_fingerprint(str(row["query"]))
        for row in official_rows
    }
    if len(official_rows) != 313 or len(official_fingerprints) != 313:
        raise RuntimeError("official validation integrity check failed")

    index_start = time.perf_counter()
    index = MarkdownBM25Index.from_directory(
        DOCS_ROOT,
        chunk_chars=1200,
        overlap_chars=160,
        k1=1.5,
        b=0.75,
    )
    index_seconds = time.perf_counter() - index_start
    print(
        f"[grounded-cloze-data] indexed={index.num_documents} "
        f"seconds={index_seconds:.1f}",
        flush=True,
    )
    tokenizer = _load_tokenizer()
    objective_train, objective_validation, objective_available = (
        _objective_replay(tokenizer, official_fingerprints)
    )
    print(
        f"[grounded-cloze-data] objective train={len(objective_train)} "
        f"validation={len(objective_validation)}",
        flush=True,
    )
    raw_candidates, extraction_reasons = _extract_raw_candidates(index)
    print(
        f"[grounded-cloze-data] raw_candidates={len(raw_candidates)}",
        flush=True,
    )
    pools, pool_reasons = _validated_pools(
        index,
        raw_candidates,
        tokenizer,
    )
    print(
        f"[grounded-cloze-data] pools="
        f"{ {split: len(rows) for split, rows in pools.items()} }",
        flush=True,
    )

    fill_by_split: dict[str, list[dict[str, Any]]] = {}
    pair_reasons: Counter[str] = Counter()
    for split, targets in TARGETS.items():
        one_hop = [
            copy.deepcopy(candidate["_one_hop_record"])
            for candidate in pools[split][: targets["one_hop"]]
        ]
        two_hop, reasons = _build_pairs(
            index,
            pools[split],
            tokenizer,
            targets["two_hop"],
        )
        pair_reasons.update(reasons)
        fill_by_split[split] = [*one_hop, *two_hop]
        print(
            f"[grounded-cloze-data] split={split} "
            f"one_hop={len(one_hop)} two_hop={len(two_hop)}",
            flush=True,
        )
    train_fill = fill_by_split["train"]
    validation_fill = fill_by_split["validation"]
    train = [*train_fill, *objective_train]
    validation = [*validation_fill, *objective_validation]
    train.sort(
        key=lambda row: _stable_hash(
            f"train:{row['question_fingerprint']}:{row.get('objective_exposure', 0)}"
        )
    )
    validation.sort(
        key=lambda row: _stable_hash(
            f"validation:{row['question_fingerprint']}"
        )
    )

    train_fingerprints = {
        str(row["question_fingerprint"]) for row in train
    }
    validation_fingerprints = {
        str(row["question_fingerprint"]) for row in validation
    }
    official_overlap = sorted(
        (train_fingerprints | validation_fingerprints)
        & official_fingerprints
    )
    split_question_overlap = sorted(
        train_fingerprints & validation_fingerprints
    )
    split_source_overlap = _split_source_overlap(
        train_fill,
        validation_fill,
    )
    fill_counts = {
        split: {
            "one_hop": sum(
                int(row["search_turns"]) == 1
                for row in rows
            ),
            "two_hop": sum(
                int(row["search_turns"]) == 2
                for row in rows
            ),
        }
        for split, rows in fill_by_split.items()
    }
    max_tokens = max(
        int(row["_audit"]["token_length"])
        for row in [*train_fill, *validation_fill]
    )
    machine_gate = {
        "target_fill_counts": TARGETS,
        "actual_fill_counts": fill_counts,
        "official_validation_overlap_count": len(official_overlap),
        "split_question_overlap_count": len(split_question_overlap),
        "split_source_overlap_count": len(split_source_overlap),
        "maximum_token_length": max_tokens,
        "passed": (
            fill_counts == TARGETS
            and not official_overlap
            and not split_question_overlap
            and not split_source_overlap
            and max_tokens <= TRAIN_MAX_TOKENS
        ),
    }
    objective_fraction = len(objective_train) / len(train) if train else 0.0
    summary = {
        "mode": "document_grounded_contrastive_cloze_sft_pack",
        "sources": {
            "docs_root": str(DOCS_ROOT),
            "official_validation": str(OFFICIAL_VALIDATION),
            "objective_manifest": str(DEFAULT_V1_MANIFEST),
        },
        "index": {
            "num_documents": index.num_documents,
            "build_seconds": index_seconds,
            "quality_category_counts": index.quality_category_counts,
        },
        "candidate_extraction": {
            "raw_candidate_count": len(raw_candidates),
            "raw_split_counts": dict(
                sorted(
                    Counter(
                        str(candidate["split"])
                        for candidate in raw_candidates
                    ).items()
                )
            ),
            "answer_kind_counts": dict(
                sorted(
                    Counter(
                        str(candidate["answer_kind"])
                        for candidate in raw_candidates
                    ).items()
                )
            ),
            "reason_counts": extraction_reasons,
        },
        "validated_pool_counts": {
            split: len(rows) for split, rows in pools.items()
        },
        "retrieval_rejection_counts": dict(
            sorted((pool_reasons + pair_reasons).items())
        ),
        "output_counts": {
            "train": len(train),
            "validation": len(validation),
        },
        "profiles": {
            "train": _profile(train),
            "validation": _profile(validation),
        },
        "objective_replay": {
            "train_count": len(objective_train),
            "validation_count": len(objective_validation),
            "train_fraction": objective_fraction,
            "available": objective_available,
        },
        "machine_gate": machine_gate,
        "human_reviewed": False,
        "training_authorized_by_user": True,
        "training_submitted": False,
        "outputs": {
            "train": str(output_dir / "train.jsonl"),
            "validation": str(output_dir / "validation.jsonl"),
            "review_sample": str(output_dir / "review_sample.jsonl"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    review_sample = [
        *train_fill[:8],
        *[
            row for row in train_fill
            if int(row["search_turns"]) == 2
        ][:12],
        *validation_fill[:8],
    ]
    _write_jsonl(output_dir / "train.jsonl", train)
    _write_jsonl(output_dir / "validation.jsonl", validation)
    _write_jsonl(output_dir / "review_sample.jsonl", review_sample)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("[grounded-cloze-data]", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
