#!/usr/bin/env python
"""Filter grounded cloze candidates with Qwen, then rebuild and hard-audit."""

from __future__ import annotations

import argparse
import copy
import gc
import itertools
import json
import os
import random
import sys
import time
from collections import Counter
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
)
from common.retrieval.markdown_bm25 import MarkdownBM25Index  # noqa: E402
from common.retrieval.qa_target_rebuild import question_fingerprint  # noqa: E402
from experiments.qa_grounded_cloze_data_wanghaonan import run as cloze  # noqa: E402

MODEL_NAME = "Qwen/Qwen3.5-9B"
DEFAULT_POOL_TARGETS = {"train": 1400, "validation": 280}
MIN_CONFIDENCE = 0.9
CRITIC_NOISE_FIELDS = (
    "table_fragment",
    "ppt_or_slide_fragment",
    "header_footer_or_copyright",
    "ocr_corruption",
    "machine_translation",
    "truncated_fragment",
    "button_or_operation_fragment",
    "product_or_code_identifier",
    "other_noise",
)
CRITIC_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "decision",
        "confidence",
        "complete_natural_statement",
        "meaningful_answer",
        "noise",
        "reason",
    ],
    "properties": {
        "decision": {"enum": ["accept", "reject"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "complete_natural_statement": {"type": "boolean"},
        "meaningful_answer": {"type": "boolean"},
        "noise": {
            "type": "object",
            "additionalProperties": False,
            "required": list(CRITIC_NOISE_FIELDS),
            "properties": {
                field: {"type": "boolean"} for field in CRITIC_NOISE_FIELDS
            },
        },
        "reason": {"type": "string", "minLength": 1, "maxLength": 240},
    },
}
CRITIC_SYSTEM_PROMPT = """
你是 grounded cloze 数据的严格语义质检员。你只能检查输入文本质量，不能使用外部知识
判断技术事实真伪，不能改写原句、masked_question 或 answer，也不能提出替代答案。

逐项判断：
1. sentence 是否是语法完整、自然、自包含的陈述，而不是在句中或从句中突然结束；
2. answer 是否是该句中有意义的技术概念、缩写或带单位数值，而不是产品型号、物料号、
   页面代码、按钮文字、泛化英文词；
3. 是否有表格行、PPT/讲义碎片、页眉页脚/版权、OCR 乱码、明显机翻、截断残句、
   操作按钮残句、产品/代码编号或其他文档噪声。

严格规则：
- “when performing”“是由某公司”之类未完成从句、缺少宾语或谓语的文本必须 reject；
- 命令式按钮操作碎片、菜单/按键串、产品型号介绍残片必须 reject；
- 只根据给定 sentence、masked_question、answer、answer_kind、source、heading 判断；
- 不确定时 reject；accept 仅用于所有检查都清晰通过且 confidence >= 0.90；
- 只能输出一个符合给定 JSON Schema 的 JSON 对象，不要 Markdown 或额外文字。

JSON Schema:
""".strip() + "\n" + json.dumps(CRITIC_JSON_SCHEMA, ensure_ascii=False, sort_keys=True)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def _output_dir(overrides: Sequence[str]) -> Path:
    for override in overrides:
        if override.startswith("logger.log_dir="):
            output = (
                Path(override.split("=", 1)[1]).parent
                / "grounded_cloze_semantic"
            )
            output.mkdir(parents=True, exist_ok=True)
            return output
    output = THIS_DIR / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _chat_prompt(tokenizer, candidate: Mapping[str, Any]) -> str:
    user = json.dumps(
        {
            "sentence": str(candidate["sentence"]),
            "masked_question": str(candidate["masked_sentence"]),
            "answer": str(candidate["answer"]),
            "answer_kind": str(candidate["answer_kind"]),
            "source": str(candidate["source"]),
            "heading": str(candidate["heading"]),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "add_special_tokens": False,
    }
    try:
        return str(
            tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **kwargs,
            )
        )
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        return str(tokenizer.apply_chat_template(messages, **kwargs))


def _validate_verdict(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(raw.strip())
    except (json.JSONDecodeError, TypeError):
        return None, "invalid_json"
    if not isinstance(payload, dict):
        return None, "schema_not_object"
    required = set(CRITIC_JSON_SCHEMA["required"])
    if set(payload) != required:
        return None, "schema_fields"
    if payload["decision"] not in {"accept", "reject"}:
        return None, "schema_decision"
    confidence = payload["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= float(confidence) <= 1
    ):
        return None, "schema_confidence"
    if type(payload["complete_natural_statement"]) is not bool:
        return None, "schema_complete_statement"
    if type(payload["meaningful_answer"]) is not bool:
        return None, "schema_meaningful_answer"
    noise = payload["noise"]
    if not isinstance(noise, dict) or set(noise) != set(CRITIC_NOISE_FIELDS):
        return None, "schema_noise_fields"
    if any(type(noise[field]) is not bool for field in CRITIC_NOISE_FIELDS):
        return None, "schema_noise_type"
    reason = payload["reason"]
    if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 240:
        return None, "schema_reason"
    semantically_clean = (
        payload["complete_natural_statement"]
        and payload["meaningful_answer"]
        and not any(noise.values())
    )
    if (payload["decision"] == "accept") != semantically_clean:
        return None, "schema_contradiction"
    payload["confidence"] = float(confidence)
    payload["reason"] = reason.strip()
    return payload, None


def _load_critic():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.enable_cudnn_sdp(False)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        .eval()
        .to("cuda")
    )
    torch.manual_seed(42)
    random.seed(42)
    print(
        "[cloze-critic] model=Qwen/Qwen3.5-9B thinking=false "
        "sampling=false cudnn_sdp=false",
        flush=True,
    )
    return tokenizer, model


def _generate_verdicts(
    candidates: Sequence[Mapping[str, Any]],
    tokenizer,
    model,
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    import torch

    audits = []
    prompts = [_chat_prompt(tokenizer, candidate) for candidate in candidates]
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        input_width = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=320,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        raw_outputs = tokenizer.batch_decode(
            generated[:, input_width:],
            skip_special_tokens=True,
        )
        for candidate, raw in zip(
            candidates[start : start + len(batch_prompts)],
            raw_outputs,
            strict=True,
        ):
            verdict, parse_error = _validate_verdict(raw)
            accepted = bool(
                verdict
                and verdict["decision"] == "accept"
                and verdict["confidence"] >= MIN_CONFIDENCE
            )
            rejection = parse_error
            if verdict and verdict["decision"] == "reject":
                rejection = "model_reject"
            elif verdict and verdict["confidence"] < MIN_CONFIDENCE:
                rejection = "confidence_below_threshold"
            audits.append(
                {
                    "candidate_id": str(candidate["candidate_id"]),
                    "split": str(candidate["split"]),
                    "source": str(candidate["source"]),
                    "heading": str(candidate["heading"]),
                    "sentence": str(candidate["sentence"]),
                    "masked_sentence": str(candidate["masked_sentence"]),
                    "answer": str(candidate["answer"]),
                    "answer_kind": str(candidate["answer_kind"]),
                    "raw_generation": raw,
                    "verdict": verdict,
                    "parse_error": parse_error,
                    "accepted": accepted,
                    "rejection": rejection,
                }
            )
        completed = min(start + len(batch_prompts), len(prompts))
        print(f"[cloze-critic] reviewed={completed}/{len(prompts)}", flush=True)
    return audits


def _pool_artifact(candidate: Mapping[str, Any]) -> dict[str, Any]:
    record = candidate["_one_hop_record"]
    return {
        **{
            key: value
            for key, value in candidate.items()
            if key != "_one_hop_record"
        },
        "one_hop_retrieval_gate": {
            "search_hops": record["search_hops"],
            "token_length": record["_audit"]["token_length"],
        },
    }


def _retrieval_signature(hop: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    return [
        (
            str(item["source"]),
            str(item["heading"]),
            str(item["text"]),
        )
        for item in hop["top_k_results"]
    ]


def _hard_gate_issues(
    index: MarkdownBM25Index,
    row: Mapping[str, Any],
    corpus_by_source: Mapping[str, Sequence[str]],
) -> list[str]:
    issues = []
    query = str(row["query"])
    expected = str(row["expected_answer"])
    _question_type, keypoints = expected_keypoints(expected)
    candidates = row["source_candidates"]
    normalized_query = normalize_evidence_text(query)
    expected_answers = [
        value.strip() for value in expected.split("]", 1)[-1].split("|||")
    ]
    if expected_answers != [str(candidate["answer"]) for candidate in candidates]:
        issues.append("gold_candidate_mismatch")
    for candidate in candidates:
        answer = str(candidate["answer"])
        sentence = str(candidate["sentence"])
        masked = str(candidate["masked_sentence"])
        if answer not in sentence:
            issues.append("gold_not_sentence_exact_span")
        if normalize_evidence_text(answer) in normalized_query:
            issues.append("answer_leakage")
        if normalize_evidence_text(answer) in normalize_evidence_text(masked):
            issues.append("masked_answer_leakage")
        if normalize_evidence_text(answer) in normalize_evidence_text(
            str(candidate["model_query"])
        ):
            issues.append("search_query_answer_leakage")
        source_texts = corpus_by_source.get(str(candidate["source"]), ())
        if not any(
            answer in text
            and normalize_evidence_text(sentence)
            in normalize_evidence_text(text)
            for text in source_texts
        ):
            issues.append("sentence_not_reference_span")

    stored_hops = row["search_hops"]
    first = cloze._search(
        index,
        model_query=str(stored_hops[0]["model_search_query"]),
        original_query=query,
        keypoints=keypoints,
    )
    if _retrieval_signature(first) != _retrieval_signature(stored_hops[0]):
        issues.append("first_not_true_bm25_top4")
    if set(first["hits"]) != {0}:
        issues.append("first_hop_not_only_blank1")
    if not cloze._source_supports_answer(
        first,
        str(candidates[0]["source"]),
        0,
        keypoints,
    ):
        issues.append("first_hop_source_not_supporting_blank1")
    if list(stored_hops[0]["new_hits"]) != [0]:
        issues.append("first_hop_increment")

    if int(row["search_turns"]) == 2:
        first_sources = {result.source for result in first["results"]}
        second = cloze._search(
            index,
            model_query=str(stored_hops[1]["model_search_query"]),
            original_query=query,
            keypoints=keypoints,
            exclude_sources=first_sources,
        )
        if _retrieval_signature(second) != _retrieval_signature(stored_hops[1]):
            issues.append("second_not_true_bm25_top4")
        if first_sources & {result.source for result in second["results"]}:
            issues.append("second_hop_source_not_excluded")
        if 1 not in set(second["hits"]) or set(first["hits"]) | set(
            second["hits"]
        ) != {0, 1}:
            issues.append("second_hop_not_incremental_blank2")
        if not cloze._source_supports_answer(
            second,
            str(candidates[1]["source"]),
            1,
            keypoints,
        ):
            issues.append("second_hop_source_not_supporting_blank2")
        if list(stored_hops[1]["new_hits"]) != [1]:
            issues.append("second_hop_increment")
    elif len(stored_hops) != 1:
        issues.append("one_hop_count_mismatch")

    if int(row["_audit"]["token_length"]) > cloze.TRAIN_MAX_TOKENS:
        issues.append("maximum_length")
    return sorted(set(issues))


def _review_sample(
    fill_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    target: int = 32,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen = set()
    kinds = ("numeric_unit", "acronym", "quoted_term")
    dimensions = itertools.product(
        ("train", "validation"),
        (1, 2),
        kinds,
    )
    all_rows = [
        row
        for split in ("train", "validation")
        for row in fill_by_split[split]
    ]
    for split, hop, kind in dimensions:
        match = next(
            (
                row
                for row in fill_by_split[split]
                if int(row["search_turns"]) == hop
                and any(
                    candidate["answer_kind"] == kind
                    for candidate in row["source_candidates"]
                )
                and row["question_fingerprint"] not in seen
            ),
            None,
        )
        if match:
            selected.append(copy.deepcopy(match))
            seen.add(match["question_fingerprint"])
    for row in all_rows:
        if len(selected) >= min(target, len(all_rows)):
            break
        if row["question_fingerprint"] not in seen:
            selected.append(copy.deepcopy(row))
            seen.add(row["question_fingerprint"])
    return selected


def _critic_rejection_counts(audits: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for audit in audits:
        if audit["accepted"]:
            continue
        counts[str(audit["rejection"])] += 1
        verdict = audit.get("verdict")
        if verdict:
            counts[f"reason:{verdict['reason']}"] += 1
            for field, present in verdict["noise"].items():
                if present:
                    counts[f"noise:{field}"] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    _, overrides = _parse_args()
    output_dir = _output_dir(overrides)
    pool_targets = {
        "train": _env_int(
            "CLOZE_CRITIC_POOL_TRAIN",
            DEFAULT_POOL_TARGETS["train"],
        ),
        "validation": _env_int(
            "CLOZE_CRITIC_POOL_VALIDATION",
            DEFAULT_POOL_TARGETS["validation"],
        ),
    }
    max_per_split = _env_int("CLOZE_CRITIC_MAX_PER_SPLIT", 0)
    batch_size = _env_int("CLOZE_CRITIC_BATCH_SIZE", 8)
    if batch_size < 1:
        raise ValueError("CLOZE_CRITIC_BATCH_SIZE must be positive")

    required = [
        cloze.DOCS_ROOT,
        cloze.OFFICIAL_VALIDATION,
        cloze.DEFAULT_V1_MANIFEST,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing grounded cloze inputs: {missing}")
    official_rows = cloze._read_jsonl(cloze.OFFICIAL_VALIDATION)
    official_fingerprints = {
        question_fingerprint(str(row["query"])) for row in official_rows
    }
    if len(official_rows) != 313 or len(official_fingerprints) != 313:
        raise RuntimeError("official validation integrity check failed")

    started = time.perf_counter()
    index = MarkdownBM25Index.from_directory(
        cloze.DOCS_ROOT,
        chunk_chars=1200,
        overlap_chars=160,
        k1=1.5,
        b=0.75,
    )
    raw_candidates, extraction_reasons = cloze._extract_raw_candidates(index)
    tokenizer, model = _load_critic()
    objective_train, objective_validation, objective_available = (
        cloze._objective_replay(tokenizer, official_fingerprints)
    )
    pools, pool_reasons = cloze._validated_pools(
        index,
        raw_candidates,
        tokenizer,
        pool_targets=pool_targets,
    )
    if max_per_split:
        pools = {
            split: rows[:max_per_split] for split, rows in pools.items()
        }
    pool_rows = [
        candidate
        for split in ("train", "validation")
        for candidate in pools[split]
    ]
    print(
        f"[cloze-critic] raw={len(raw_candidates)} "
        f"validated={ {split: len(rows) for split, rows in pools.items()} }",
        flush=True,
    )
    _write_jsonl(
        output_dir / "validated_candidate_pool.jsonl",
        [_pool_artifact(candidate) for candidate in pool_rows],
    )
    audits = _generate_verdicts(
        pool_rows,
        tokenizer,
        model,
        batch_size=batch_size,
    )
    _write_jsonl(output_dir / "semantic_verdicts.jsonl", audits)
    accepted_ids = {
        str(audit["candidate_id"]) for audit in audits if audit["accepted"]
    }
    verdict_by_id = {
        str(audit["candidate_id"]): audit["verdict"]
        for audit in audits
        if audit["accepted"]
    }
    clean_pools = {
        split: [
            {
                **candidate,
                "semantic_critic": verdict_by_id[str(candidate["candidate_id"])],
            }
            for candidate in candidates
            if candidate["candidate_id"] in accepted_ids
        ]
        for split, candidates in pools.items()
    }
    del model
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass

    fill_by_split: dict[str, list[dict[str, Any]]] = {}
    pair_reasons: Counter[str] = Counter()
    for split, targets in cloze.TARGETS.items():
        one_hop = [
            copy.deepcopy(candidate["_one_hop_record"])
            for candidate in clean_pools[split][: targets["one_hop"]]
        ]
        for row, candidate in zip(
            one_hop,
            clean_pools[split][: targets["one_hop"]],
            strict=True,
        ):
            row["source_candidates"][0]["semantic_critic"] = candidate[
                "semantic_critic"
            ]
        two_hop, reasons = cloze._build_pairs(
            index,
            clean_pools[split],
            tokenizer,
            targets["two_hop"],
        )
        pair_reasons.update(reasons)
        fill_by_split[split] = [*one_hop, *two_hop]
        print(
            f"[cloze-critic] split={split} accepted_pool="
            f"{len(clean_pools[split])} one_hop={len(one_hop)} "
            f"two_hop={len(two_hop)}",
            flush=True,
        )

    corpus_by_source: dict[str, list[str]] = {}
    for chunk in index.iter_chunks(quality_categories={"reference"}):
        corpus_by_source.setdefault(chunk.source, []).append(chunk.text)
    hard_gate_counts: Counter[str] = Counter()
    for rows in fill_by_split.values():
        for row in rows:
            hard_gate_counts.update(
                _hard_gate_issues(index, row, corpus_by_source)
            )
    if hard_gate_counts:
        raise RuntimeError(f"post-critic hard gate failed: {dict(hard_gate_counts)}")

    train_fill = fill_by_split["train"]
    validation_fill = fill_by_split["validation"]
    train = [*train_fill, *objective_train]
    validation = [*validation_fill, *objective_validation]
    train.sort(
        key=lambda row: cloze._stable_hash(
            f"train:{row['question_fingerprint']}:"
            f"{row.get('objective_exposure', 0)}"
        )
    )
    validation.sort(
        key=lambda row: cloze._stable_hash(
            f"validation:{row['question_fingerprint']}"
        )
    )
    train_fingerprints = {
        str(row["question_fingerprint"]) for row in train
    }
    validation_fingerprints = {
        str(row["question_fingerprint"]) for row in validation
    }
    split_source_overlap = cloze._split_source_overlap(
        train_fill,
        validation_fill,
    )
    fill_counts = {
        split: {
            "one_hop": sum(row["search_turns"] == 1 for row in rows),
            "two_hop": sum(row["search_turns"] == 2 for row in rows),
        }
        for split, rows in fill_by_split.items()
    }
    max_tokens = max(
        int(row["_audit"]["token_length"]) for row in [*train, *validation]
    )
    machine_gate = {
        "target_fill_counts": cloze.TARGETS,
        "actual_fill_counts": fill_counts,
        "answer_leakage_count": 0,
        "gold_not_source_exact_span_count": 0,
        "bm25_top4_mismatch_count": 0,
        "two_hop_increment_failure_count": 0,
        "official_validation_overlap_count": len(
            (train_fingerprints | validation_fingerprints)
            & official_fingerprints
        ),
        "split_question_overlap_count": len(
            train_fingerprints & validation_fingerprints
        ),
        "split_source_overlap_count": len(split_source_overlap),
        "maximum_token_length": max_tokens,
    }
    machine_gate["passed"] = bool(
        fill_counts == cloze.TARGETS
        and not machine_gate["official_validation_overlap_count"]
        and not machine_gate["split_question_overlap_count"]
        and not machine_gate["split_source_overlap_count"]
        and max_tokens <= cloze.TRAIN_MAX_TOKENS
    )
    review_rows = _review_sample(fill_by_split)
    review_packet = {
        "fixed_selection": True,
        "human_reviewed": False,
        "row_count": len(review_rows),
        "coverage": {
            "split_counts": dict(
                Counter(str(row["split"]) for row in review_rows)
            ),
            "hop_counts": dict(
                Counter(str(row["search_turns"]) for row in review_rows)
            ),
            "answer_kind_counts": dict(
                Counter(
                    str(candidate["answer_kind"])
                    for row in review_rows
                    for candidate in row["source_candidates"]
                )
            ),
        },
        "rows": review_rows,
    }
    summary = {
        "mode": "qwen_semantic_critic_then_grounded_cloze_rebuild",
        "critic": {
            "model": MODEL_NAME,
            "temperature": 0,
            "minimum_confidence": MIN_CONFIDENCE,
            "json_schema": CRITIC_JSON_SCHEMA,
            "pool_targets": pool_targets,
            "validated_pool_counts": {
                split: len(rows) for split, rows in pools.items()
            },
            "accepted_pool_counts": {
                split: len(rows) for split, rows in clean_pools.items()
            },
            "rejection_counts": _critic_rejection_counts(audits),
        },
        "candidate_extraction": {
            "raw_candidate_count": len(raw_candidates),
            "reason_counts": extraction_reasons,
        },
        "retrieval_rejection_counts": dict(
            sorted((pool_reasons + pair_reasons).items())
        ),
        "output_counts": {
            "train": len(train),
            "validation": len(validation),
        },
        "profiles": {
            "train": cloze._profile(train),
            "validation": cloze._profile(validation),
        },
        "objective_replay": {
            "train_count": len(objective_train),
            "validation_count": len(objective_validation),
            "available": objective_available,
        },
        "post_critic_hard_gate_issue_counts": dict(hard_gate_counts),
        "machine_gate": machine_gate,
        "human_reviewed": False,
        "training_authorized": False,
        "training_submitted": False,
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {
            "train": str(output_dir / "train.jsonl"),
            "validation": str(output_dir / "validation.jsonl"),
            "validated_pool": str(
                output_dir / "validated_candidate_pool.jsonl"
            ),
            "semantic_verdicts": str(
                output_dir / "semantic_verdicts.jsonl"
            ),
            "review_sample": str(output_dir / "review_sample.jsonl"),
            "review_packet": str(output_dir / "review_packet.json"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    _write_jsonl(output_dir / "train.jsonl", train)
    _write_jsonl(output_dir / "validation.jsonl", validation)
    _write_jsonl(output_dir / "review_sample.jsonl", review_rows)
    (output_dir / "review_packet.json").write_text(
        json.dumps(review_packet, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("[cloze-critic-summary]", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    for index_value, row in enumerate(review_rows, start=1):
        print(
            "[cloze-review-row] "
            + json.dumps(
                {
                    "index": index_value,
                    "split": row["split"],
                    "search_turns": row["search_turns"],
                    "query": row["query"],
                    "expected_answer": row["expected_answer"],
                    "sources": row["source_candidates"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
