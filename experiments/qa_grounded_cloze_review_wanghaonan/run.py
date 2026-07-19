#!/usr/bin/env python
"""Build a compact review packet for grounded cloze trajectories."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from common.retrieval.evidence import normalize_evidence_text
from experiments.qa_grounded_cloze_data_wanghaonan.run import (
    candidate_quality_issues,
)

PACK_ROOT = Path(
    "/shared/outputs/wanghaonan/qa_grounded_cloze_data_wanghaonan/"
    "qa_grounded_cloze_data_wanghaonan-wanghaonan-20260719-061248/"
    "grounded_cloze"
)
_BAD_ACRONYM = {
    "AND",
    "ARE",
    "CAN",
    "FOR",
    "FROM",
    "HAS",
    "NOT",
    "THE",
    "THIS",
    "USE",
    "WITH",
}
_OCR_NOISE = re.compile(r"[\ufffd]|(?:[_=-]){4,}")


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def _output_dir(overrides: Sequence[str]) -> Path:
    for override in overrides:
        if override.startswith("logger.log_dir="):
            output = Path(override.split("=", 1)[1]).parent / "cloze_review"
            output.mkdir(parents=True, exist_ok=True)
            return output
    output = Path(__file__).resolve().parent / "outputs"
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


def review_issues(row: Mapping[str, Any]) -> list[str]:
    issues = []
    normalized_query = normalize_evidence_text(str(row["query"]))
    expected = str(row["expected_answer"]).split("]", 1)[-1]
    answers = [answer.strip() for answer in expected.split("|||")]
    if any(
        normalize_evidence_text(answer) in normalized_query
        for answer in answers
        if normalize_evidence_text(answer)
    ):
        issues.append("answer_visible_in_question")

    candidates = row.get("source_candidates", [])
    if len(candidates) != int(row["search_turns"]):
        issues.append("candidate_hop_mismatch")
    for candidate in candidates:
        answer = str(candidate["answer"])
        kind = str(candidate["answer_kind"])
        sentence = str(candidate["sentence"])
        masked = str(candidate["masked_sentence"])
        model_query = str(candidate["model_query"])
        if answer not in sentence:
            issues.append("answer_missing_from_source_sentence")
        if answer in masked or answer in model_query:
            issues.append("answer_leak")
        if "【1】" not in masked:
            issues.append("missing_mask")
        if _OCR_NOISE.search(sentence):
            issues.append("ocr_noise")
        if kind == "acronym" and answer.upper() in _BAD_ACRONYM:
            issues.append("generic_english_token")
        if len(sentence) < 28:
            issues.append("sentence_fragment")
        issues.extend(candidate_quality_issues(sentence, answer, kind))
    return sorted(set(issues))


def _compact(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "split": str(row["split"]),
        "search_turns": int(row["search_turns"]),
        "query": str(row["query"]),
        "expected_answer": str(row["expected_answer"]),
        "final_response": str(row["messages"][-1]["content"]),
        "search_queries": [
            str(hop["model_search_query"])
            for hop in row["search_hops"]
        ],
        "sources": [
            {
                "source": str(candidate["source"]),
                "heading": str(candidate["heading"]),
                "sentence": str(candidate["sentence"]),
                "masked_sentence": str(candidate["masked_sentence"]),
                "answer": str(candidate["answer"]),
                "answer_kind": str(candidate["answer_kind"]),
            }
            for candidate in row["source_candidates"]
        ],
        "issues": review_issues(row),
    }


def main() -> None:
    _, overrides = _parse_args()
    output_dir = _output_dir(overrides)
    sample_path = PACK_ROOT / "review_sample.jsonl"
    if not sample_path.is_file():
        raise FileNotFoundError(f"missing grounded cloze review sample: {sample_path}")
    rows = _read_jsonl(sample_path)
    if len(rows) != 28:
        raise RuntimeError(f"expected 28 review rows, got {len(rows)}")
    compact = [_compact(row) for row in rows]
    issue_counts = Counter(
        issue for row in compact for issue in row["issues"]
    )
    answer_kind_counts = Counter(
        source["answer_kind"]
        for row in compact
        for source in row["sources"]
    )
    packet = {
        "source": str(sample_path),
        "row_count": len(compact),
        "one_hop_count": sum(row["search_turns"] == 1 for row in compact),
        "two_hop_count": sum(row["search_turns"] == 2 for row in compact),
        "answer_kind_counts": dict(sorted(answer_kind_counts.items())),
        "machine_review_issue_counts": dict(sorted(issue_counts.items())),
        "machine_review_passed": not issue_counts,
        "human_reviewed": False,
        "rows": compact,
    }
    output_path = output_dir / "review_packet.json"
    output_path.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("[grounded-cloze-review]", flush=True)
    print(json.dumps(packet, ensure_ascii=False, indent=2), flush=True)
    print(f"[grounded-cloze-review] saved={output_path}", flush=True)


if __name__ == "__main__":
    main()
