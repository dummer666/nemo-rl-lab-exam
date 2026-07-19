#!/usr/bin/env python
"""Train the bounded fill SFT pilot from the merged retrieval-SFT model."""

from __future__ import annotations

import json
import os
import runpy
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

MODEL_PATH = Path(
    "/shared/outputs/wanghaonan/sft_qwen3.5-9b_qa-retrieval-v1_wanghaonan/"
    "sft_qwen3.5-9b_qa-retrieval-v1_wanghaonan-wanghaonan-20260718-130828/"
    "hf_export/step_50"
)
TRAIN_PATH = Path(
    "/shared/outputs/wanghaonan/qa_fill_sft_pilot_data_wanghaonan/"
    "qa_fill_sft_pilot_data_wanghaonan-wanghaonan-20260719-032625/"
    "fill_sft_pilot_data/fill_pilot_train.jsonl"
)
VALIDATION_PATH = TRAIN_PATH.with_name("fill_pilot_validation.jsonl")
PASSTHROUGH_CHAT_TEMPLATE = (
    "{% for message in messages %}{{ message['content'] }}{% endfor %}"
)
MAX_SEQUENCE_LENGTH = 3072
EXPECTED_PROFILES = {
    "train": {0: 15, 1: 31, 2: 7},
    "validation": {0: 6, 1: 4, 2: 1},
}


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


def _messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("SFT row is missing messages")
    normalized = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("SFT message must be an object")
        role = str(message.get("role", ""))
        content = message.get("content")
        if role not in {"user", "assistant", "environment"}:
            raise ValueError(f"invalid SFT role: {role}")
        if not isinstance(content, str) or not content:
            raise ValueError("SFT message content must be non-empty text")
        normalized.append({"role": role, "content": content})
    roles = [message["role"] for message in normalized]
    expected = [
        "user",
        *[
            "assistant" if index % 2 else "environment"
            for index in range(1, len(roles))
        ],
    ]
    if roles != expected or roles[-1] != "assistant":
        raise ValueError(f"invalid runtime role sequence: {roles}")
    return normalized


def dataset_profile(rows: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    counts = Counter(
        sum(message["role"] == "environment" for message in _messages(row))
        for row in rows
    )
    return dict(sorted(counts.items()))


def select_preflight_samples(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_turns: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}
    for row in rows:
        turns = sum(
            message["role"] == "environment"
            for message in _messages(row)
        )
        if turns in by_turns:
            by_turns[turns].append(dict(row))
    if len(by_turns[2]) < 2 or len(by_turns[1]) < 2 or not by_turns[0]:
        raise ValueError("preflight needs two two-hop, two one-hop, and one objective row")
    return [
        *by_turns[2][:2],
        *by_turns[1][:2],
        by_turns[0][0],
    ]


def _drop_unused_generation_overrides() -> None:
    generation_overrides = [
        argument
        for argument in sys.argv[1:]
        if argument.startswith("policy.generation.")
    ]
    if not generation_overrides:
        return
    sys.argv = [
        sys.argv[0],
        *[
            argument
            for argument in sys.argv[1:]
            if not argument.startswith("policy.generation.")
        ],
    ]
    print(
        "[fill-sft-preflight] ignored unused generation overrides: "
        + ", ".join(generation_overrides),
        flush=True,
    )


def _run_sft_preflight() -> None:
    required = [
        MODEL_PATH / "config.json",
        MODEL_PATH / "tokenizer_config.json",
        TRAIN_PATH,
        VALIDATION_PATH,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing fill SFT inputs: {missing}")
    if not list(MODEL_PATH.glob("*.safetensors")):
        raise FileNotFoundError(f"merged SFT weights missing from {MODEL_PATH}")

    train_rows = _read_jsonl(TRAIN_PATH)
    validation_rows = _read_jsonl(VALIDATION_PATH)
    profiles = {
        "train": dataset_profile(train_rows),
        "validation": dataset_profile(validation_rows),
    }
    if profiles != EXPECTED_PROFILES:
        raise RuntimeError(
            f"fill SFT data profile changed: expected={EXPECTED_PROFILES}, actual={profiles}"
        )

    import torch
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.data.interfaces import TaskDataSpec
    from nemo_rl.data.llm_message_utils import (
        add_loss_mask_to_message_log,
        get_formatted_message_log,
    )

    minimum_free_gib = float(
        os.environ.get("QA_FILL_SFT_MIN_FREE_GPU_GIB", "48")
    )
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except RuntimeError as error:
        raise RuntimeError(
            "shared H200 cannot initialize CUDA; retry after competing work exits"
        ) from error
    gib = 1024**3
    free_gib = free_bytes / gib
    total_gib = total_bytes / gib
    print(
        f"[fill-sft-preflight] gpu free={free_gib:.2f} GiB "
        f"total={total_gib:.2f} GiB required={minimum_free_gib:.2f} GiB",
        flush=True,
    )
    if free_gib < minimum_free_gib:
        raise RuntimeError(
            f"shared H200 is occupied: {free_gib:.2f} GiB free "
            f"< {minimum_free_gib:.2f} GiB required"
        )

    tokenizer = get_tokenizer(
        {
            "name": str(MODEL_PATH),
            "chat_template": PASSTHROUGH_CHAT_TEMPLATE,
            "chat_template_kwargs": None,
        }
    )
    summaries = []
    for sample_index, row in enumerate(select_preflight_samples(train_rows)):
        messages = _messages(row)
        formatted = get_formatted_message_log(
            messages,
            tokenizer,
            TaskDataSpec(task_name="qa-fill-sft-v2"),
            add_bos_token=False,
            add_eos_token=True,
            add_generation_prompt=False,
        )
        add_loss_mask_to_message_log(
            [formatted],
            roles_to_train_on=["assistant"],
            only_unmask_final=False,
        )
        expected_chunks = []
        for message_index, message in enumerate(messages):
            content = message["content"]
            if (
                message_index == len(messages) - 1
                and tokenizer.eos_token is not None
                and not content.rstrip("\n").endswith(tokenizer.eos_token)
            ):
                content += tokenizer.eos_token
            expected_chunks.append(
                tokenizer(
                    content,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"][0]
            )
        if len(formatted) != len(messages):
            raise RuntimeError("SFT preflight changed the number of turns")
        for message, actual, expected_ids in zip(
            messages,
            formatted,
            expected_chunks,
            strict=True,
        ):
            if not torch.equal(actual["token_ids"], expected_ids):
                raise RuntimeError(
                    f"passthrough rendering changed role={message['role']}"
                )
            expected_mask = 1 if message["role"] == "assistant" else 0
            if not torch.all(actual["token_loss_mask"] == expected_mask):
                raise RuntimeError(
                    f"incorrect loss mask for role={message['role']}"
                )
        token_length = sum(len(message["token_ids"]) for message in formatted)
        if token_length > MAX_SEQUENCE_LENGTH:
            raise RuntimeError(
                f"preflight length {token_length} exceeds {MAX_SEQUENCE_LENGTH}"
            )
        summaries.append(
            {
                "sample": sample_index,
                "environment_turns": sum(
                    message["role"] == "environment"
                    for message in messages
                ),
                "token_length": token_length,
                "assistant_tokens": sum(
                    int(message["token_loss_mask"].sum())
                    for message in formatted
                ),
            }
        )
    print(
        "[fill-sft-preflight] exact chunks and assistant-only masks verified",
        flush=True,
    )
    print(
        json.dumps(
            {"profiles": profiles, "samples": summaries},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    _run_sft_preflight()
    _drop_unused_generation_overrides()
    runpy.run_path("examples/run_sft.py", run_name="__main__")
