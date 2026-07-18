from __future__ import annotations

import json
import runpy
from pathlib import Path

import torch

MODEL_NAME = "Qwen/Qwen3.5-9B"
PASSTHROUGH_CHAT_TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
TRAIN_PATH = Path(
    "/shared/outputs/wanghaonan/qa_sft_trajectory_build_wanghaonan/"
    "qa_sft_trajectory_build_wanghaonan-wanghaonan-20260718-123640/"
    "sft_trajectories/sft_train.jsonl"
)
MAX_SEQUENCE_LENGTH = 3072


def _load_retrieval_samples(path: Path, count: int = 5) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    selected: list[dict] = []
    for minimum_environment_turns in (2, 2, 1, 1, 1):
        for row in rows:
            if row in selected:
                continue
            environment_turns = sum(message["role"] == "environment" for message in row["messages"])
            if environment_turns >= minimum_environment_turns:
                selected.append(row)
                break

    if len(selected) != count:
        raise RuntimeError(f"Expected {count} retrieval samples for preflight, found {len(selected)}")
    return selected


def _run_sft_preflight() -> None:
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.data.interfaces import TaskDataSpec
    from nemo_rl.data.llm_message_utils import (
        add_loss_mask_to_message_log,
        get_formatted_message_log,
    )

    tokenizer = get_tokenizer(
        {
            "name": MODEL_NAME,
            "chat_template": PASSTHROUGH_CHAT_TEMPLATE,
            "chat_template_kwargs": None,
        }
    )
    samples = _load_retrieval_samples(TRAIN_PATH)

    summaries = []
    for sample_index, sample in enumerate(samples):
        messages = sample["messages"]
        formatted = get_formatted_message_log(
            messages,
            tokenizer,
            TaskDataSpec(task_name="qa-retrieval-sft"),
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
            raise RuntimeError("SFT preflight changed the number of conversation turns")

        for message, actual, expected_ids in zip(messages, formatted, expected_chunks, strict=True):
            if not torch.equal(actual["token_ids"], expected_ids):
                raise RuntimeError(f"Passthrough rendering inserted or removed tokens for role={message['role']}")
            expected_mask_value = 1 if message["role"] == "assistant" else 0
            if not torch.all(actual["token_loss_mask"] == expected_mask_value):
                raise RuntimeError(f"Incorrect token loss mask for role={message['role']}")

        token_length = sum(len(message["token_ids"]) for message in formatted)
        if token_length > MAX_SEQUENCE_LENGTH:
            raise RuntimeError(f"Preflight sample length {token_length} exceeds {MAX_SEQUENCE_LENGTH}")

        environment_turns = sum(message["role"] == "environment" for message in messages)
        if environment_turns == 0:
            raise RuntimeError("Preflight sample does not contain retrieval observations")

        summaries.append(
            {
                "sample": sample_index,
                "roles": [message["role"] for message in messages],
                "environment_turns": environment_turns,
                "token_length": token_length,
                "assistant_tokens": sum(int(message["token_loss_mask"].sum()) for message in formatted),
            }
        )

    print("[sft-preflight] exact runtime chunks and assistant-only masks verified")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _run_sft_preflight()
    runpy.run_path("examples/run_sft.py", run_name="__main__")
