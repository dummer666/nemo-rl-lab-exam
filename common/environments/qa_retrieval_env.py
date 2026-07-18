"""Ray/NeMo wrapper for the multi-turn Markdown retrieval QA runner."""

from __future__ import annotations

import os
import sys
from typing import Any

import ray
import torch
from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common.retrieval.markdown_bm25 import MarkdownBM25Index  # noqa: E402
from common.retrieval.qa_agent import (  # noqa: E402
    QARetrievalMetadata,
    QARetrievalRunner,
)


def _last_assistant_text(message_log: LLMMessageLogType) -> str:
    for message in reversed(message_log):
        if message.get("role") == "assistant":
            return str(message.get("content", "")).strip()
    return ""


@ray.remote  # pragma: no cover
class QARetrievalEnv(EnvironmentInterface[QARetrievalMetadata]):
    """Build one local index per actor and serve batched Agent turns."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        config = cfg or {}
        docs_dir = os.environ.get("QA_DOCS_DIR") or config.get("docs_dir", "/data/docs")
        index = MarkdownBM25Index.from_directory(
            docs_dir,
            chunk_chars=int(config.get("chunk_chars", 1200)),
            overlap_chars=int(config.get("chunk_overlap_chars", 160)),
            k1=float(config.get("bm25_k1", 1.5)),
            b=float(config.get("bm25_b", 0.75)),
            quality_weights=dict(config.get("quality_weights", {})),
        )
        print(
            f"[qa-retrieval] indexed {index.num_documents} chunks from {docs_dir}; "
            f"quality={index.quality_category_counts}"
        )

        if bool(config.get("use_judge", True)):
            from common.rewards.qa_judge_reward import qa_judge_reward_fn

            reward_fn = qa_judge_reward_fn
        else:
            from common.rewards.qa_reward import qa_rule_reward_fn

            reward_fn = qa_rule_reward_fn
        self.runner = QARetrievalRunner(
            index,
            reward_fn,
            max_searches=int(config.get("max_searches", 2)),
            max_invalid_actions=int(config.get("max_invalid_actions", 2)),
            top_k=int(config.get("top_k", 3)),
            candidate_k=int(config.get("candidate_k", config.get("top_k", 3))),
            quality_rerank=bool(config.get("quality_rerank", False)),
            max_result_chars=int(config.get("max_result_chars", 1800)),
            per_result_chars=int(config.get("per_result_chars", 520)),
        )

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[QARetrievalMetadata],
    ) -> EnvironmentReturn[QARetrievalMetadata]:
        turns = [
            self.runner.process(_last_assistant_text(log), info)
            for log, info in zip(message_log_batch, metadata, strict=False)
        ]
        return EnvironmentReturn(
            observations=[{"role": "environment", "content": turn.observation} for turn in turns],
            metadata=[turn.metadata for turn in turns],
            next_stop_strings=[turn.stop_strings for turn in turns],
            rewards=torch.tensor(
                [turn.reward for turn in turns],
                dtype=torch.float32,
            ),
            terminateds=torch.tensor(
                [turn.terminated for turn in turns],
                dtype=torch.bool,
            ),
            answers=[turn.answer for turn in turns],
        )

    def shutdown(self):
        pass

    def global_post_process_and_metrics(
        self,
        batch: BatchedDataDict,
    ) -> tuple[BatchedDataDict, dict]:
        rewards = batch.get(
            "total_reward",
            torch.tensor([0.0] * len(batch["idx"])),
        ).float()
        if len(rewards) == 0:
            return batch, {}
        accuracy = rewards.clamp(min=0.0, max=1.0).mean().item()
        perfect_rate = (rewards >= 1.0).float().mean().item()
        return batch, {
            "accuracy": accuracy,
            "qa_mean_reward": rewards.mean().item(),
            "qa_perfect_rate": perfect_rate,
            "qa_format_penalty_rate": (rewards < 0).float().mean().item(),
        }
