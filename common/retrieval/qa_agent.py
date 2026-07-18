"""Pure-Python state machine for the multi-turn retrieval QA protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypedDict

from common.retrieval.markdown_bm25 import (
    MarkdownBM25Index,
    build_retrieval_query,
    extract_search_query,
    format_search_results,
)
from common.rewards.qa_reward import FORMAT_PENALTY, extract_boxed


class QARetrievalMetadata(TypedDict, total=False):
    expected_answer: str
    query: str
    bank: str
    search_count: int
    search_queries: list[str]
    invalid_count: int


RewardFn = Callable[..., list[float]]


@dataclass(frozen=True)
class AgentTurn:
    observation: str
    reward: float
    terminated: bool
    stop_strings: list[str] | None
    metadata: QARetrievalMetadata | None
    answer: str | None


class QARetrievalRunner:
    """Process search/final-answer actions without Ray or NeMo dependencies."""

    def __init__(
        self,
        index: MarkdownBM25Index,
        reward_fn: RewardFn,
        *,
        max_searches: int = 2,
        max_invalid_actions: int = 2,
        top_k: int = 3,
        max_result_chars: int = 1800,
        per_result_chars: int = 520,
    ):
        if max_searches < 1:
            raise ValueError("max_searches must be positive")
        if max_invalid_actions < 1:
            raise ValueError("max_invalid_actions must be positive")
        self.index = index
        self.reward_fn = reward_fn
        self.max_searches = max_searches
        self.max_invalid_actions = max_invalid_actions
        self.top_k = top_k
        self.max_result_chars = max_result_chars
        self.per_result_chars = per_result_chars

    def _invalid_action(
        self,
        metadata: QARetrievalMetadata,
        reason: str,
    ) -> AgentTurn:
        next_metadata = dict(metadata)
        invalid_count = int(next_metadata.get("invalid_count", 0)) + 1
        next_metadata["invalid_count"] = invalid_count
        searches_exhausted = int(metadata.get("search_count", 0)) >= self.max_searches
        if invalid_count >= self.max_invalid_actions or searches_exhausted:
            ending = "已无剩余检索回合" if searches_exhausted else "连续格式错误"
            observation = (
                f"[格式错误] {reason}\n{ending}，回合结束。"
                r"检索须用 <search>关键词</search>，最终答案须用 \boxed{...}。"
            )
            return AgentTurn(
                observation=observation,
                reward=FORMAT_PENALTY,
                terminated=True,
                stop_strings=None,
                metadata=None,
                answer=None,
            )
        observation = (
            f"[格式提示] {reason}\n"
            r"请只选择一个动作：<search>关键词</search>；"
            r"或给出分析并以 \boxed{...} 提交最终答案。"
        )
        return AgentTurn(
            observation=observation,
            reward=0.0,
            terminated=False,
            stop_strings=["</search>"],
            metadata=next_metadata,
            answer=None,
        )

    def process(
        self,
        response: str,
        metadata: QARetrievalMetadata,
    ) -> AgentTurn:
        expected = str(metadata.get("expected_answer", ""))
        original_query = str(metadata.get("query", ""))

        if extract_boxed(response) is not None:
            reward = float(
                self.reward_fn(
                    [original_query],
                    [response],
                    [expected],
                )[0]
            )
            return AgentTurn(
                observation=f"[最终答案已提交] reward={reward:.3f}",
                reward=reward,
                terminated=True,
                stop_strings=None,
                metadata=None,
                answer=expected,
            )

        search_query = extract_search_query(response)
        if search_query is None:
            return self._invalid_action(metadata, "未检测到检索动作或最终 boxed 答案")
        if not search_query:
            return self._invalid_action(metadata, "检索关键词为空")

        search_count = int(metadata.get("search_count", 0))
        if search_count >= self.max_searches:
            observation = (
                f"[检索次数已用完] 最多允许 {self.max_searches} 次检索。"
                r"本轮应提交 \boxed{...}。"
            )
            return AgentTurn(
                observation=observation,
                reward=FORMAT_PENALTY,
                terminated=True,
                stop_strings=None,
                metadata=None,
                answer=None,
            )

        retrieval_query = build_retrieval_query(
            search_query,
            original_query,
            str(metadata.get("bank", "")),
        )
        results = self.index.search(retrieval_query, top_k=self.top_k)
        rendered = format_search_results(
            results,
            retrieval_query,
            max_chars=self.max_result_chars,
            per_result_chars=self.per_result_chars,
        )

        next_metadata = dict(metadata)
        next_count = search_count + 1
        next_metadata["search_count"] = next_count
        next_metadata["invalid_count"] = 0
        next_metadata["search_queries"] = [
            *list(metadata.get("search_queries", [])),
            search_query,
        ]
        remaining = self.max_searches - next_count
        if remaining:
            guidance = (
                f"\n\n还可检索 {remaining} 次。证据不足时换更具体的关键词继续检索；"
                r"证据足够时提交 \boxed{...}。"
            )
        else:
            guidance = r"\n\n检索次数已用完，下一轮必须提交 \boxed{...}。"
        return AgentTurn(
            observation=rendered + guidance,
            reward=0.0,
            terminated=False,
            stop_strings=["</search>"],
            metadata=next_metadata,
            answer=None,
        )
