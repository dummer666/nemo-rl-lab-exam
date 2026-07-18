"""Optional transformer reranking for bounded BM25 candidate sets."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from common.retrieval.markdown_bm25 import SearchResult


def rerank_by_semantic(
    candidates: Sequence[SearchResult],
    semantic_scores: Sequence[float],
    *,
    apply_quality_weight: bool = False,
) -> list[SearchResult]:
    """Sort candidates by dense similarity, optionally applying document quality."""
    if len(candidates) != len(semantic_scores):
        raise ValueError("semantic score count must match candidate count")
    reranked = []
    for result, semantic_score in zip(candidates, semantic_scores, strict=True):
        score = float(semantic_score)
        if apply_quality_weight:
            score *= result.quality_weight
        reranked.append(replace(result, score=score))
    return sorted(
        reranked,
        key=lambda result: (-result.score, -float(result.raw_score or 0.0), result.source),
    )


def reciprocal_rank_fusion(
    candidates: Sequence[SearchResult],
    semantic_scores: Sequence[float],
    *,
    rank_constant: float = 60.0,
    apply_quality_weight: bool = True,
) -> list[SearchResult]:
    """Fuse BM25 and semantic ranks while retaining a non-destructive quality prior."""
    if len(candidates) != len(semantic_scores):
        raise ValueError("semantic score count must match candidate count")
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")

    semantic_order = sorted(
        range(len(candidates)),
        key=lambda index: (-float(semantic_scores[index]), index),
    )
    semantic_rank = {index: rank for rank, index in enumerate(semantic_order, start=1)}
    fused = []
    for index, result in enumerate(candidates):
        bm25_rank = index + 1
        score = 1.0 / (rank_constant + bm25_rank)
        score += 1.0 / (rank_constant + semantic_rank[index])
        if apply_quality_weight:
            score *= result.quality_weight
        fused.append(replace(result, score=score))
    return sorted(
        fused,
        key=lambda result: (-result.score, -float(result.raw_score or 0.0), result.source),
    )


class TransformerSemanticReranker:
    """Mean-pooled Hugging Face encoder loaded only by the server-side audit."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        *,
        device: str = "auto",
        batch_size: int = 64,
        max_length: int = 512,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        local_files_only: bool = True,
    ):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - cluster dependency
            raise RuntimeError(
                "semantic audit requires torch and transformers in the cluster environment"
            ) from exc

        if batch_size < 1 or max_length < 8:
            raise ValueError("batch_size and max_length must be positive")
        resolved_device = device
        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

        self._torch = torch
        self.device = resolved_device
        self.batch_size = batch_size
        self.max_length = max_length
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        model_kwargs = {
            "local_files_only": local_files_only,
            "low_cpu_mem_usage": True,
        }
        if self.device.startswith("cuda"):
            model_kwargs["torch_dtype"] = torch.float16
        self.model = (
            AutoModel.from_pretrained(
                model_name,
                **model_kwargs,
            )
            .eval()
            .to(self.device)
        )

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        torch = self._torch
        embeddings = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            with torch.inference_mode():
                output = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            embeddings.extend(pooled.cpu().tolist())
        return embeddings

    def score(self, query: str, candidates: Sequence[SearchResult]) -> list[float]:
        if not candidates:
            return []
        query_embedding = self._encode([self.query_prefix + query])[0]
        passages = [
            self.passage_prefix + f"{result.heading}\n{result.text}"
            for result in candidates
        ]
        passage_embeddings = self._encode(passages)
        return [
            sum(left * right for left, right in zip(query_embedding, passage, strict=True))
            for passage in passage_embeddings
        ]
