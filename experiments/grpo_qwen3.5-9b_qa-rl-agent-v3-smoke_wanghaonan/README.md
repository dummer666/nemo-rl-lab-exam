# grpo_qwen3.5-9b_qa-rl-agent-v3-smoke_wanghaonan

This independent 30-step smoke experiment tests whether retrieval can be
preserved without changing the retained 0.7384 baseline.

- Reads the non-destructive `clean_train.jsonl` artifact.
- Steps 1-10 use two objective, one supported fill, and one supported short
  prompt. The two open prompts must search before answering.
- Steps 11-30 use three objective prompts and one alternating fill/short prompt.
- Training searches receive `0.10 * new_evidence_coverage - 0.01`; an exact
  repeated query costs another `0.02`.
- Validation runs in a separate environment with all shaping disabled.
- Quality-aware BM25 reranking is enabled; the failed Qwen hidden-state
  semantic reranker and unavailable online judge remain disabled.

The smoke keeps baseline LoRA, KL, optimizer, generation, and batch settings.
It should only be extended to 120 steps if search behavior survives and
validation does not regress materially.

```bash
uv run lab validate grpo_qwen3.5-9b_qa-rl-agent-v3-smoke_wanghaonan
uv run lab submit grpo_qwen3.5-9b_qa-rl-agent-v3-smoke_wanghaonan
```
