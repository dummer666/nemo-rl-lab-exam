# grpo_qwen3.5-9b_qa-rl-agent-rerank_wanghaonan

Quality-aware retrieval v2 derived from the retained baseline experiment.
The baseline directory and its step-100 checkpoint remain unchanged.

Changes are intentionally limited to retrieval:

- retrieve 50 BM25 candidates;
- classify chunks as answer-bearing, reference, question-only, or noise;
- rerank candidates with a document-quality prior;
- return four bounded evidence snippets;
- disable the unavailable online short-answer judge.

Semantic retrieval is not enabled in training until
`retrieval_qa_audit_wanghaonan` demonstrates better evidence coverage and
acceptable latency on the server corpus. GRPO, LoRA, KL, optimizer, and
generation settings remain the same to isolate retrieval quality.

```bash
uv run lab validate grpo_qwen3.5-9b_qa-rl-agent-rerank_wanghaonan
```
