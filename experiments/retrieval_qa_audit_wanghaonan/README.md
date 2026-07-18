# retrieval_qa_audit_wanghaonan

Server-side retrieval audit over `/data/docs` and `/data/datasets/qa_rl/val.jsonl`.
It compares:

- current BM25;
- BM25 Top-50 with document-quality reranking;
- transformer semantic reranking over the BM25 candidates;
- BM25/semantic reciprocal-rank fusion with the quality prior.

Only fill and short-answer records contribute to evidence coverage because an
objective question-only document already contains every option text. The report
includes evidence coverage/full-coverage at 3 and 20, question-only Top-1 rates,
corpus quality categories, representative improvements, and timing.

Default semantic model: `intfloat/multilingual-e5-small`. Optional environment
variables:

```text
QA_AUDIT_CANDIDATE_K=50
QA_AUDIT_MAX_ROWS=0
QA_AUDIT_SEMANTIC=1
QA_SEMANTIC_MODEL=intfloat/multilingual-e5-small
QA_SEMANTIC_LOCAL_ONLY=1
QA_SEMANTIC_BATCH_SIZE=64
QA_SEMANTIC_MAX_LENGTH=512
```

The cluster disables outbound Hugging Face traffic. If the requested encoder is
not cached, the audit records the error and available cache names, then still
completes the BM25 and quality-reranking comparison.

Run through the normal cluster service:

```bash
uv run lab validate retrieval_qa_audit_wanghaonan
uv run lab submit retrieval_qa_audit_wanghaonan
```
