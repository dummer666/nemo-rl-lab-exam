# retrieval_qa_qwen_semantic_audit_wanghaonan

High-cost offline diagnostic using the already cached
`Qwen/Qwen3.5-9B-Base` hidden states as semantic representations.

- BM25 candidate set: 20
- semantic batch size: 8
- maximum encoded length: 256
- semantic work: only scorable fill/short validation rows
- no GRPO policy, optimizer, rollout worker, or checkpoint is initialized

This does not assume that a causal LM is a production-quality embedding model.
It only measures whether the available cached model can improve evidence
selection enough to justify a dedicated E5/BGE cache request.
