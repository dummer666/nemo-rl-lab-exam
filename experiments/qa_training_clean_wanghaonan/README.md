# qa_training_clean_wanghaonan

Non-destructive server preprocessing for `/data/datasets/qa_rl/train.jsonl`.

Outputs:

- `clean_manifest.jsonl`: one record per original row with issues, duplicate
  metadata, evidence support, and sampling weight;
- `clean_train.jsonl`: structurally valid canonical rows with `_clean` metadata;
- `summary.json`: counts, timing, and persistent paths.

The original dataset is read-only. Objective questions receive structural
validation; fill and short answers additionally use quality-reranked BM25
Top-20 evidence coverage. Unsupported open questions are downweighted rather
than deleted.
