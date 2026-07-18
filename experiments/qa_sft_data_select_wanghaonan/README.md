# qa_sft_data_select_wanghaonan

Read-only server preprocessing for multi-turn retrieval SFT.

The selector rechecks every cleaned fill and short-answer row against the
quality-reranked BM25 index and distinguishes:

- `ready_one_search`: all gold keypoints are visible in the actual rendered
  Top-4 observation;
- `needs_query_rewrite`: Top-20 contains every keypoint, but a verified second
  query is still required;
- `partial_review`: only part of the gold answer is grounded;
- `excluded_unsupported`: no gold keypoint is retrievable.

Only the first two groups enter the primary train/validation candidate split.
Partial rows are persisted separately for later review, and unsupported rows
remain only in the audit manifest.

Outputs are written under the run's `sft_selection/` directory:

- `selection_manifest.jsonl`
- `primary_train_candidates.jsonl`
- `primary_validation_candidates.jsonl`
- `secondary_review_candidates.jsonl`
- `summary.json`
