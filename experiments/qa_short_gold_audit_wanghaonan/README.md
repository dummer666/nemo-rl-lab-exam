# qa_short_gold_audit_wanghaonan

Read-only audit of every remaining training short label, official-validation
short label, verified SFT trajectory, deterministic SFT step-50 short rollout,
and sampled short-GRPO validation rollout.

The job preserves the original rule reward and writes:

- `short_gold_audit.jsonl`: source row, full gold, baseline query, Top-20 source
  text, answer-bearing hits, label defects, and reward attacks;
- `short_trajectory_audit.jsonl`: every real search hop, Top-20 source text,
  visible evidence, completion, rule matches, logged/recomputed reward, and
  failure attribution;
- `rebuild_candidates.jsonl`: strict evidence-complete candidates that still
  require a reconstructed 2-6 point target;
- `rejected_short_labels.jsonl`: labels rejected by the conservative audit;
- `representative_examples.jsonl`: 10-20 compact human-review examples;
- `summary.json`: counts, rates, evidence/failure cross tables, and the required
  step-20 sample-180 reward-hacking regression.

No row is marked SFT-v2-ready until a complete target has been rebuilt and each
answer point is bound to visible evidence.
