# qa_fill_only_audit_wanghaonan

Read-only reconstruction and audit of every unique fill question in the v1
trajectory manifest.

The experiment:

- reuses the joint SFT v2 pack's `_fill_trajectory` implementation unchanged;
- performs real BM25 Top-4 retrieval with at most two searches;
- requires trusted visible evidence, leak-free queries, valid protocol, and a
  maximum 6000-token runtime trajectory;
- excludes all 313 official-validation fingerprints;
- preserves source splits and rejects cross-split or conflicting duplicates;
- writes full accepted/rejected diagnostics and a review set containing every
  two-hop result plus a deterministic random sample of 20 other results.
- reports the isolated single/multiple/bool replay capacity and the balanced
  25%-35% selection size as informational metadata only.
- safely probes the injected Judge endpoint without emitting its URL or API
  key, and records `/models` status/latency/model IDs plus one synthetic score;
- when that probe succeeds, re-scores the existing 22 step-50 short
  completions beside their legacy keyword rewards. Probe failure is recorded
  as a sanitized fallback cause and does not stop the fill audit.

It does not read short-answer rebuild outputs, bypass the joint pack gate, or
start training. Even when the machine gate passes, every record remains
`human_reviewed=false`.

The machine gate requires at least 40 accepted train questions, 8 validation
questions, and 8 RL-holdout questions, with no split or official-validation
fingerprint overlap. Human review remains mandatory after that gate.

```bash
uv run lab validate qa_fill_only_audit_wanghaonan
uv run lab submit qa_fill_only_audit_wanghaonan
```
