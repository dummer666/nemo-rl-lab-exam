# qa_fill_sft_v3_trajectory_audit_wanghaonan

Read-only audit of the persisted retrieval-SFT baseline and stronger fill-SFT
step 20/40/60 trajectories. It deterministically replays both BM25 hops against
the same 421,976-chunk index and distinguishes:

- duplicate or redundant second queries;
- new sources without new gold evidence;
- trusted visible incremental keypoint evidence;
- evidence found but not used in the boxed answer;
- reward gains and regressions.

It loads no language model and performs no training.
