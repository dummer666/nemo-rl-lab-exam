# qa_fill_retrieval_oracle_wanghaonan

Read-only audit of all 16 official fill questions. It compares the persisted
retrieval-SFT first hop with:

- the same query while excluding every first-hop source;
- answer-free local context queries for each numbered blank;
- acronym-definition and threshold variants derived only from question text.

The report separates fragile one-character/digit substring hits from robust
visible evidence. No model is loaded and no training is submitted.
