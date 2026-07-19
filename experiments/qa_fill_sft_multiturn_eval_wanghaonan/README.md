# qa_fill_sft_multiturn_eval_wanghaonan

Greedy, deterministic evaluation of both the preserved retrieval-SFT step 50
baseline and the new fill-SFT step 13 model on all 313 official validation
questions using the current production BM25 multi-turn environment.

The comparison reports per-type reward/perfect counts, retrieval rate, one/two
search counts, and protocol errors. It is the promotion gate before any GRPO.
