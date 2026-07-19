# sft_qwen3.5-9b_qa-grounded-cloze_wanghaonan

One conservative epoch from the merged retrieval-SFT step 50 model.

- 160 grounded one-hop fill trajectories.
- 120 grounded incremental two-hop fill trajectories.
- 96 balanced objective replay trajectories.
- Fresh LoRA rank 8 / alpha 16 at learning rate `3e-6`.
- Global batch 4, maximum length 3072, and 94 optimizer steps.
- Validation and checkpoints around steps 31, 62, and 93.

The semantic critic only filters candidates. Gold answers remain exact reference
spans, and every retained trajectory passes the deterministic retrieval and
split-isolation gates.
