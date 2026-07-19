# sft_qwen3.5-9b_qa-fill-v3_wanghaonan

Stronger fill SFT from the preserved merged retrieval-SFT step 50 model.

- 31 one-hop fill exposures.
- 28 two-hop fill exposures from four deterministic passes over 7 verified rows.
- 24 balanced objective replay rows (28.9% of train exposures).
- LoRA rank 8, learning rate `5e-6`, global batch 4.
- Three epochs / 60 optimizer steps.
- Checkpoints and validation at steps 20, 40, and 60.

All source splits, leakage exclusions, and official-validation isolation remain
unchanged.
