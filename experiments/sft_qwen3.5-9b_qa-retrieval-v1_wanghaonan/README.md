# sft_qwen3.5-9b_qa-retrieval-v1_wanghaonan

## Goal

Cold-start Qwen3.5-9B Instruct on evidence-grounded retrieval behavior before
a short GRPO run. The SFT and later RL question IDs are disjoint.

## Data

- 141 verified one-search trajectories.
- 33 verified two-search trajectories from eight-way query rejection sampling.
- 72 objective-question retention samples in train and 12 in validation.
- Final split: 203 train, 29 validation, and 26 reserved RL holdout rows.
- Maximum audited length: 2519 tokens.

The source artifacts are under
`/shared/outputs/wanghaonan/qa_sft_trajectory_build_wanghaonan/qa_sft_trajectory_build_wanghaonan-wanghaonan-20260718-123640/sft_trajectories/`.

## Training

- Qwen3.5-9B Instruct, Megatron PEFT LoRA rank 16 / alpha 32.
- Passthrough chat template and assistant-only loss over all assistant turns.
- Global batch 8, sequence cap 3072, learning rate 2e-5, two epochs.
- Validation and checkpointing at the end of each 25-step epoch.
- `run.py` blocks training unless five real retrieval rows preserve exact raw
  chunks, visible environment observations, and assistant-only token masks.
