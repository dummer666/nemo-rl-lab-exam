# sft_qwen3.5-9b_qa-fill-v2_wanghaonan

User-authorized bounded fill SFT pilot.

- Starts from the merged retrieval-SFT step 50 Hugging Face model.
- Trains on 38 fill trajectories, including 7 true two-hop examples.
- Replays 15 objective questions, balanced across single/multiple/bool.
- Uses LoRA rank 8, learning rate `8e-6`, global batch 4, and one epoch.
- Preserves a separate 5-fill validation set and 7-fill RL holdout.
- Uses passthrough rendering and assistant-only loss on every assistant turn.

Run the smoke experiment first. The full run is allowed only after smoke success.
