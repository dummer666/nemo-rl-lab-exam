# qa_sft_checkpoint_export_wanghaonan

Merges the epoch-one and epoch-two Megatron LoRA checkpoints from
`sft_qwen3.5-9b_qa-retrieval-v1_wanghaonan` into standalone Hugging Face
models for end-to-end retrieval evaluation.

The platform's generic post-training exporter handles full Megatron and
DTensor checkpoints but does not provide the base checkpoint required by
Megatron LoRA. This experiment invokes NeMo-RL's official
`convert_lora_to_hf.py` with the cached Qwen3.5-9B Megatron base.
