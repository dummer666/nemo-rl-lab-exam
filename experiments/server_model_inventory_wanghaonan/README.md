# server_model_inventory_wanghaonan

Read-only inventory of model artifacts available inside the training container.
It scans Hugging Face snapshot directories and NeMo converted checkpoints,
reads configuration metadata, identifies possible embedding/reranker models,
and writes `model_inventory/model_inventory.json` under the shared run output.

The script does not load weights or modify caches. Run it after the active
training job releases the single concurrent job slot.
