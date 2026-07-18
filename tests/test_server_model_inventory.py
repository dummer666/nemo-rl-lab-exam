from __future__ import annotations

import json

from experiments.server_model_inventory_wanghaonan.run import (
    _model_id_from_cache_dir,
    _scan_hf_hub,
    _scan_nemo_checkpoints,
)


def test_model_id_from_huggingface_cache_directory():
    assert _model_id_from_cache_dir("models--intfloat--multilingual-e5-small") == (
        "intfloat/multilingual-e5-small"
    )
    assert _model_id_from_cache_dir("not-a-model") is None


def test_scan_hf_hub_reads_config_and_artifacts(tmp_path):
    snapshot = (
        tmp_path
        / "models--BAAI--bge-small-zh-v1.5"
        / "snapshots"
        / "revision-one"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bert",
                "architectures": ["BertModel"],
                "hidden_size": 512,
                "num_hidden_layers": 12,
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")

    records = _scan_hf_hub(tmp_path)

    assert len(records) == 1
    assert records[0]["model_id"] == "BAAI/bge-small-zh-v1.5"
    assert records[0]["roles"] == ["embedding"]
    assert records[0]["has_config"]
    assert records[0]["has_tokenizer"]
    assert records[0]["has_weights"]


def test_scan_nemo_checkpoints_detects_converted_model(tmp_path):
    checkpoint = tmp_path / "Qwen" / "Qwen3.5-9B-Base" / "iter_0000000"
    checkpoint.mkdir(parents=True)

    records = _scan_nemo_checkpoints(tmp_path)

    assert records == [
        {
            "model_id": "Qwen/Qwen3.5-9B-Base",
            "path": str(checkpoint.parent),
            "iterations": ["iter_0000000"],
            "has_checkpoint": True,
        }
    ]
