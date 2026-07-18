#!/usr/bin/env python
"""Inventory locally cached server models without loading their weights."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Sequence

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
_MODEL_NAME = re.compile(r"model_name:\s*[\"']?([^\"'\s#]+)")
_EMBEDDING_HINTS = ("bge", "e5", "gte", "embed", "jina", "sentence-transform")
_RERANKER_HINTS = ("rerank", "cross-encoder")


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Inventory cached server models")
    parser.add_argument("--config")
    return parser.parse_known_args()


def _output_dir(overrides: Sequence[str]) -> Path:
    for override in overrides:
        if override.startswith("logger.log_dir="):
            output = Path(override.split("=", 1)[1]).parent / "model_inventory"
            output.mkdir(parents=True, exist_ok=True)
            return output
    output = THIS_DIR / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _model_id_from_cache_dir(name: str) -> str | None:
    if not name.startswith("models--"):
        return None
    parts = name.removeprefix("models--").split("--", 1)
    return "/".join(parts) if len(parts) == 2 else None


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _model_roles(model_id: str, config: dict) -> list[str]:
    text = " ".join(
        [
            model_id,
            str(config.get("model_type", "")),
            " ".join(map(str, config.get("architectures", []))),
        ]
    ).lower()
    roles = []
    if any(hint in text for hint in _EMBEDDING_HINTS):
        roles.append("embedding")
    if any(hint in text for hint in _RERANKER_HINTS):
        roles.append("reranker")
    if not roles:
        roles.append("generative_or_unknown")
    return roles


def _snapshot_record(model_id: str, snapshot: Path) -> dict:
    config = _read_json(snapshot / "config.json")
    filenames = {path.name for path in snapshot.iterdir()} if snapshot.is_dir() else set()
    return {
        "model_id": model_id,
        "snapshot": snapshot.name,
        "path": str(snapshot),
        "roles": _model_roles(model_id, config),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures", []),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "has_config": "config.json" in filenames,
        "has_tokenizer": any(
            name in filenames
            for name in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")
        ),
        "has_weights": any(
            name.endswith((".safetensors", ".bin", ".pt"))
            or name.endswith(".safetensors.index.json")
            for name in filenames
        ),
    }


def _scan_hf_hub(root: Path) -> list[dict]:
    records = []
    if not root.is_dir():
        return records
    for model_dir in sorted(root.glob("models--*")):
        model_id = _model_id_from_cache_dir(model_dir.name)
        if not model_id:
            continue
        snapshots = model_dir / "snapshots"
        if not snapshots.is_dir():
            records.append(
                {
                    "model_id": model_id,
                    "snapshot": None,
                    "path": str(model_dir),
                    "roles": _model_roles(model_id, {}),
                    "model_type": None,
                    "architectures": [],
                    "hidden_size": None,
                    "num_hidden_layers": None,
                    "has_config": False,
                    "has_tokenizer": False,
                    "has_weights": False,
                }
            )
            continue
        records.extend(
            _snapshot_record(model_id, snapshot)
            for snapshot in sorted(snapshots.iterdir())
            if snapshot.is_dir()
        )
    return records


def _scan_nemo_checkpoints(root: Path) -> list[dict]:
    records = []
    if not root.is_dir():
        return records
    for organization in sorted(path for path in root.iterdir() if path.is_dir()):
        for model in sorted(path for path in organization.iterdir() if path.is_dir()):
            iterations = sorted(path.name for path in model.glob("iter_*") if path.is_dir())
            records.append(
                {
                    "model_id": f"{organization.name}/{model.name}",
                    "path": str(model),
                    "iterations": iterations,
                    "has_checkpoint": bool(iterations),
                }
            )
    return records


def _declared_models(repo_root: Path) -> list[str]:
    models = set()
    for path in repo_root.glob("configs/**/*.yaml"):
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        models.update(_MODEL_NAME.findall(content))
    return sorted(model for model in models if model.lower() != "null")


def _cache_roots() -> list[Path]:
    roots = {
        Path("/data/huggingface/hub"),
        Path("/root/.cache/huggingface/hub"),
    }
    for name in ("HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        if value := os.environ.get(name):
            roots.add(Path(value))
    if value := os.environ.get("HF_HOME"):
        roots.add(Path(value) / "hub")
    return sorted(roots, key=str)


def main() -> None:
    _, overrides = _parse_args()
    hub_records = [
        record
        for root in _cache_roots()
        for record in _scan_hf_hub(root)
    ]
    nemo_records = _scan_nemo_checkpoints(Path("/data/huggingface/nemo_rl"))
    cached_ids = {
        record["model_id"]
        for record in [*hub_records, *nemo_records]
        if record.get("has_weights") or record.get("has_checkpoint")
    }
    embedding_candidates = sorted(
        {
            record["model_id"]
            for record in hub_records
            if "embedding" in record["roles"] or "reranker" in record["roles"]
        }
    )
    declared = _declared_models(REPO_ROOT)
    report = {
        "offline_environment": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
        "cache_roots": [str(root) for root in _cache_roots()],
        "hub_snapshots": hub_records,
        "nemo_checkpoints": nemo_records,
        "declared_models": [
            {"model_id": model_id, "confirmed_cached": model_id in cached_ids}
            for model_id in declared
        ],
        "summary": {
            "confirmed_cached_model_ids": sorted(cached_ids),
            "embedding_or_reranker_candidates": embedding_candidates,
            "hub_snapshot_count": len(hub_records),
            "nemo_checkpoint_count": len(nemo_records),
        },
    }
    output_path = _output_dir(overrides) / "model_inventory.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[model-inventory] report")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[model-inventory] saved={output_path}")


if __name__ == "__main__":
    main()
