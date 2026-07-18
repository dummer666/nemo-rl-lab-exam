#!/usr/bin/env python
"""Merge the two retrieval-SFT LoRA checkpoints into Hugging Face models."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

MODEL_NAME = "Qwen/Qwen3.5-9B"
BASE_CHECKPOINT = Path("/data/huggingface/nemo_rl/Qwen/Qwen3.5-9B/iter_0000000")
SFT_ROOT = Path(
    "/shared/outputs/wanghaonan/sft_qwen3.5-9b_qa-retrieval-v1_wanghaonan/"
    "sft_qwen3.5-9b_qa-retrieval-v1_wanghaonan-wanghaonan-20260718-130828"
)
STEPS = (25, 50)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def _weight_files(path: Path) -> list[Path]:
    patterns = ("*.safetensors", "pytorch_model*.bin")
    return sorted(file for pattern in patterns for file in path.glob(pattern))


def _prepare_output(path: Path) -> bool:
    if not path.exists():
        return True
    if (path / "config.json").is_file() and _weight_files(path):
        print(f"[sft-export] already complete: {path}", flush=True)
        return False
    if any(path.iterdir()):
        raise RuntimeError(f"Refusing to overwrite partial non-empty export: {path}")
    path.rmdir()
    return True


def _export_step(nemo_rl_dir: Path, step: int) -> dict:
    step_dir = SFT_ROOT / f"step_{step}"
    adapter_checkpoint = step_dir / "policy" / "weights" / "iter_0000000"
    tokenizer_dir = step_dir / "policy" / "tokenizer"
    output_dir = SFT_ROOT / "hf_export" / f"step_{step}"

    required = [
        BASE_CHECKPOINT,
        adapter_checkpoint,
        adapter_checkpoint / "run_config.yaml",
        tokenizer_dir,
        nemo_rl_dir / "examples" / "converters" / "convert_lora_to_hf.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing export inputs: {missing}")

    if _prepare_output(output_dir):
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "uv",
            "run",
            "--no-sync",
            "--extra",
            "mcore",
            "python",
            "-u",
            "examples/converters/convert_lora_to_hf.py",
            "--base-ckpt",
            str(BASE_CHECKPOINT),
            "--adapter-ckpt",
            str(adapter_checkpoint),
            "--hf-model-name",
            MODEL_NAME,
            "--hf-ckpt-path",
            str(output_dir),
        ]
        print(f"[sft-export] step={step} command={' '.join(command)}", flush=True)
        subprocess.run(command, cwd=nemo_rl_dir, check=True)
        shutil.copytree(tokenizer_dir, output_dir, dirs_exist_ok=True)

    weights = _weight_files(output_dir)
    if not (output_dir / "config.json").is_file() or not weights:
        raise RuntimeError(f"Incomplete Hugging Face export: {output_dir}")
    size_bytes = sum(path.stat().st_size for path in weights)
    summary = {
        "step": step,
        "output": str(output_dir),
        "weight_files": len(weights),
        "weight_bytes": size_bytes,
    }
    print(f"[sft-export] verified {summary}", flush=True)
    return summary


def main(overrides: Sequence[str]) -> None:
    del overrides
    nemo_rl_dir = Path(os.environ["NEMO_RL_DIR"])
    if not BASE_CHECKPOINT.is_dir():
        raise FileNotFoundError(f"Missing cached base checkpoint: {BASE_CHECKPOINT}")
    summaries = [_export_step(nemo_rl_dir, step) for step in STEPS]
    print(f"[sft-export] complete: {summaries}", flush=True)


if __name__ == "__main__":
    _args, unknown = _parse_args()
    main(unknown)
