#!/usr/bin/env python
"""Run isolated objective GRPO from the selected objective SFT pilot."""

from __future__ import annotations

import argparse
import json
import os
import pprint
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from torch.utils.data import Dataset

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup  # noqa: E402
from nemo_rl.algorithms.utils import get_tokenizer, set_seed  # noqa: E402
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType  # noqa: E402
from nemo_rl.distributed.virtual_cluster import init_ray  # noqa: E402
from nemo_rl.models.generation import configure_generation_config  # noqa: E402
from nemo_rl.utils.config import (  # noqa: E402
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir  # noqa: E402

from common.environments.qa_retrieval_env import QARetrievalEnv  # noqa: E402
from common.retrieval.qa_objective_grpo import (  # noqa: E402
    select_objective_curriculum,
)
from common.retrieval.qa_sft import AGENT_INSTRUCTIONS, format_agent_prompt  # noqa: E402
from common.retrieval.qa_target_rebuild import question_fingerprint  # noqa: E402
from experiments.qa_objective_sft_data_wanghaonan import run as objective_data  # noqa: E402

TASK_NAME = "qa_retrieval"
CLEAN_TRAIN_PATH = Path(
    "/shared/outputs/wanghaonan/qa_training_clean_wanghaonan/"
    "qa_training_clean_wanghaonan-wanghaonan-20260718-092437/"
    "cleaned_data/clean_train.jsonl"
)
PACK_ROOT = Path(
    "/shared/outputs/wanghaonan/qa_objective_sft_data_wanghaonan/"
    "qa_objective_sft_data_wanghaonan-wanghaonan-20260719-103532/"
    "objective_sft_data"
)
MEGATRON_CACHE = Path("/shared/outputs/wanghaonan/nemo_rl_megatron_cache")


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


class QAObjectiveDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        tokenizer,
        input_key: str,
        output_key: str,
        system_prompt: str | None = None,
    ):
        self.rows = [dict(row) for row in rows]
        self.tokenizer = tokenizer
        self.input_key = input_key
        self.output_key = output_key
        self.system_prompt = system_prompt or AGENT_INSTRUCTIONS

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> DatumSpec:
        row = self.rows[idx]
        query = str(row[self.input_key])
        expected = str(row[self.output_key])
        curriculum = row["_curriculum"]
        prompt_text = format_agent_prompt(
            self.tokenizer,
            query,
            system_prompt=self.system_prompt,
        )
        token_ids = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0]
        message_log: LLMMessageLogType = [
            {"role": "user", "content": prompt_text, "token_ids": token_ids}
        ]
        return {
            "message_log": message_log,
            "length": len(token_ids),
            "extra_env_info": {
                "expected_answer": expected,
                "query": query,
                "bank": str(row.get("bank", "")),
                "search_count": 0,
                "search_queries": [],
                "invalid_count": 0,
                "force_search": False,
                "minimum_searches": 0,
                "evidence_hits": [],
                "evidence_coverage": 0.0,
                "curriculum_step": int(curriculum["step"]),
                "curriculum_phase": "objective_only",
            },
            "loss_multiplier": 1.0,
            "idx": idx,
            "task_name": TASK_NAME,
            "stop_strings": ["</search>"],
        }


def _curriculum(
    clean_rows: Sequence[Mapping[str, Any]],
    official_rows: Sequence[Mapping[str, Any]],
    sft_rows: Sequence[Mapping[str, Any]],
    *,
    total_steps: int,
    seed: int,
) -> list[dict[str, Any]]:
    official_fingerprints = {
        question_fingerprint(str(row["query"])) for row in official_rows
    }
    sft_fingerprints = {
        str(row["question_fingerprint"]) for row in sft_rows
    }
    candidates, _rejections = objective_data._objective_candidates(
        clean_rows,
        official_fingerprints,
    )
    curriculum = select_objective_curriculum(
        candidates,
        sft_fingerprints | official_fingerprints,
        total_steps=total_steps,
        seed=seed,
    )
    selected_fingerprints = {
        str(row["question_fingerprint"]) for row in curriculum
    }
    if len(selected_fingerprints) != len(curriculum):
        raise RuntimeError("objective GRPO curriculum contains duplicates")
    if selected_fingerprints.intersection(sft_fingerprints):
        raise RuntimeError("objective GRPO overlaps objective SFT")
    if selected_fingerprints.intersection(official_fingerprints):
        raise RuntimeError("objective GRPO overlaps official validation")
    return curriculum


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = _parse_args()
    if not args.config:
        args.config = str(THIS_DIR / "config.yaml")
    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    config = OmegaConf.to_container(config, resolve=True)
    config: MasterConfig = MasterConfig(**config)
    pprint.pprint(config)

    data_dir = Path(
        os.environ.get("QA_RL_DATA_DIR", config.data["data_dir"])
    )
    official_path = data_dir / "val.jsonl"
    manifest_paths = (
        PACK_ROOT / "objective_train_manifest.jsonl",
        PACK_ROOT / "objective_validation_manifest.jsonl",
    )
    configured_exclusion_paths = tuple(
        Path(str(path))
        for path in config.data.get("excluded_manifest_paths", [])
    )
    exclusion_roots = tuple(
        Path(str(path))
        for path in config.data.get("excluded_manifest_roots", [])
    )
    missing_roots = [str(path) for path in exclusion_roots if not path.is_dir()]
    if missing_roots:
        raise FileNotFoundError(
            f"missing objective GRPO exclusion roots: {missing_roots}"
        )
    discovered_exclusion_paths = []
    for root in exclusion_roots:
        matches = sorted(root.rglob("curriculum.jsonl"))
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one curriculum.jsonl under {root}, "
                f"found {len(matches)}: {matches}"
            )
        discovered_exclusion_paths.extend(matches)
    extra_exclusion_paths = (
        *configured_exclusion_paths,
        *discovered_exclusion_paths,
    )
    required = [
        CLEAN_TRAIN_PATH,
        official_path,
        *manifest_paths,
        *extra_exclusion_paths,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing objective GRPO inputs: {missing}")
    curriculum = _curriculum(
        _read_jsonl(CLEAN_TRAIN_PATH),
        _read_jsonl(official_path),
        [
            row
            for path in (*manifest_paths, *extra_exclusion_paths)
            for row in _read_jsonl(path)
        ],
        total_steps=int(config.grpo["max_num_steps"]),
        seed=int(config.grpo["seed"]),
    )
    expected_rows = int(config.grpo["max_num_steps"]) * int(
        config.grpo["num_prompts_per_step"]
    )
    if len(curriculum) != expected_rows:
        raise RuntimeError(
            f"objective curriculum rows changed: {len(curriculum)} "
            f"!= {expected_rows}"
        )
    audit = {
        "rows": len(curriculum),
        "question_types": dict(
            Counter(row["question_type"] for row in curriculum)
        ),
        "unique_fingerprints": len(
            {row["question_fingerprint"] for row in curriculum}
        ),
        "sft_overlap": 0,
        "official_overlap": 0,
        "extra_exclusion_manifests": [
            str(path) for path in extra_exclusion_paths
        ],
    }
    print(f"[objective-grpo] curriculum={json.dumps(audit)}", flush=True)

    config.logger["log_dir"] = get_next_experiment_dir(
        config.logger["log_dir"]
    )
    artifact_dir = (
        Path(config.logger["log_dir"]).parent / "objective_grpo_audit"
    )
    _write_jsonl(artifact_dir / "curriculum.jsonl", curriculum)
    (artifact_dir / "summary.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    MEGATRON_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "NRL_MEGATRON_CHECKPOINT_DIR",
        str(MEGATRON_CACHE),
    )
    init_ray()
    set_seed(config.grpo["seed"])
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"],
        tokenizer,
    )
    dataset_args = {
        "tokenizer": tokenizer,
        "input_key": config.data.get("input_key", "query"),
        "output_key": config.data.get("output_key", "expected_answer"),
        "system_prompt": config.data.get("system_prompt") or None,
    }
    train_dataset = QAObjectiveDataset(curriculum, **dataset_args)
    validation_dataset = QAObjectiveDataset(
        [
            {
                **row,
                "_curriculum": {
                    "step": 0,
                    "slot": "validation",
                    "phase": "validation",
                },
            }
            for row in _read_jsonl(official_path)
        ],
        **dataset_args,
    )
    print(
        f"[objective-grpo] train={len(train_dataset)} "
        f"validation={len(validation_dataset)}",
        flush=True,
    )

    train_env_cfg = dict(config.env[TASK_NAME]["cfg"])
    validation_env_cfg = {
        **train_env_cfg,
        "evidence_reward_scale": 0.0,
        "search_cost": 0.0,
        "duplicate_query_penalty": 0.0,
    }
    train_env = QARetrievalEnv.options(num_gpus=0).remote(
        cfg=train_env_cfg
    )
    validation_env = QARetrievalEnv.options(num_gpus=0).remote(
        cfg=validation_env_cfg
    )
    (
        policy,
        policy_generation,
        _nemo_gym,
        _cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(
        config,
        tokenizer,
        train_dataset,
        validation_dataset,
    )
    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        {TASK_NAME: train_env},
        {TASK_NAME: validation_env},
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()
