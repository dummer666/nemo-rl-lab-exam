#!/usr/bin/env python
"""Run the 30-step evidence-aware retrieval QA smoke experiment."""

from __future__ import annotations

import argparse
import json
import os
import pprint
import sys
from collections import Counter
from typing import Any

from omegaconf import OmegaConf
from torch.utils.data import Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

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
from common.retrieval.qa_curriculum import (  # noqa: E402
    build_v3_curriculum,
    question_type,
)
from common.retrieval.qa_sft import (  # noqa: E402
    AGENT_INSTRUCTIONS,
    format_agent_prompt,
)

TASK_NAME = "qa_retrieval"


def parse_args():
    parser = argparse.ArgumentParser(description="30-step evidence-aware QA GRPO smoke")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    args, overrides = parser.parse_known_args()
    return args, overrides


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("query") or not row.get("expected_answer"):
                raise ValueError(f"{path}:{line_number} missing query/expected_answer")
            rows.append(row)
    return rows


class QAAgentDataset(Dataset):
    """Convert prepared QA rows to multi-turn environment prompts."""

    def __init__(
        self,
        rows: list[dict],
        tokenizer,
        input_key: str,
        output_key: str,
        system_prompt: str | None = None,
    ):
        self.rows = rows
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
        row_meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        curriculum = (
            row.get("_curriculum") if isinstance(row.get("_curriculum"), dict) else {}
        )

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
                "bank": str(row_meta.get("bank", "")),
                "search_count": 0,
                "search_queries": [],
                "invalid_count": 0,
                "force_search": bool(curriculum.get("force_search", False)),
                "evidence_hits": [],
                "evidence_coverage": 0.0,
                "curriculum_step": int(curriculum.get("step", 0)),
                "curriculum_phase": str(curriculum.get("phase", "validation")),
            },
            "loss_multiplier": 1.0,
            "idx": idx,
            "task_name": TASK_NAME,
            "stop_strings": ["</search>"],
        }


def main():
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    if not args.config:
        args.config = os.path.join(THIS_DIR, "config.yaml")

    config = load_config(args.config)
    print(f"Loaded config: {args.config}")
    if overrides:
        print(f"CLI overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)
    config = OmegaConf.to_container(config, resolve=True)
    config: MasterConfig = MasterConfig(**config)
    pprint.pprint(config)

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"Log directory: {config.logger['log_dir']}")

    init_ray()
    set_seed(config.grpo["seed"])
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"],
        tokenizer,
    )

    data_cfg: dict[str, Any] = config.data
    data_dir = os.environ.get("QA_RL_DATA_DIR") or data_cfg.get("data_dir")
    clean_train_path = os.environ.get("QA_CLEAN_TRAIN_PATH") or data_cfg.get(
        "clean_train_path"
    )
    if not data_dir:
        raise SystemExit("QA_RL_DATA_DIR or data.data_dir is required")
    if not clean_train_path:
        raise SystemExit("QA_CLEAN_TRAIN_PATH or data.clean_train_path is required")
    val_path = os.path.join(data_dir, "val.jsonl")
    for path in (clean_train_path, val_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"QA data file does not exist: {path}")

    curriculum_cfg = dict(data_cfg.get("curriculum") or {})
    train_rows = build_v3_curriculum(
        _read_jsonl(clean_train_path),
        warmup_steps=int(curriculum_cfg.get("warmup_steps", 10)),
        total_steps=int(config.grpo["max_num_steps"]),
        prompts_per_step=int(config.grpo["num_prompts_per_step"]),
        seed=int(config.grpo["seed"]),
    )
    expected_train_rows = (
        int(config.grpo["max_num_steps"]) * int(config.grpo["num_prompts_per_step"])
    )
    if len(train_rows) != expected_train_rows:
        raise ValueError(
            f"curriculum produced {len(train_rows)} rows, expected {expected_train_rows}"
        )
    val_rows = _read_jsonl(val_path)
    print(
        f"Curriculum rows={len(train_rows)} types={dict(Counter(map(question_type, train_rows)))} "
        f"phases={dict(Counter(row['_curriculum']['phase'] for row in train_rows))}; "
        f"validation rows={len(val_rows)}"
    )

    dataset_args = {
        "tokenizer": tokenizer,
        "input_key": data_cfg.get("input_key", "query"),
        "output_key": data_cfg.get("output_key", "expected_answer"),
        "system_prompt": data_cfg.get("system_prompt") or None,
    }
    train_dataset = QAAgentDataset(train_rows, **dataset_args)
    val_dataset = QAAgentDataset(val_rows, **dataset_args)

    train_env_cfg = dict(config.env[TASK_NAME]["cfg"])
    val_env_cfg = {
        **train_env_cfg,
        "evidence_reward_scale": 0.0,
        "search_cost": 0.0,
        "duplicate_query_penalty": 0.0,
    }
    train_env = QARetrievalEnv.options(num_gpus=0).remote(cfg=train_env_cfg)
    val_env = QARetrievalEnv.options(num_gpus=0).remote(cfg=val_env_cfg)

    (
        policy,
        policy_generation,
        _nemo_gym,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, train_dataset, val_dataset)

    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        {TASK_NAME: train_env},
        {TASK_NAME: val_env},
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()
