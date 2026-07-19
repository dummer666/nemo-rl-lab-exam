#!/usr/bin/env python
"""Run a 20-step question-isolated GRPO refinement from retrieval-SFT step 50."""

from __future__ import annotations

import argparse
import json
import os
import pprint
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

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
from common.retrieval.qa_curriculum import question_type  # noqa: E402
from common.retrieval.qa_sft import AGENT_INSTRUCTIONS, format_agent_prompt  # noqa: E402
from common.retrieval.qa_short_grpo import (  # noqa: E402
    build_balanced_open_grpo_curriculum,
    build_short_grpo_curriculum,
)
from common.retrieval.qa_target_rebuild import question_fingerprint  # noqa: E402

TASK_NAME = "qa_retrieval"
SFT_DATA_ROOT = Path(
    "/shared/outputs/wanghaonan/qa_sft_trajectory_build_wanghaonan/"
    "qa_sft_trajectory_build_wanghaonan-wanghaonan-20260718-123640/"
    "sft_trajectories"
)
CLEAN_TRAIN_PATH = Path(
    "/shared/outputs/wanghaonan/qa_training_clean_wanghaonan/"
    "qa_training_clean_wanghaonan-wanghaonan-20260718-092437/"
    "cleaned_data/clean_train.jsonl"
)
MEGATRON_CACHE = Path("/shared/outputs/wanghaonan/nemo_rl_megatron_cache")


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    return parser.parse_known_args()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class QAAgentDataset(Dataset):
    """Render exact SFT-compatible prompts and attach retrieval constraints."""

    def __init__(
        self,
        rows: Sequence[dict],
        tokenizer,
        input_key: str,
        output_key: str,
        system_prompt: str | None = None,
    ):
        self.rows = list(rows)
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
        curriculum = row.get("_curriculum") if isinstance(row.get("_curriculum"), dict) else {}
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
        message_log: LLMMessageLogType = [{"role": "user", "content": prompt_text, "token_ids": token_ids}]
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
                "minimum_searches": int(curriculum.get("minimum_searches", 0)),
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


def _curriculum_audit(rows: Sequence[dict]) -> dict:
    return {
        "rows": len(rows),
        "question_types": dict(Counter(question_type(row) for row in rows)),
        "slots": dict(Counter(str(row["_curriculum"]["slot"]) for row in rows)),
        "minimum_searches": dict(Counter(int(row["_curriculum"]["minimum_searches"]) for row in rows)),
        "unique_source_questions": len({int(row["_curriculum"]["source_row_id"]) for row in rows}),
    }


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

    required = [
        SFT_DATA_ROOT / "rl_holdout.jsonl",
        SFT_DATA_ROOT / "trajectory_manifest.jsonl",
        CLEAN_TRAIN_PATH,
    ]
    data_dir = Path(os.environ.get("QA_RL_DATA_DIR", config.data["data_dir"]))
    val_path = data_dir / "val.jsonl"
    required.append(val_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing short-GRPO inputs: {missing}")

    val_rows = _read_jsonl(val_path)
    official_fingerprints = {
        question_fingerprint(str(row["query"])) for row in val_rows
    }
    holdout_rows = [
        row
        for row in _read_jsonl(SFT_DATA_ROOT / "rl_holdout.jsonl")
        if question_fingerprint(str(row["query"]))
        not in official_fingerprints
    ]
    clean_rows = [
        row
        for row in _read_jsonl(CLEAN_TRAIN_PATH)
        if question_fingerprint(str(row["query"]))
        not in official_fingerprints
    ]
    curriculum_builder = (
        build_balanced_open_grpo_curriculum
        if config.data.get("curriculum_mode") == "balanced_open"
        else build_short_grpo_curriculum
    )
    train_rows = curriculum_builder(
        holdout_rows,
        clean_rows,
        _read_jsonl(
            SFT_DATA_ROOT / "trajectory_manifest.jsonl"
        ),
        total_steps=int(config.grpo["max_num_steps"]),
        prompts_per_step=int(config.grpo["num_prompts_per_step"]),
        seed=int(config.grpo["seed"]),
    )
    expected_rows = int(config.grpo["max_num_steps"]) * int(config.grpo["num_prompts_per_step"])
    if len(train_rows) != expected_rows:
        raise ValueError(f"curriculum produced {len(train_rows)} rows, expected {expected_rows}")
    audit = _curriculum_audit(train_rows)
    audit["official_overlap"] = 0
    audit["curriculum_mode"] = str(
        config.data.get("curriculum_mode", "open_heavy")
    )
    print(f"[short-grpo] curriculum={json.dumps(audit, ensure_ascii=False)}")

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    artifact_dir = Path(config.logger["log_dir"]).parent / "short_grpo_audit"
    _write_jsonl(artifact_dir / "curriculum.jsonl", train_rows)
    (artifact_dir / "summary.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[short-grpo] log_dir={config.logger['log_dir']}")

    MEGATRON_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NRL_MEGATRON_CHECKPOINT_DIR", str(MEGATRON_CACHE))
    init_ray()
    set_seed(config.grpo["seed"])
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"],
        tokenizer,
    )

    data_cfg: dict[str, Any] = config.data
    dataset_args = {
        "tokenizer": tokenizer,
        "input_key": data_cfg.get("input_key", "query"),
        "output_key": data_cfg.get("output_key", "expected_answer"),
        "system_prompt": data_cfg.get("system_prompt") or None,
    }
    train_dataset = QAAgentDataset(train_rows, **dataset_args)
    val_dataset = QAAgentDataset(val_rows, **dataset_args)
    print(f"[short-grpo] train={len(train_dataset)} validation={len(val_dataset)}")

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
        _cluster,
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
