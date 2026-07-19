#!/usr/bin/env python
"""Run a bounded GRPO pilot with strict fill/short and objective replay."""

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
from common.retrieval.markdown_bm25 import MarkdownBM25Index  # noqa: E402
from common.retrieval.qa_curriculum import question_type  # noqa: E402
from common.retrieval.qa_mixed_open_grpo import (  # noqa: E402
    build_mixed_open_curriculum,
    strict_open_candidates,
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
TRAJECTORY_MANIFEST = Path(
    "/shared/outputs/wanghaonan/qa_sft_trajectory_build_wanghaonan/"
    "qa_sft_trajectory_build_wanghaonan-wanghaonan-20260718-123640/"
    "sft_trajectories/trajectory_manifest.jsonl"
)
OBJECTIVE_PACK_ROOT = Path(
    "/shared/outputs/wanghaonan/qa_objective_sft_data_wanghaonan/"
    "qa_objective_sft_data_wanghaonan-wanghaonan-20260719-103532/"
    "objective_sft_data"
)
PRIOR_CURRICULA = (
    Path(
        "/shared/outputs/wanghaonan/grpo_qwen3.5-9b_qa-objective-short_wanghaonan/"
        "grpo_qwen3.5-9b_qa-objective-short_wanghaonan-wanghaonan-20260719-132128/"
        "logs/objective_grpo_audit/curriculum.jsonl"
    ),
    Path(
        "/shared/outputs/wanghaonan/grpo_qwen3.5-9b_qa-objective-100_wanghaonan/"
        "grpo_qwen3.5-9b_qa-objective-100_wanghaonan-wanghaonan-20260719-141205/"
        "logs/objective_grpo_audit/curriculum.jsonl"
    ),
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


class QAMixedDataset(Dataset):
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
        row_meta = row.get("meta") if isinstance(row.get("meta"), Mapping) else {}
        curriculum = (
            row.get("_curriculum")
            if isinstance(row.get("_curriculum"), Mapping)
            else {}
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
                "bank": str(row_meta.get("bank", row.get("bank", ""))),
                "search_count": 0,
                "search_queries": [],
                "invalid_count": 0,
                "force_search": bool(curriculum.get("force_search", False)),
                "minimum_searches": int(curriculum.get("minimum_searches", 0)),
                "evidence_hits": [],
                "evidence_coverage": 0.0,
                "curriculum_step": int(curriculum.get("step", 0)),
                "curriculum_phase": str(
                    curriculum.get("phase", "validation")
                ),
            },
            "loss_multiplier": 1.0,
            "idx": idx,
            "task_name": TASK_NAME,
            "stop_strings": ["</search>"],
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

    data_dir = Path(
        os.environ.get("QA_RL_DATA_DIR", config.data["data_dir"])
    )
    official_path = data_dir / "val.jsonl"
    pack_manifests = (
        OBJECTIVE_PACK_ROOT / "objective_train_manifest.jsonl",
        OBJECTIVE_PACK_ROOT / "objective_validation_manifest.jsonl",
    )
    required = [
        CLEAN_TRAIN_PATH,
        TRAJECTORY_MANIFEST,
        official_path,
        *pack_manifests,
        *PRIOR_CURRICULA,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing mixed GRPO inputs: {missing}")

    clean_rows = _read_jsonl(CLEAN_TRAIN_PATH)
    official_rows = _read_jsonl(official_path)
    official_fingerprints = {
        question_fingerprint(str(row["query"])) for row in official_rows
    }
    excluded_fingerprints = set(official_fingerprints)
    for path in (*pack_manifests, *PRIOR_CURRICULA):
        excluded_fingerprints.update(
            str(row["question_fingerprint"])
            for row in _read_jsonl(path)
            if row.get("question_fingerprint")
        )
    excluded_row_ids = {
        int(row["row_id"])
        for row in _read_jsonl(TRAJECTORY_MANIFEST)
        if row.get("split") in {"train", "validation"}
    }

    index = MarkdownBM25Index.from_directory(
        config.env[TASK_NAME]["cfg"]["docs_dir"],
        chunk_chars=1200,
        overlap_chars=160,
        k1=1.5,
        b=0.75,
        quality_weights=dict(
            config.env[TASK_NAME]["cfg"]["quality_weights"]
        ),
    )
    objective_candidates, objective_rejections = (
        objective_data._objective_candidates(
            clean_rows,
            official_fingerprints,
        )
    )
    objective_candidates = [
        row
        for row in objective_candidates
        if int(row["source_row_id"]) not in excluded_row_ids
        and str(row["question_fingerprint"]) not in excluded_fingerprints
    ]
    fill_candidates, short_candidates, open_rejections = (
        strict_open_candidates(
            clean_rows,
            index,
            excluded_fingerprints,
            excluded_row_ids,
        )
    )
    curriculum = build_mixed_open_curriculum(
        objective_candidates,
        fill_candidates,
        short_candidates,
        total_steps=int(config.grpo["max_num_steps"]),
        seed=int(config.grpo["seed"]),
    )
    expected_rows = int(config.grpo["max_num_steps"]) * int(
        config.grpo["num_prompts_per_step"]
    )
    if len(curriculum) != expected_rows:
        raise RuntimeError(
            f"mixed curriculum rows changed: {len(curriculum)} "
            f"!= {expected_rows}"
        )
    audit = {
        "rows": len(curriculum),
        "question_types": dict(
            Counter(question_type(row) for row in curriculum)
        ),
        "unique_fingerprints": len(
            {str(row["question_fingerprint"]) for row in curriculum}
        ),
        "strict_pool_sizes": {
            "objective": len(objective_candidates),
            "fill": len(fill_candidates),
            "short": len(short_candidates),
        },
        "selected_unique_by_type": {
            current_type: len(
                {
                    str(row["question_fingerprint"])
                    for row in curriculum
                    if question_type(row) == current_type
                }
            )
            for current_type in ("single", "multiple", "bool", "fill", "short")
        },
        "open_rejections": open_rejections,
        "objective_rejections": dict(objective_rejections),
        "official_overlap": 0,
        "prior_training_overlap": 0,
    }
    print(
        f"[mixed-open-grpo] curriculum="
        f"{json.dumps(audit, ensure_ascii=False)}",
        flush=True,
    )
    for row in short_candidates[:10]:
        print(
            "[mixed-open-grpo] accepted_short="
            + json.dumps(
                {
                    "source_row_id": row["source_row_id"],
                    "query": row["query"],
                    "expected_answer": row["expected_answer"],
                    "audit": row["_open_audit"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    config.logger["log_dir"] = get_next_experiment_dir(
        config.logger["log_dir"]
    )
    artifact_dir = (
        Path(config.logger["log_dir"]).parent / "mixed_open_grpo_audit"
    )
    _write_jsonl(artifact_dir / "curriculum.jsonl", curriculum)
    _write_jsonl(
        artifact_dir / "strict_fill_candidates.jsonl",
        fill_candidates,
    )
    _write_jsonl(
        artifact_dir / "strict_short_candidates.jsonl",
        short_candidates,
    )
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
    train_dataset = QAMixedDataset(curriculum, **dataset_args)
    validation_dataset = QAMixedDataset(official_rows, **dataset_args)
    print(
        f"[mixed-open-grpo] train={len(train_dataset)} "
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
