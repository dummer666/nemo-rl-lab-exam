# grpo_qwen3.5-9b_qa-rl-agent_wanghaonan

Qwen 3.5 9B Base 的多轮 QA GRPO 实验。模型可在最终作答前检索集群
`/data/docs` 下的 Markdown 技术资料，目标指标是 `validation/accuracy`。

## 交互协议

每轮只能执行一种动作：

```text
<search>设备名、流程名或规范关键词</search>
```

环境返回最多 3 个 BM25 文档片段。最多检索两次，随后必须按题型提交：

```text
\boxed{B}
\boxed{A,C,D}
\boxed{空1; 空2}
```

客观题和填空沿用 `common/rewards/qa_reward.py`，简答题沿用平台注入的
LLM judge；judge 不可达时按仓库既有逻辑回退到关键词覆盖率。

## 实现

- `run.py`：读取 `/data/datasets/qa_rl`，构造多轮 `DatumSpec` 并启动 GRPO。
- `common/environments/qa_retrieval_env.py`：处理检索动作、回灌结果和最终判分。
- `common/retrieval/markdown_bm25.py`：纯 Python Markdown 分块与中英混合 BM25。
- `config.yaml`：单卡 H200、Megatron + LoRA，最多 3 轮 rollout。

检索器把模型关键词与原题题干、题库名称共同查询，提升缩写和短查询的召回；
每次回灌限制为 1500 字符，防止两轮检索挤掉最终答案上下文。

## 提交

```bash
uv run lab validate grpo_qwen3.5-9b_qa-rl-agent_wanghaonan
uv run lab submit grpo_qwen3.5-9b_qa-rl-agent_wanghaonan
uv run lab logs <job_id>
```

目标集群由同目录 `cluster` 文件固定为 `h200`。训练期间重点观察：

- `validation/accuracy`
- `validation/qa_format_penalty_rate`
- `train/avg_turns_per_sample`
- `train/truncation_rate`
