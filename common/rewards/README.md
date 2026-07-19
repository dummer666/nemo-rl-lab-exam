# 题库 QA 奖励函数

把"答案判分"逻辑与训练框架解耦：环境（`common/environments/qa_env.py`）调用这里的函数算 reward。

## 文件

| 文件 | 入口 | 说明 |
| --- | --- | --- |
| `qa_reward.py` | `qa_rule_reward_fn(queries, completions, expected_answers)` | 纯规则、零成本判分 |
| `qa_judge_reward.py` | `qa_judge_reward_fn(...)` | 混合：简答走 LLM 裁判，失败回退关键词覆盖率 |
| `synonyms.json` | — | 同义词表（填空/简答匹配时自动扩展） |

接口一致：三个**等长列表** → 返回等长 `float` 列表。

## 答案格式（`expected_answer`）

模型须把最终答案写进 `\boxed{...}`，否则 `FORMAT_PENALTY`（-0.5）。`expected_answer` 用 `[type]` 前缀分派：

| 前缀 | 判分规则 |
| --- | --- |
| `[single]` / `[bool]` | 字母完全相等 → 1.0 |
| `[multiple]` | 默认 `partial_penalty`：`(选对−选错)/应选数`，截断 [0,1]（防全选刷分） |
| `[fill]` | 逐空匹配（`|||` 分隔空，`/` 分隔同一空的多种写法）+ 同义词扩展，reward = 答对空数/总空数 |
| `[short]` | 规则版=关键词覆盖率；裁判版=裁判 LLM 打 0~1（失败回退覆盖率） |

## 简答裁判（LLM-as-judge）

简答没有唯一答案，用一个裁判 LLM 打分。NeMo Lab 的托管作业由中心服务注入
`JUDGE_BASE_URL`、`JUDGE_MODEL` 和 `JUDGE_API_KEY`；以下本地 vLLM 配置只用于
仓库外的独立开发，不应覆盖 Lab 注入值：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001
export JUDGE_BASE_URL=http://127.0.0.1:8001/v1
export JUDGE_MODEL=Qwen/Qwen2.5-7B-Instruct
export JUDGE_API_KEY=EMPTY
export JUDGE_CONCURRENCY=16
export JUDGE_TIMEOUT=30
```

客户端使用标准 Bearer 鉴权。端点连不上时会回退到关键词覆盖率；是否启用裁判由
实验 `config.yaml` 的 `env.qa.cfg.use_judge` 控制。只读审计可调用
`probe_judge_endpoint()`：报告变量是否存在、模型 ID、HTTP 状态和延迟，但不会返回
端点 URL 或 API key。

## 自测

```bash
PYTHONPATH=. python3 common/rewards/qa_reward.py        # 规则判分用例
PYTHONPATH=. python3 common/rewards/qa_judge_reward.py  # 混合（无端点会回退）
```
