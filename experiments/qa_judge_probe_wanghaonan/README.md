# qa_judge_probe_wanghaonan

Minimal read-only probe for the platform-injected short-answer Judge.

It does not load Qwen, read exam data, or train. It reports only:

- whether `JUDGE_BASE_URL`, `JUDGE_MODEL`, and `JUDGE_API_KEY` are present;
- sanitized `/models` and chat-completion HTTP status and latency;
- model IDs returned by `/models`;
- scores for a semantic paraphrase and a contradictory keyword trap;
- whether the endpoint meaningfully separates those two answers.

The endpoint URL, API key, and raw response text are never written.

```bash
uv run lab validate qa_judge_probe_wanghaonan
uv run lab submit qa_judge_probe_wanghaonan
```
