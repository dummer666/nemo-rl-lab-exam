# qa_sft_trajectory_build_wanghaonan

Builds evidence-grounded OpenAI-message trajectories before SFT:

- converts `ready_one_search` candidates directly;
- samples eight second-query rewrites from cached `Qwen/Qwen3.5-9B`;
- rejects duplicate, ungrounded, answer-leaking, or non-incremental queries;
- requires cumulative rendered evidence to cover every gold keypoint;
- validates role order, string observations, final answer rendering, and token
  length;
- separates open-question SFT train, validation, and future RL holdout rows;
- adds a small balanced objective-question retention set.

The job fails after persisting diagnostics if fewer than ten clean two-search
trajectories survive.
