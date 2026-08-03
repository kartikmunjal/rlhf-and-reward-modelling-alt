# PPO vs GRPO v2 — preregistered full comparison

**Status:** preregistered, not run. The machine-readable source of truth is
[`configs/ppo_grpo_v2.json`](../configs/ppo_grpo_v2.json).

V1 reached a zero-accuracy floor because its 64-token cap prevented answers and
its sparse reward supplied almost no contrast. The first v2 pilot fixed that
floor but was too easy for group-relative optimization; v2b restored contrast
but fell below the frozen exactness minimum. The disjoint v2c pilot passed every
prewritten gate. None of those pilot problems may enter this study.

Both methods start from the same hashed arithmetic-SFT checkpoint and train LoRA
adapters on the same versioned four-operation generator. The primary compute
budget is 200 optimizer steps. Each method must also execute exactly 100 rollout
groups and 400 sampled completions; runtime assertions convert any mismatch into
an infrastructure failure rather than a result. GRPO uses `num_iterations=1`
and two training steps per generated group, correcting v1's double-reuse error.

The primary outcome is greedy numeric exact match on the same frozen 400
problems. Strict tagged exact match is secondary so formatting cannot erase
otherwise-correct arithmetic. The unchanged programmatic reward includes exact
outcomes and two intermediate calculations. A shared baseline evaluation of the
SFT checkpoint measures whether either RL method improves over its actual start.

Three paired seeds are mandatory. The final report uses 2,000 deterministic
hierarchical-bootstrap replicates over paired seeds and paired evaluation
problems. Reward variance, directly comparable per-token KL, clipping,
truncation, peak VRAM, wall time, and actual generated tokens are diagnostics.
An interval overlapping zero is inconclusive, not evidence of equivalence.
