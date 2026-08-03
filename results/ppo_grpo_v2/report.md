# PPO vs GRPO v2 — preregistered result

## Design and integrity

- Shared SFT checkpoint SHA-256: `1096d9e367ccc1a01bc135143a9729b41965f2b2b3fda1fe3299edab67f876e7`.
- `N_trials=3` paired seeds; 200 optimizer steps, 100 rollout groups, and 400 sampled completions per method/seed.
- Frozen evaluation: 400 identical problems per run; 2,000 deterministic hierarchical-bootstrap replicates.
- All six identities, SFT hashes, compute budgets, evaluation IDs/order, trajectories, and prediction counts passed validation.
- Smoke and pilot outputs are excluded from the confirmatory analysis.

## Primary outcome

| Method | Numeric exact match (hierarchical-bootstrap 95% CI) | Per-seed successes | Tagged format rate |
|---|---:|---:|---:|
| PPO | 0.0758 [0.0325, 0.1242] | 50 / 12 / 29 of 400 | 0.9433 |
| GRPO | 0.3658 [0.3292, 0.4033] | 157 / 145 / 137 of 400 | 0.9950 |

Shared SFT baseline: **0.1100** (44/400).
Paired GRPO − PPO difference: **0.2900 [0.2525, 0.3325]**.
PPO − SFT difference: **-0.0342 [-0.0800, 0.0125]**; GRPO − SFT difference: **0.2558 [0.2217, 0.2917]**.

The preregistered superiority rule is met for GRPO over PPO because the paired interval excludes zero. This is a result on the locked synthetic arithmetic task, not a general claim about either algorithm.

## Stability and systems diagnostics

All intervals below resample the three independent seeds (`N_trials=3`). Native KL values are not placed in one cross-method column because TRL logs PPO KL as a sequence sum and GRPO KL per token.

| Method | Reward variance | Native KL AUC | Peak VRAM MB | Train seconds | Completions/s |
|---|---:|---:|---:|---:|---:|
| PPO | 0.01407 [0.01159, 0.01644] | 2.74829 [2.14247, 3.62466] | 4870.3 [4870.3, 4870.3] | 929.1 [927.8, 930.8] | 0.431 [0.430, 0.431] |
| GRPO | 0.08780 [0.08544, 0.09037] | 0.00411 [0.00360, 0.00469] | 2855.3 [2649.9, 2958.0] | 370.2 [364.3, 373.8] | 1.081 [1.070, 1.098] |

GRPO zero-within-group-reward-variance fraction: **0.3333 [0.2900, 0.4000]**.
Exact generated-token throughput and directly comparable PPO per-token KL were preregistered diagnostics but cannot be reconstructed: the PPO artifact did not persist response token counts. Completion throughput and native KL are reported instead. This telemetry omission does not affect the primary accuracy outcome or compute-budget checks.

## Finding

GRPO produced a large, consistent accuracy gain, while PPO was unstable across seeds and did not reliably improve over the shared SFT start. Mechanically, group-relative updates benefited from frequent within-group verifier contrast; the zero-variance-group diagnostic quantifies where that signal was absent. GRPO also used substantially less peak VRAM and trained faster in this implementation, but those systems results are implementation- and hardware-specific.

This single-GPU LoRA study establishes a reproducible task-specific comparison. It does not establish general PPO/GRPO superiority, full-model behavior, or multi-GPU scaling.
