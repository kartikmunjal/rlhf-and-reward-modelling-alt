# PPO vs GRPO v1 — preregistered result

## Design and integrity

- Base model: `Qwen/Qwen2.5-0.5B-Instruct` at resolved revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- Paired seeds / trials: 3; optimizer steps per run: 200.
- Evaluation: 400 frozen problems per seed and method; bootstrap replicates: 2000.
- Smoke outputs were excluded. All six full manifests, trajectories, and prediction sets passed validation.

## Primary outcome

| Method | Exact match (hierarchical bootstrap 95% CI) | Per-seed successes | Valid tagged answers |
|---|---:|---:|---:|
| PPO | 0.0000 [0.0000, 0.0000] | 0 / 0 / 0 of 400 | 0 / 0 / 0 of 400 |
| GRPO | 0.0000 [0.0000, 0.0000] | 0 / 0 / 0 of 400 | 0 / 0 / 0 of 400 |

Paired GRPO − PPO exact-match difference: **0.0000 [0.0000, 0.0000]**. The bootstrap interval collapses at zero because every prediction failed; this is a floor effect and cannot establish equivalence. A per-seed Wilson interval for 0/400 is [0.0000, 0.0095].

## Stability and systems metrics

All cells are means across three independent seeds with seed-bootstrap 95% intervals and `N_trials=3`.
PPO native KL is sequence-summed by TRL and is divided by the locked 64-token completion length for the per-token comparison; GRPO logs per-token KL directly.

| Method | Reward variance | KL/token AUC | Peak VRAM MB | Train seconds | Generated tok/s |
|---|---:|---:|---:|---:|---:|
| PPO | 0.000002 [0.000002, 0.000002] | 0.102151 [0.098872, 0.107775] | 3758.9 [3758.9, 3758.9] | 419.0 [417.5, 420.0] | 61.1 [60.9, 61.3] |
| GRPO | 0.000002 [0.000001, 0.000003] | 0.000213 [0.000164, 0.000243] | 2567.6 [2567.6, 2567.6] | 223.9 [223.1, 225.0] | 114.3 [113.8, 114.7] |

GRPO groups with zero reward standard deviation: **0.7000 [0.7000, 0.7000]**. Both methods generated to the 64-token cap without producing valid tagged answers on evaluation.

## Finding

The preregistered comparison is valid as an execution study but uninformative about relative task performance. The base policy almost never entered the verifier's positive-support region: exact-answer reward was absent, format reward was absent on held-out generations, and GRPO frequently had no within-group contrast. Consequently, both algorithms remained at the evaluation floor. GRPO used less GPU memory and trained faster, but those systems differences do not imply better optimization when the reward supplies essentially no task signal.

This replaces the prior scaffold caveat with a real preregistered negative result. A follow-up must be separately preregistered and should add outcome-blind reward shaping or arithmetic SFT so that both methods encounter nonzero correctness signal; this v1 result must not be overwritten or reframed as a tie.
