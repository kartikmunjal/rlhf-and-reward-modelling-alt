# PPO vs GRPO v1 — preregistered single-GPU comparison

**Status:** preregistered, not run. The machine-readable source of truth is
[`configs/ppo_grpo_v1.json`](../configs/ppo_grpo_v1.json). Full-run outputs may
not overwrite or mutate that file.

## Question and hypothesis

On the same small instruction model, LoRA parameterization, sampled responses,
optimizer-step budget, clipping radius, and KL coefficient, does GRPO differ
from PPO in held-out exact-match accuracy or training stability on a
GSM8K-style arithmetic task?

The null hypothesis is a paired exact-match difference of zero. No direction
is preregistered. GRPO's absence of a learned value head may reduce memory and
value-estimation noise, while its within-prompt normalization can discard
signal when all four completions receive the same reward. Either mechanism can
dominate at this scale.

## Frozen task and reward

The repository generates deterministic, multi-step integer word problems from
a versioned grammar. Training and evaluation use disjoint generator seeds. A
completion earns 1.0 for the exact integer in `<answer>...</answer>`, 0.1 for a
well-formed but incorrect tagged integer, and -0.1 without a valid tag. Tokens
after 48 receive the small locked penalty in the configuration. The identical
reward function is used by both methods.

This synthetic-verifiable task was selected because the Windows host has no SFT
or reward-model checkpoints and because a sparse learned reward would confound
algorithm differences with reward-model error. It is GSM8K-style, not GSM8K;
results must not be reported as GSM8K benchmark accuracy.

## Compute matching

Each paired run starts from the same base-model revision and seed. Both methods
sample four completions for one problem, reuse that frozen group for two clipped
updates, and stop after exactly 200 optimizer steps. Prompt and completion token
limits and sampling parameters are shared. This makes optimizer steps the
primary compute budget and generated tokens and wall time audited secondary
budgets. PPO's value head is retained because removing it would change PPO's
mechanism; its parameter and memory cost are reported rather than hidden.

## Outcomes and uncertainty

The primary outcome is greedy exact-match accuracy on the same frozen 400
problems. The report must contain each seed, method-level estimates, and the
paired GRPO-minus-PPO difference. A deterministic 2,000-replicate hierarchical
bootstrap resamples paired seeds and, within selected seeds, paired evaluation
problems. Training diagnostics are reward mean/variance, sampled-policy KL to
the frozen reference at every rollout group, KL area under the trajectory,
non-finite failures, gradient-norm clipping frequency, peak allocated GPU
memory, elapsed training time, and generated tokens per second.

No post-hoc seed removal, hyperparameter selection, or early stopping based on
quality is allowed. Smoke tests use separate output paths and cannot contribute
to the reported estimates.

## Interpretation boundary

This study can support a claim about these implementations on this model, task,
budget, and GPU. It cannot establish broad PPO/GRPO superiority, substitute for
human-preference evaluation, or demonstrate multi-GPU scaling. A confidence
interval overlapping zero is an inconclusive comparison, not a tie.
