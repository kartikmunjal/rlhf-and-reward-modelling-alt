# Preregistration: Recursive Self-Improvement v1

**Frozen before:** data materialization, baseline training, iterative training,
self-label experiments, and inspection of any new result.

## Integrity correction and starting point

The repository's prior 57.1%, 62.3%, and 65.8% iterative-DPO values were
illustrative expectations without committed empirical artifacts. They are not
prior observations and will not enter any fit, threshold, or comparison. The
new study starts at round zero and may produce entirely different results.

## Primary family and exclusions

The primary family is one GPT-2-medium lineage: immutable base, SFT, ordinary
DPO, and rolling-2 iterative-DPO rounds 1–8. All checkpoints must share the
same base revision, tokenizer, context length, task, and deterministic data
partition. Existing Qwen PPO/GRPO checkpoints are reported separately and
excluded from every primary curve. PPO and GRPO will not be newly trained in
this study.

## Data separation

Anthropic HH-RLHF harmless-base is partitioned by SHA-256 rank of normalized,
deduplicated prompts. SFT training, reward-ensemble training/validation,
iterative improvement prompts, and independent evaluation prompts are
disjoint. Exact source revision, selected IDs, normalization version, and file
hashes must be locked before training.

The independent suite never supplies DPO labels. Primary capability is paired
win rate against the same frozen SFT reference. Prompt-level records—not only
means—must be retained. A Wilson interval reports valid-evaluation coverage;
paired prompt bootstrap intervals report all capability differences.

## Stage 1: horizon and plateau

Rolling-2 iterative DPO runs for eight rounds from the verified SFT checkpoint,
using a frozen K=3 human-preference reward ensemble as labeler. Each round uses
256 on-policy prompts, two sampled candidates per prompt, and 200 update steps.
The exact policy, preference buffer, generated candidates, ensemble member
scores, training tokens, optimizer steps, runtime, and evaluation outputs are
persisted per round.

A plateau is declared only if the paired 95% bootstrap CI for the incremental
win-rate gain includes zero in two consecutive rounds. Linear and saturating
exponential curves are both fit. The selected descriptive curve is the one
with lower leave-one-round-out mean squared prediction error; a difference
below 1e-6 selects linear. No curve is extrapolated beyond round eight.

## Stage 2: capability versus compute

Base, SFT, ordinary DPO, and all eight iterative checkpoints are evaluated on
the identical independent prompts. The x-axis is cumulative non-padding
training tokens, with optimizer steps, examples, wall time, and peak allocated
GPU memory retained as diagnostics. Base has zero incremental training tokens.
Linear versus saturating-exponential selection uses leave-one-checkpoint-out
prediction error under the same fixed rule. Qwen results may appear in a
visually separate table but never in this fit.

## Stage 3: increasing reliance on self-generated labels

Five independent conditions begin from the identical Stage-1 round-3
checkpoint and use 0%, 25%, 50%, 75%, or 100% self labels for four rounds.
Mixture assignment is deterministic before labels are observed. External
training labels come from a frozen Claude pairwise prompt. Self labels are the
current policy's length-normalized conditional likelihood ranking of its own
candidates. This operationalization is intentionally narrow and will not be
described as reflective self-judgment.

The frozen K=3 human-trained reward ensemble is evaluation-only throughout
Stage 3 and never supplies a training label. Primary comparison is each
condition's round-4 external-evaluator win rate minus the 0% condition with a
paired prompt bootstrap CI and Holm correction. We also report training-label
agreement with the evaluator, ensemble disagreement, response-length shift,
and KL from the common Stage-3 start. Degradation requires the upper CI bound
for the primary difference to be below zero.

## Reporting boundaries

All failures, invalid rows, resumptions, and protocol amendments are retained.
Prompt-bootstrap intervals do not capture training-seed uncertainty because
this first execution has one seed. This experiment studies a small controlled
self-training loop; it does not measure frontier-scale recursive improvement.
