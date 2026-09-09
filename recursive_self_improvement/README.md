# Recursive Self-Improvement Study

This extension replaces the repository's unsupported illustrative iterative-DPO
figures with an artifact-backed GPT-2-medium study. The primary checkpoint
family is deliberately limited to base, SFT, DPO, and iterative-DPO rounds
1–8. Existing Qwen PPO/GRPO artifacts remain valid within their own experiment
but are excluded from the shared capability–compute curve.

The complete design and decision rules are frozen in `preregistration.md` and
`study_config.json`. No result is claimed until raw example-level evaluations,
checkpoint hashes, training-token accounting, environment manifests, and the
generated statistical report exist.

## Study structure

1. Materialize and validate the GPT-2-medium SFT and K=3 reward-ensemble stack.
2. Re-run rolling-2 iterative DPO from round 1 through round 8.
3. Evaluate base, SFT, ordinary DPO, and every iterative checkpoint on one
   immutable held-out suite using an evaluator never used as a training labeler.
4. Compare preregistered linear and saturating-exponential capability curves by
   leave-one-checkpoint-out predictive error.
5. Run a separate self-reliance experiment with 0/25/50/75/100% self-generated
   labels. The human-trained reward ensemble is evaluation-only in this stage.

PPO and GRPO training is explicitly deferred. Adding either requires a new
preregistration and a clear disclosure that it was trained later to expand the
controlled family.
