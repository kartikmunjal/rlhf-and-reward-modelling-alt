# LLM-as-Judge for Summarization Quality

This module extends the repository's reward-modeling work from short response
preferences to full-summary evaluation. It uses SummEval human annotations as
ground truth, a pointwise rubric judge, and a pairwise judge interpreted through
the same Bradley–Terry preference framework used elsewhere in the repository.

The study has completed **Stage 1: dataset prepared, no judge calls run**. The immutable
design is in [`preregistration.md`](preregistration.md), the machine-readable
lock is [`study_config.json`](study_config.json), and the frozen prompt source
is [`prompts.json`](prompts.json). [`data_manifest.json`](data_manifest.json)
records source and output hashes plus the exact article-disjoint split. Raw and
processed text remains gitignored. Do not inspect held-out labels or run judge
calls until the final prompt manifest has been generated and committed.

Planned stages:

1. ~~Materialize and hash a document-level 20/80 SummEval split.~~
2. Iterate prompts only on the 20-document development partition.
3. Freeze prompt bytes and model snapshots.
4. Run pointwise and bidirectional pairwise Claude Haiku 4.5 judgments.
5. Run the prespecified GPT-5 mini pointwise cross-provider check.
6. Evaluate correlations, ROUGE-L deltas, length bias, and position bias with
   deterministic bootstrap confidence intervals.
