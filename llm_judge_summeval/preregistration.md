# SummEval LLM-as-Judge v1 preregistration

Status: **locked before dataset acquisition, label inspection, prompt tuning, or
judge execution on 2026-08-24**. The machine-readable source of truth is
[`study_config.json`](study_config.json). Any later change requires a versioned
amendment that states whether any outcomes had been observed.

## Research question

Does a rubric-guided LLM judge track expert human judgments of summary quality
on held-out SummEval articles, and does it track them more closely than
ROUGE-L? A secondary question is whether the finding and individual ratings are
stable across model providers.

## Data and leakage boundary

The dataset is the original SummEval human-annotation release: 100
CNN/DailyMail source articles, 16 system summaries per article, and expert
ratings for coherence, consistency, fluency, and relevance. Expert-annotator
means are the sole human target; crowd judgments are excluded from confirmatory
analysis.

Splitting is by source article, never by summary. Canonical article IDs are
UTF-8 encoded, SHA-256 hashed with the study split salt, sorted by digest, and
assigned as follows:

- first 20 articles: development (320 summaries);
- remaining 80 articles: held-out validation (1,280 summaries).

The development partition may be used for prompt iteration. Held-out human
ratings must not be loaded by prompt-development code. The held-out runner must
verify the frozen prompt manifest before issuing any request. Failed or refused
API responses are retained and counted; they are not silently retried with a
changed prompt.

## Judges and prompts

The confirmatory primary judge is the pinned Anthropic snapshot
`claude-haiku-4-5-20251001`. The secondary cross-provider judge is the pinned
OpenAI snapshot `gpt-5-mini-2025-08-07`; it is a prespecified robustness check
and cannot determine primary success.

Temperature is 0, no tools or retrieval are allowed, and responses must satisfy
the schemas in `prompts.json`. Pointwise calls receive the source article and
one candidate summary and return integer 1–5 ratings for all four axes in one
response. Pairwise calls receive the article and two summaries and return an
`A`, `B`, or `tie` verdict separately for every axis. Explanations are stored
for audit but never used as numeric outcomes.

The four axes are evaluated. Relevance and consistency are confirmatory
primary axes; coherence and fluency are secondary and have no pass/fail rule.

## Pairwise sampling and Bradley–Terry mapping

Pairwise evaluation is a diagnostic, not part of the primary success rule. For
each held-out article, five unordered summary pairs are selected by a
deterministic hash-ranked greedy algorithm, subject to no repeated unordered
pair and approximately balanced summary participation. This gives 400 unique
held-out pairs. Every pair is judged in both A/B orders by Claude, yielding 800
requests whose single structured response covers all axes.

Per-axis Bradley–Terry scores are fit with tie outcomes contributing half a win
to each side and a fixed L2 penalty from the config. Scores are centered within
article. Pairwise agreement with the sign of the human-score difference and
Spearman correlation of fitted scores with human scores are secondary.

## Baseline and estimands

ROUGE-L F1 against the SummEval reference summary is computed once per candidate
summary and compared with each human axis. The primary correlation is
summary-level Spearman rho. Article-cluster bootstrap resampling preserves the
16-summary groups and produces 95% percentile confidence intervals using 2,000
deterministic replicates.

For each axis and judge, report:

- Spearman rho with expert human mean and its article-bootstrap 95% CI;
- ROUGE-L Spearman rho on exactly the same rows;
- paired bootstrap distribution of judge rho minus ROUGE-L rho;
- valid-response count and rate, with a Wilson 95% interval for the rate.

Cross-provider agreement is Spearman rho between Claude and GPT-5 mini
pointwise ratings on jointly valid held-out rows, with an article-bootstrap CI.
It is descriptive and cannot rescue a failed primary judge.

## Locked success rule

The primary judge succeeds only if **both** relevance and consistency satisfy
both conditions on the untouched held-out partition:

1. point estimate Spearman rho is at least 0.40; and
2. the paired-bootstrap 95% interval for `judge rho - ROUGE-L rho` excludes zero
   on the positive side.

No multiplicity-adjusted claim is made beyond this intersection rule: both
primary axes must pass. A confidence interval overlapping zero is
inconclusive, not evidence of equivalence. Coherence, fluency, pairwise results,
and GPT-5 mini results remain secondary regardless of magnitude.

## Failure-mode diagnostics

Length bias is Spearman rho between pointwise judge rating and whitespace-token
summary length, reported per axis with article-bootstrap CIs. A partial
Spearman correlation between judge and human scores controlling for length is
also reported as a sensitivity analysis; it does not replace the primary
estimand.

Position bias is measured on the 400 bidirectionally judged Claude pairs. A
flip occurs when the preferred underlying summary changes after swapping A and
B; ties changing to a preference, or a preference changing to a tie, are
reported separately as order instability. Rates receive pair-cluster bootstrap
CIs and Wilson intervals.

## Prespecified mitigation test

Before seeing pairwise outcomes, we predict that order-symmetrization will
improve human pairwise agreement relative to either single ordering. The
symmetrized verdict accepts a preference only when the two orderings agree on
the same underlying summary; all other cases become ties. Its human-agreement
difference versus the first-order verdict receives a paired pair-bootstrap 95%
CI. The mitigation is supported only if that interval excludes zero positively;
otherwise it is rejected or deemed inconclusive. This test is secondary.

## Missingness, stopping, and scope

Malformed responses receive one schema-identical retry. Transport/rate-limit
failures use exponential backoff without changing prompt text. After the retry
budget, the row is missing and remains in the denominator for validity rates.
No score is imputed. Analysis requires at least 90% valid pointwise responses
per provider and 90% complete bidirectional Claude pair judgments; otherwise
the affected claim is labeled infrastructure-incomplete.

All scheduled held-out calls run unless an API outage, authentication failure,
budget cap, or reproducible software error stops execution. Apparent performance
is never a stopping reason. Claims are limited to these fixed model snapshots,
SummEval, the frozen prompts, and summary-level expert means.

