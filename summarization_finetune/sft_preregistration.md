# Summarization SFT v1 preregistration

Status: **locked before dataset preparation, generation, training, or judge
execution on 2026-08-26**. The machine-readable source of truth is
[`study_config.json`](study_config.json).

## Question and fixed instrument

Does one epoch of rank-16 LoRA supervised fine-tuning of `gpt2-medium` on the
CNN/DailyMail training split improve frozen-judge relevance and consistency on
held-out CNN/DailyMail articles relative to the unmodified base model?

The Claude Haiku 4.5 pointwise SummEval judge is a fixed measurement instrument.
Its exact prompt hash and final freeze manifest are recorded in the config. No
prompt, rubric, schema, model snapshot, or postprocessing change is permitted.
GPT-5-mini is a secondary cross-provider audit under SummEval amendment 001 and
cannot determine success.

## Data and contamination controls

Training uses the standard CNN/DailyMail 3.0.0 train split for one epoch. Every
known SummEval article is excluded by both canonical story ID and SHA-256 of
normalized source text. The preparation step must prove disjointness across the
training, preference-building, SFT-loss-evaluation, and final-evaluation
partitions and write source hashes and exact selected IDs to a manifest.

The preference pool and SFT loss-evaluation pool are disjoint deterministic
SHA-ranked samples from validation. Final evaluation is a deterministic sample
of 200 test articles after SummEval exclusion. It is untouched by training,
candidate selection, DPO, or prompt development.

## Training and generation

The prompt is `Article:\n{source}\n\nSummary:\n`. GPT-2 receives at most 768
source-side tokens and generates at most 192 summary tokens within its 1,024
token context. SFT loss is applied only to reference-summary tokens. Training
uses the single fixed seed and LoRA/optimizer schedule in the config. The base
and SFT models generate on exactly the same final articles with identical greedy
decoding and a fixed no-repeat constraint.

## Estimand and success

The primary estimand is the article-level paired mean difference in frozen
Claude pointwise score, SFT minus base, separately for relevance and
consistency. Each receives a deterministic 2,000-replicate article bootstrap
95% percentile interval. SFT succeeds only if the lower bound is above zero for
both axes. Coherence, fluency, ROUGE-L, GPT-5-mini, and length are secondary.

Valid response rates include all planned articles and receive Wilson intervals.
No missing score is imputed. Below 90% joint-valid paired coverage, the affected
claim is infrastructure-incomplete. Article bootstrap uncertainty does not
measure training-seed variability; only one training seed is in scope.
