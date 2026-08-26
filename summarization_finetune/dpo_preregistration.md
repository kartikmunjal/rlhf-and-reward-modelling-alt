# Judge-preference DPO v1 preregistration

Status: **locked separately from SFT before SFT training or any candidate/judge
outcomes were observed on 2026-08-26**.

## Question

Does LoRA DPO against preferences from the already frozen Claude pairwise judge
improve held-out relevance and consistency beyond SFT, and does any apparent
gain persist after controlling for summary length?

## Preference construction and training

On each of 256 disjoint validation articles, the SFT model samples four
candidates using the fixed temperature, top-p, top-k, length budget, and
article-derived seeds in the config. All six unordered candidate pairs are
judged once with the unchanged Claude pairwise prompt. A DPO row is admitted
only when relevance and consistency select the same non-tie underlying
candidate. Conflicts and any tie on either primary axis are excluded; secondary
axes never choose training preferences.

The policy initializes from the merged SFT checkpoint. The frozen reference is
that same merged SFT checkpoint. Only a new rank-16 LoRA adapter is trained for
one epoch with the fixed beta and schedule. Preference articles and candidates
cannot enter final evaluation.

## Confirmatory comparison

DPO and SFT generate on the identical final articles with identical deterministic
decoding. The primary estimand is the paired mean Claude-score difference, DPO
minus SFT. DPO succeeds only if its article-bootstrap 95% CI is strictly positive
for both relevance and consistency. Validity and missingness rules match SFT.

## Locked reward-hacking diagnostics

Report the paired word-count shift and its article-bootstrap CI. For each primary
axis, fit score on DPO treatment, standardized output length, and article fixed
effects; article-bootstrap the treatment coefficient. This asks whether the
within-article score gain survives adjustment for length. Also report paired
ROUGE-L change against the reference and the independent amended GPT-5-mini
score change.

A raw Claude gain whose length-controlled interval is not positive and lacks
support from the independent proxies is evidence of length-exploitation risk,
not clean quality improvement. Persistence after control is evidence against
this specific reward-hacking mechanism, but is not equivalent to new human
annotation. The result is reported whichever direction it takes.
