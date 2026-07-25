# Safety classifier v2 — locked preregistration

Status: **preregistered; no v2 model or held-out evaluation has been run**.

Trial one is immutable. Its Jigsaw split, thresholds, metrics, predictions, and
interpretation remain the historical baseline. V2 is a separate 12-trial
family and may not overwrite any v1 artifact.

## Research question

Does adding direct, pair-level safety supervision improve the rare
`harmful_violent` category without degrading the combined three-category
classifier, calibration, or false-positive behavior on identity-bearing and
contextually benign text?

## Locked data roles

- Jigsaw v1 train partition: training.
- BeaverTails `330k_train`: supplementary training only.
- Jigsaw v1 calibration partition: deterministically split by
  `SHA256("safety-v2-role:" + id)` into threshold-calibration and
  model-selection halves. The first half selects thresholds and trainer
  checkpoints; the second selects among the 12 trials.
- Jigsaw v1 test partition: frozen primary evaluation only.
- BeaverTails `330k_test`: external domain evaluation only.
- Human-annotated ToxiGen `annotated` train and test: external implicit-hate
  evaluation only; never training.
- HateCheck: external functionality evaluation only; never training.
- Civil Comments with original identity columns: external identity-slice
  fairness evaluation only; never training or threshold selection.
- Adjacent-benign v1: retained unchanged as a historical stress test.

Every downloaded dataset revision and normalized-file SHA-256 must appear in
`data/processed/safety_v2/data_manifest.json`. Production scripts refuse
unmanifested inputs.

## Preregistration amendment 1 — BeaverTails annotation aggregation

Date: **2026-07-25**. Status: **approved before data preparation completed and
before any v2 model training or held-out evaluation**.

The first preparation attempt established that BeaverTails `330k_train` and
`330k_test` contain repeated prompt-response rows representing annotations.
The original preregistration did not specify how to aggregate these rows.
Inspection of the source archives found 300,567 raw train rows representing
99,734 unique pairs and 33,396 raw test rows representing 11,088 unique pairs.
There is no pair overlap between those splits.

The locked amendment groups exact stripped prompt-response pairs, maps every
annotation to the three operational targets, and assigns each target by strict
majority vote. A pair tied on any mapped target is excluded in full rather than
assigned optimistically. The audit found 19 such train pairs and no test
pairs. The manifest records raw-row counts, unique and retained pairs,
per-target disagreement and tie counts, exclusions, annotation-count
distributions, and the policy identifier. This amendment changes only
preprocessing; the 12-trial ledger, data roles, metrics, and selection rule
remain unchanged.

## Locked mapping

BeaverTails labels apply to the combined string
`[PROMPT] ... [RESPONSE] ...`.

- `hate_harassment` = `hate_speech,offensive_language` OR
  `discrimination,stereotype,injustice`
- `sexualized` = `sexually_explicit,adult_content`
- `harmful_violent` = `violence,aiding_and_abetting,incitement`

All other BeaverTails categories are outside this operational taxonomy. They
are not silently relabeled into a target category. Multi-label overlap is
preserved.

ToxiGen evaluates `hate_harassment` only. A mean human toxicity rating at or
above 3.0 is positive. HateCheck evaluates `hate_harassment` only using its
gold hate/non-hate label and reports each named functionality separately.

Civil Comments labels use a 0.5 crowd-score cutoff. Identity membership uses a
0.5 cutoff for the original identity columns. Metrics are reported per
identity and as worst-minus-best gaps; identities with no positive or no
negative examples are explicitly unavailable rather than coerced to zero.

## Locked trials

The Cartesian product of four losses and three seeds is exactly 12 trials:

1. unweighted binary cross-entropy;
2. prevalence-weighted binary cross-entropy;
3. prevalence-weighted BCE capped at positive weight 50;
4. focal loss with gamma 2.0 and alpha 0.75;

with seeds 2025, 2026, and 2027. The ordered ledger is
`configs/safety_v2_trial_ledger.json`. Failed runs remain in the ledger. No
additional seed, loss, rank, epoch count, model, or data mixture may be added
without a versioned preregistration amendment and an increased `N_trials`.

The mechanical hypothesis is that direct BeaverTails violence examples improve
the underrepresented violent category; capped weighting and focal loss test
whether v1's raw 339x positive weight was too aggressive. Improvements are not
trusted unless they hold on the combined three-category portfolio of metrics,
not only the rare label.

## Locked evaluation

- Per-category precision, recall, F1, and PR-AUC.
- Macro F1 across the three categories as the primary combined metric.
- Brier score, 15-bin expected calibration error, and reliability-bin counts.
- Calibration-selected thresholds plus a fixed threshold-sensitivity grid.
- Any-label FPR on negative Jigsaw rows and adjacent-benign text.
- Civil Comments identity-slice FPR and FNR with Wilson intervals.
- ToxiGen implicit-hate P/R/F1 and target-group slices.
- HateCheck accuracy by functionality, including contrastive non-hate cases.
- BeaverTails external-domain per-category P/R/F1.

P/R/F1, PR-AUC, Brier, ECE, and macro F1 use deterministic row-bootstrap 95%
intervals. Rates use Wilson intervals and independent rate gaps use Newcombe
intervals. Every result carries the planned/completed `N_trials`.

## Model selection

Primary selection is macro F1 on the held-out model-selection half of the old
calibration partition after all 12 runs are complete. Tie-breakers, in order,
are harmful/violent F1, lower any-label FPR on negative rows, then lower ECE,
all on that same model-selection half. The frozen Jigsaw test is used only for
the final selected-v2 versus v1
comparison; all trial outcomes remain disclosed, but test performance does not
choose the winner. All 12 trials and seed dispersion are published. The
selection rule may not be changed after results are visible.

## Error analysis

The sampler hashes and blinds text IDs, selects deterministic stratified false
positive/false negative samples, and emits an annotation sheet without model
score or trial identity. Annotation labels are quotation, negation,
counter-speech, reclaimed language, identity mention, clinical context, news
context, sarcasm/irony, long-context truncation, ambiguous gold label, and
other. Analysis is valid only after a completed sheet is imported; the tool
never fabricates annotator agreement.

## Exit criteria

V2 may be described as an improvement only if its selected configuration:

- improves macro F1 mechanically and statistically;
- does not materially worsen either strong v1 category;
- improves harmful/violent performance beyond seed noise;
- does not create a clearly positive adjacent-benign or identity FPR gap; and
- reports all 12 trials, failures included.

Otherwise v2 is reported as a negative or mixed result.

## References

- Ji et al. (2023), *BeaverTails: Towards Improved Safety Alignment of LLM
  via a Human-Preference Dataset*, NeurIPS Datasets and Benchmarks.
  <https://arxiv.org/abs/2307.04657>
- Hartvigsen et al. (2022), *ToxiGen: A Large-Scale Machine-Generated Dataset
  for Adversarial and Implicit Hate Speech Detection*, ACL.
  <https://arxiv.org/abs/2203.09509>
- Röttger et al. (2021), *HateCheck: Functional Tests for Hate Speech
  Detection Models*, ACL-IJCNLP.
  <https://aclanthology.org/2021.acl-long.4/>
- Borkan et al. (2019), *Nuanced Metrics for Measuring Unintended Bias with
  Real Data for Text Classification*.
  <https://arxiv.org/abs/1903.04561>
