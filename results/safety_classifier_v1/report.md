# Safety classifier evaluation

- Experiment: `jigsaw_distilbert_lora_v1`
- Full run: `True`
- N_trials: 1
- Bootstrap replicates: 2000
- Test examples: 15878

Category P/R/F1 intervals are row-bootstrap 95% confidence intervals.
FPR intervals use Wilson scores and the independent FPR gap uses a Newcombe
score interval. Thresholds were selected once on the calibration split and
then frozen for both evaluations.

| Category | Support | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Threshold |
|---|---:|---:|---:|---:|---:|
| hate_harassment | 1583 | 0.850 [0.833, 0.868] | 0.820 [0.801, 0.838] | 0.835 [0.820, 0.848] | 0.92 |
| sexualized | 810 | 0.791 [0.765, 0.817] | 0.872 [0.847, 0.894] | 0.830 [0.809, 0.848] | 0.88 |
| harmful_violent | 53 | 0.527 [0.397, 0.653] | 0.547 [0.413, 0.682] | 0.537 [0.416, 0.649] | 0.94 |

## False-positive fairness stress test

| Population | N | Any-label FPR (95% CI) |
|---|---:|---:|
| Jigsaw all-negative test rows | 14256 | 0.015 [0.013, 0.017] |
| Adjacent-benign stress set | 60 | 0.000 [0.000, 0.060] |
| ↳ clinical_medical | 20 | 0.000 [0.000, 0.161] |
| ↳ news_reporting_violence | 20 | 0.000 [0.000, 0.161] |
| ↳ reclaimed_language | 20 | 0.000 [0.000, 0.161] |

Adjacent-benign FPR gap: -0.015 [-0.017, 0.045].

Interpretation: this curated set is a targeted failure-mode stress test, not a
representative demographic sample. A positive gap means the classifier flags
contextually benign adjacent-domain text more often than ordinary negative
Jigsaw comments. No causal or population-fairness claim follows from it.
