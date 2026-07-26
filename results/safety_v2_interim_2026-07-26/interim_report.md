## Safety Classifier & Fairness Extension

This extension turns the reward-model infrastructure into a real
three-label safety classifier: `hate_harassment`, `sexualized`, and
`harmful_violent`. V1 is a completed Jigsaw baseline. V2 is a
preregistered 4-loss × 3-seed study; the table below is an interim
snapshot and is not a final model-selection result.

### Experiment status

| Item | Recorded result |
|---|---|
| V1 | Complete; N_trials=1 |
| V2 training | 12/12 trials trained |
| V2 evaluation snapshot | 5/12 trials evaluated; phase `evaluation` |
| V2 training examples | 227,297: 127,582 Jigsaw + 99,715 unique BeaverTails pairs |
| V2 positive labels | hate/harassment 23,827; sexualized 8,866; harmful/violent 26,708 |
| External evaluation | BeaverTails 11,088; ToxiGen 9,900; HateCheck 3,728; Civil Comments 1,804,874 |
| Uncertainty | 2,000 deterministic bootstrap replicates for model metrics; Wilson/Newcombe intervals for rates and gaps |

The v2 data manifest pins source revisions and hashes. BeaverTails repeated
annotation rows are aggregated by strict per-target majority vote; pairs
tied on any mapped label are excluded under preregistration amendment 1.
External sets never enter training or threshold selection.

### Published v1 baseline

| Category | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Test support |
|---|---:|---:|---:|---:|
| `hate_harassment` | 0.850 [0.833, 0.868] | 0.820 [0.801, 0.838] | 0.835 [0.820, 0.848] | 1,583 |
| `sexualized` | 0.791 [0.765, 0.817] | 0.872 [0.847, 0.894] | 0.830 [0.809, 0.848] | 810 |
| `harmful_violent` | 0.527 [0.397, 0.653] | 0.547 [0.413, 0.682] | 0.537 [0.416, 0.649] | 53 |

V1 descriptive macro F1 is **0.734** (arithmetic mean of the
three category point estimates; category bootstrap intervals are shown
above). Any-label FPR on 14,256 negative Jigsaw rows is
**1.47% [1.29%, 1.68%]**. The frozen
adjacent-benign set has 0/60 flags, or
**0.00% [0.00%, 6.02%]**; its small size
makes the upper confidence bound important.

### Interim v2 results — completed evaluations only

The locked selection metric is model-selection macro F1. Jigsaw test
metrics are disclosed for transparency but cannot choose the winner.

| Trial | Selection macro F1 (95% CI) | Jigsaw test macro F1 (95% CI) | Beaver macro F1 (95% CI) | Beaver violent F1 (95% CI) | Jigsaw FPR (95% CI) |
|---|---:|---:|---:|---:|---:|
| `raw_weighted_bce_seed2025` | 0.737 [0.675, 0.785] | 0.731 [0.690, 0.767] | 0.756 [0.736, 0.774] | 0.832 [0.822, 0.842] | 1.78% [1.58%, 2.01%] |
| `raw_weighted_bce_seed2026` | 0.709 [0.652, 0.759] | 0.723 [0.681, 0.761] | 0.741 [0.722, 0.760] | 0.812 [0.802, 0.822] | 1.81% [1.60%, 2.04%] |
| `unweighted_bce_seed2025` | 0.687 [0.617, 0.744] | 0.735 [0.687, 0.776] | 0.775 [0.756, 0.791] | 0.866 [0.857, 0.876] | 1.30% [1.12%, 1.50%] |
| `unweighted_bce_seed2026` | 0.737 [0.674, 0.789] | 0.748 [0.705, 0.786] | 0.773 [0.754, 0.791] | 0.865 [0.855, 0.874] | 1.14% [0.98%, 1.33%] |
| `unweighted_bce_seed2027` | 0.732 [0.675, 0.781] | 0.742 [0.701, 0.779] | 0.772 [0.755, 0.789] | 0.845 [0.836, 0.854] | 1.35% [1.18%, 1.56%] |

### Interim external-validity and fairness diagnostics

| Trial | ToxiGen recall (95% CI) | ToxiGen F1 (95% CI) | HateCheck accuracy (95% CI) | HateCheck F1 (95% CI) | Civil identity FPR gap (95% CI) |
|---|---:|---:|---:|---:|---:|
| `raw_weighted_bce_seed2025` | 0.252 [0.238, 0.265] | 0.376 [0.359, 0.392] | 0.602 [0.586, 0.618] | 0.689 [0.674, 0.703] | 0.084 [-0.276, 0.090] |
| `raw_weighted_bce_seed2026` | 0.211 [0.198, 0.224] | 0.329 [0.312, 0.346] | 0.577 [0.561, 0.592] | 0.659 [0.643, 0.674] | 0.080 [-0.280, 0.086] |
| `unweighted_bce_seed2025` | 0.204 [0.192, 0.216] | 0.324 [0.307, 0.341] | 0.543 [0.527, 0.559] | 0.625 [0.609, 0.641] | 0.143 [-0.329, 0.513] |
| `unweighted_bce_seed2026` | 0.194 [0.181, 0.207] | 0.312 [0.296, 0.329] | 0.559 [0.543, 0.575] | 0.642 [0.627, 0.656] | 0.059 [-0.300, 0.064] |
| `unweighted_bce_seed2027` | 0.210 [0.198, 0.223] | 0.332 [0.314, 0.348] | 0.545 [0.529, 0.560] | 0.626 [0.610, 0.643] | 0.143 [-0.329, 0.513] |

### What can be concluded at this checkpoint

- **Raw weighted BCE:** selection macro F1 0.723 ± 0.020 SD (N_seeds=2); BeaverTails harmful/violent F1 0.822 ± 0.015 SD (N_seeds=2).
- **Unweighted BCE:** selection macro F1 0.719 ± 0.027 SD (N_seeds=3); BeaverTails harmful/violent F1 0.859 ± 0.012 SD (N_seeds=3).
- **Direct violence supervision transfers:** completed trials produce BeaverTails harmful/violent F1 from 0.812 to 0.866. This is the clearest current mechanical signal from adding direct
violence examples; it is not yet proof of final improvement.
- **Implicit hate remains weak:** ToxiGen F1 spans 0.312–0.376, with recall 0.194–0.252. High precision with low recall indicates domain/construct mismatch,
not merely a threshold-free success.
- **Adjacent-benign behavior is stable so far:** all completed trials have 0/60 flags. Each 0/60 estimate still has a 6.02% Wilson upper bound.
- **Identity-gap claims are unresolved:** completed Civil Comments FPR gap point estimates span 0.059–0.143; all intervals cross zero. Sparse identity intersections create wide intervals, so point-estimate
rankings are not treated as demographic conclusions.
- **No final winner exists yet:** capped-weighted BCE and focal-loss
evaluations are absent from this snapshot. The preregistered selector
must wait for 12/12 evaluations and uses model-selection data—not the
frozen Jigsaw test.

### Reproducibility and implemented safeguards

- Exact 12-trial ledger: four locked losses × seeds 2025, 2026, 2027.
- SHA-256 dataset manifest, pinned Hugging Face revisions, deterministic
  Jigsaw splits, disjoint threshold-calibration/model-selection roles.
- Per-category P/R/F1 and PR-AUC; Brier score, 15-bin ECE, threshold
  sensitivity, fairness slices, bootstrap/Wilson/Newcombe intervals.
- External evaluation on BeaverTails, ToxiGen, HateCheck, and original
  Civil Comments identity columns; adjacent-benign clinical, news, and
  reclaimed-language stress tests.
- Blinded deterministic error-analysis sheet with two independent
  annotation columns and Cohen's kappa; no fabricated annotations.
- Generated model card and inference CLI with input hashing and a
  high-impact-use warning. Raw harmful text and model weights remain
  excluded from Git.

Primary snapshot metrics and the generated interim report are in
[`results/safety_v2_interim_2026-07-26/`](results/safety_v2_interim_2026-07-26/).
Regenerate this entire section with:

```bash
python scripts/summarize_safety_v2_interim.py
```
