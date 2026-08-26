# SummEval LLM-as-Judge v1 — preregistered result

## Integrity

- Study status: `primary_complete_secondary_infrastructure_incomplete`.
- Primary model: pinned Claude Haiku 4.5 snapshot.
- Secondary model: pinned GPT-5 mini snapshot; amendment 001 omitted its unsupported temperature field.
- Claude pairwise recovery used the additional provider-error attempt authorized by amendment 001.
- Confidence intervals: 2,000 source-article or unordered-pair cluster-bootstrap replicates, as preregistered.

## Data completeness and API usage

| Component | Valid / preregistered N_trials | Validity rate (Wilson 95% CI) | Input tokens | Output tokens |
|---|---:|---:|---:|---:|
| Claude pointwise | 1274 / 1280 | 0.995 [0.990, 0.998] | 1806655 | 392137 |
| GPT-5 mini pointwise | 1223 / 1280 | 0.955 [0.943, 0.965] | 1143767 | 882668 |
| Claude pairwise | 348 / 400 | 0.870 [0.833, 0.899] | 1200095 | 230324 |

Recorded-token API cost at the versioned runtime rates: Claude $6.1191, OpenAI $2.0513, total $8.1703.

## Pointwise human correlation

| Provider | Axis | Judge vs human Spearman ρ (95% CI) | ROUGE-L vs human ρ (95% CI) | Judge − ROUGE-L (95% CI) | Length bias ρ (95% CI) | Judge vs human, controlling length (95% CI) |
|---|---|---:|---:|---:|---:|---:|
| anthropic | coherence | 0.616 [0.571, 0.661] (N=1274; 80 source_article clusters) | 0.202 [0.124, 0.277] (N=1274; 80 source_article clusters) | 0.414 [0.344, 0.490] (N=1274; 80 source_article clusters) | -0.083 [-0.141, -0.024] (N=1274; 80 source_article clusters) | 0.599 [0.551, 0.646] (N=1274; 80 source_article clusters) |
| anthropic | consistency | 0.587 [0.534, 0.642] (N=1274; 80 source_article clusters) | 0.156 [0.087, 0.224] (N=1274; 80 source_article clusters) | 0.430 [0.339, 0.521] (N=1274; 80 source_article clusters) | 0.100 [0.034, 0.169] (N=1274; 80 source_article clusters) | 0.691 [0.639, 0.738] (N=1274; 80 source_article clusters) |
| anthropic | fluency | 0.439 [0.383, 0.490] (N=1274; 80 source_article clusters) | 0.133 [0.061, 0.202] (N=1274; 80 source_article clusters) | 0.306 [0.233, 0.381] (N=1274; 80 source_article clusters) | -0.230 [-0.288, -0.170] (N=1274; 80 source_article clusters) | 0.355 [0.293, 0.420] (N=1274; 80 source_article clusters) |
| anthropic | relevance | 0.540 [0.490, 0.592] (N=1274; 80 source_article clusters) | 0.276 [0.199, 0.348] (N=1274; 80 source_article clusters) | 0.264 [0.188, 0.344] (N=1274; 80 source_article clusters) | 0.101 [0.032, 0.168] (N=1274; 80 source_article clusters) | 0.519 [0.469, 0.570] (N=1274; 80 source_article clusters) |
| openai | coherence | 0.567 [0.525, 0.611] (N=1223; 80 source_article clusters) | 0.207 [0.127, 0.283] (N=1223; 80 source_article clusters) | 0.360 [0.288, 0.441] (N=1223; 80 source_article clusters) | 0.020 [-0.046, 0.085] (N=1223; 80 source_article clusters) | 0.541 [0.485, 0.583] (N=1223; 80 source_article clusters) |
| openai | consistency | 0.656 [0.603, 0.709] (N=1223; 80 source_article clusters) | 0.151 [0.081, 0.220] (N=1223; 80 source_article clusters) | 0.505 [0.411, 0.598] (N=1223; 80 source_article clusters) | 0.092 [0.032, 0.152] (N=1223; 80 source_article clusters) | 0.789 [0.740, 0.830] (N=1223; 80 source_article clusters) |
| openai | fluency | 0.457 [0.404, 0.505] (N=1223; 80 source_article clusters) | 0.134 [0.062, 0.205] (N=1223; 80 source_article clusters) | 0.323 [0.236, 0.413] (N=1223; 80 source_article clusters) | -0.100 [-0.160, -0.040] (N=1223; 80 source_article clusters) | 0.256 [0.188, 0.459] (N=1223; 80 source_article clusters) |
| openai | relevance | 0.527 [0.471, 0.583] (N=1223; 80 source_article clusters) | 0.275 [0.195, 0.349] (N=1223; 80 source_article clusters) | 0.252 [0.164, 0.340] (N=1223; 80 source_article clusters) | 0.269 [0.212, 0.327] (N=1223; 80 source_article clusters) | 0.486 [0.430, 0.543] (N=1223; 80 source_article clusters) |

Primary success: **True**. relevance=True, consistency=True

## Cross-provider agreement (amended secondary analysis)

| Axis | Claude vs GPT-5 mini Spearman ρ (95% CI) |
|---|---:|
| coherence | 0.731 [0.701, 0.758] (N=1219; 80 source_article clusters) |
| consistency | 0.754 [0.715, 0.789] (N=1219; 80 source_article clusters) |
| fluency | 0.652 [0.609, 0.691] (N=1219; 80 source_article clusters) |
| relevance | 0.677 [0.637, 0.714] (N=1219; 80 source_article clusters) |

## Pairwise position bias and mitigation (exploratory; infrastructure-incomplete)

| Axis | Preference flip rate | Tie instability | First-order human agreement | Symmetrized human agreement | BT vs human ρ (95% CI) | Symmetrized − first-order agreement (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| coherence | 0.126 [0.092, 0.161] (N=348; 348 unordered_pair clusters) | 0.040 [0.020, 0.063] (N=348; 348 unordered_pair clusters) | 0.687 [0.638, 0.733] (N=348; 348 unordered_pair clusters) | 0.618 [0.566, 0.670] (N=348; 348 unordered_pair clusters) | 0.372 [0.309, 0.434] (N=696; 80 source_article clusters) | -0.069 [-0.101, -0.040] (N=348; 348 unordered_pair clusters) |
| consistency | 0.101 [0.072, 0.132] (N=348; 348 unordered_pair clusters) | 0.164 [0.126, 0.204] (N=348; 348 unordered_pair clusters) | 0.586 [0.532, 0.638] (N=348; 348 unordered_pair clusters) | 0.658 [0.609, 0.707] (N=348; 348 unordered_pair clusters) | 0.311 [0.252, 0.371] (N=696; 80 source_article clusters) | 0.072 [0.034, 0.115] (N=348; 348 unordered_pair clusters) |
| fluency | 0.132 [0.098, 0.170] (N=348; 348 unordered_pair clusters) | 0.017 [0.006, 0.032] (N=348; 348 unordered_pair clusters) | 0.368 [0.316, 0.420] (N=348; 348 unordered_pair clusters) | 0.451 [0.399, 0.503] (N=348; 348 unordered_pair clusters) | 0.252 [0.183, 0.321] (N=696; 80 source_article clusters) | 0.083 [0.049, 0.118] (N=348; 348 unordered_pair clusters) |
| relevance | 0.164 [0.126, 0.201] (N=348; 348 unordered_pair clusters) | 0.003 [0.000, 0.009] (N=348; 348 unordered_pair clusters) | 0.649 [0.598, 0.698] (N=348; 348 unordered_pair clusters) | 0.589 [0.540, 0.644] (N=348; 348 unordered_pair clusters) | 0.323 [0.263, 0.381] (N=696; 80 source_article clusters) | -0.060 [-0.095, -0.029] (N=348; 348 unordered_pair clusters) |

## Interpretation boundary

The confirmatory claim applies only to the pinned Claude model, frozen prompt, expert-mean SummEval labels, and document-disjoint held-out split. Secondary axes, amended cross-provider agreement, pairwise diagnostics, and mitigation results cannot rescue failure of either primary axis.

Pairwise coverage did not reach the preregistered validity floor. Those estimates describe the complete observed pairs only; provider missingness may be non-random, so they are explicitly exploratory and cannot support a general pairwise claim. The symmetrization prediction is axis-dependent rather than generally supported.
