# SummEval prompt-development diagnostics

Development partition only; these results are not confirmatory and do not use held-out labels.

Valid responses: 317 / 320 (Wilson 95% CI 0.973–0.997).

| Axis | Judge vs human Spearman rho (95% CI) | ROUGE-L vs human (95% CI) | Judge - ROUGE-L (95% CI) |
|---|---:|---:|---:|
| coherence | 0.588 [0.485, 0.684] (N=317; 20 source_article clusters) | 0.013 [-0.136, 0.172] (N=317; 20 source_article clusters) | 0.575 [0.379, 0.755] (N=317; 20 source_article clusters) |
| consistency | 0.680 [0.593, 0.754] (N=317; 20 source_article clusters) | 0.092 [-0.003, 0.199] (N=317; 20 source_article clusters) | 0.588 [0.449, 0.718] (N=317; 20 source_article clusters) |
| fluency | 0.412 [0.327, 0.496] (N=317; 20 source_article clusters) | 0.030 [-0.075, 0.145] (N=317; 20 source_article clusters) | 0.383 [0.267, 0.485] (N=317; 20 source_article clusters) |
| relevance | 0.516 [0.390, 0.624] (N=317; 20 source_article clusters) | 0.219 [0.023, 0.387] (N=317; 20 source_article clusters) | 0.297 [0.123, 0.474] (N=317; 20 source_article clusters) |
