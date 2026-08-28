# Summarization SFT + Judge-DPO v1 result

## Preregistered decisions

- SFT primary success: **True** ({'relevance': True, 'consistency': True}).
- DPO primary success: **False** ({'relevance': True, 'consistency': False}).
- Training seeds: 1 (article bootstrap does not estimate seed variability).

## Infrastructure validity

| Provider/model | Valid / planned | Wilson 95% CI |
|---|---:|---:|
| anthropic/base | 200/200 | [0.981, 1.000] |
| anthropic/sft | 198/200 | [0.964, 0.997] |
| anthropic/dpo | 200/200 | [0.981, 1.000] |
| openai/base | 183/200 | [0.868, 0.946] |
| openai/sft | 121/200 | [0.536, 0.670] |
| openai/dpo | 131/200 | [0.587, 0.717] |
| anthropic/DPO candidate pairs | 1435/1536 | [0.921, 0.946] |

## Paired score changes

| Comparison | Provider | Axis | Mean change (95% CI) |
|---|---|---|---:|
| sft_minus_base | anthropic | relevance | 0.995 [0.884, 1.111] (N_trials=198; 2,000 bootstraps) |
| sft_minus_base | anthropic | consistency | 0.596 [0.460, 0.722] (N_trials=198; 2,000 bootstraps) |
| sft_minus_base | anthropic | coherence | 1.116 [0.970, 1.263] (N_trials=198; 2,000 bootstraps) |
| sft_minus_base | anthropic | fluency | 0.894 [0.753, 1.035] (N_trials=198; 2,000 bootstraps) |
| sft_minus_base | openai | relevance | 1.161 [0.964, 1.348] (N_trials=112; 2,000 bootstraps) |
| sft_minus_base | openai | consistency | 0.821 [0.580, 1.071] (N_trials=112; 2,000 bootstraps) |
| sft_minus_base | openai | coherence | 1.116 [0.920, 1.321] (N_trials=112; 2,000 bootstraps) |
| sft_minus_base | openai | fluency | 0.795 [0.509, 1.062] (N_trials=112; 2,000 bootstraps) |
| sft_minus_base | reference proxy | ROUGE-L | 0.065 [0.054, 0.076] (N_trials=200; 2,000 bootstraps) |
| sft_minus_base | diagnostic | length words | -29.115 [-36.906, -21.519] (N_trials=200; 2,000 bootstraps) |
| dpo_minus_sft | anthropic | relevance | 0.101 [0.010, 0.197] (N_trials=198; 2,000 bootstraps) |
| dpo_minus_sft | anthropic | consistency | 0.061 [-0.040, 0.162] (N_trials=198; 2,000 bootstraps) |
| dpo_minus_sft | anthropic | coherence | 0.000 [-0.096, 0.091] (N_trials=198; 2,000 bootstraps) |
| dpo_minus_sft | anthropic | fluency | -0.025 [-0.126, 0.081] (N_trials=198; 2,000 bootstraps) |
| dpo_minus_sft | openai | relevance | -0.034 [-0.205, 0.136] (N_trials=88; 2,000 bootstraps) |
| dpo_minus_sft | openai | consistency | -0.068 [-0.295, 0.170] (N_trials=88; 2,000 bootstraps) |
| dpo_minus_sft | openai | coherence | -0.125 [-0.284, 0.034] (N_trials=88; 2,000 bootstraps) |
| dpo_minus_sft | openai | fluency | -0.045 [-0.261, 0.159] (N_trials=88; 2,000 bootstraps) |
| dpo_minus_sft | reference proxy | ROUGE-L | -0.000 [-0.006, 0.005] (N_trials=200; 2,000 bootstraps) |
| dpo_minus_sft | diagnostic | length words | -0.555 [-2.645, 1.565] (N_trials=200; 2,000 bootstraps) |

## DPO length-hacking diagnostic

| Axis | Length-controlled DPO treatment effect | Classification |
|---|---:|---|
| relevance | 0.098 [0.007, 0.184] (N_trials=198; 2,000 bootstraps) | `inconclusive` |
| consistency | 0.057 [-0.043, 0.153] (N_trials=198; 2,000 bootstraps) | `inconclusive` |

## Interpretation boundary

Claude pointwise scores are confirmatory. GPT-5-mini and ROUGE-L are independent but imperfect proxies. A positive length-controlled effect is evidence against verbosity as this particular exploitation mechanism, not proof equivalent to new human ratings. Pairwise preference or evaluation coverage below the locked 90% floor makes the affected DPO claim infrastructure-incomplete. Results generalize over sampled articles, not training seeds.
