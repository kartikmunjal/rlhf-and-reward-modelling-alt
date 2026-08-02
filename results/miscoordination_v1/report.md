# Multi-agent miscoordination v1

- Episodes: 100 (50 matched pairs)
- API calls: 400
- Model: `claude-haiku-4-5-20251001`
- Measured API cost: $0.3263
- API-error episodes: 0
- Bootstrap replicates: 2000

Rates are episode-level with deterministic bootstrap 95% intervals.
Differences are paired shared-ledger minus isolated estimates.
Wilson rate intervals are included as a post-run boundary sensitivity
because ordinary bootstrap intervals collapse at observed rates of 0% or 100%.

| Outcome | Isolated (95% CI) | Shared ledger (95% CI) | Paired difference (95% CI) |
|---|---:|---:|---:|
| `global_success` | 100.0% [boot 100.0%, 100.0%; Wilson 92.9%, 100.0%] | 100.0% [boot 100.0%, 100.0%; Wilson 92.9%, 100.0%] | +0.0 pp [+0.0, +0.0] |
| `any_miscoordination` | 14.0% [boot 4.0%, 24.0%; Wilson 7.0%, 26.2%] | 0.0% [boot 0.0%, 0.0%; Wilson 0.0%, 7.1%] | -14.0 pp [-24.0, -4.0] |
| `redundant_work` | 14.0% [boot 4.0%, 24.0%; Wilson 7.0%, 26.2%] | 0.0% [boot 0.0%, 0.0%; Wilson 0.0%, 7.1%] | -14.0 pp [-24.0, -4.0] |
| `direct_contradiction` | 2.0% [boot 0.0%, 6.0%; Wilson 0.4%, 10.5%] | 0.0% [boot 0.0%, 0.0%; Wilson 0.0%, 7.1%] | -2.0 pp [-6.0, +0.0] |
| `silent_undo` | 0.0% [boot 0.0%, 0.0%; Wilson 0.0%, 7.1%] | 0.0% [boot 0.0%, 0.0%; Wilson 0.0%, 7.1%] | +0.0 pp [+0.0, +0.0] |
| `communication_breakdown` | 0.0% [boot 0.0%, 0.0%; Wilson 0.0%, 7.1%] | 0.0% [boot 0.0%, 0.0%; Wilson 0.0%, 7.1%] | +0.0 pp [+0.0, +0.0] |

## Coordination overhead

Means use episode-bootstrap 95% intervals; differences are paired.

| Metric | Isolated mean (95% CI) | Shared-ledger mean (95% CI) | Paired difference (95% CI) |
|---|---:|---:|---:|
| `api_calls` | 4.00 [4.00, 4.00] | 4.00 [4.00, 4.00] | +0.00 [+0.00, +0.00] |
| `input_tokens` | 1020.74 [1020.60, 1020.86] | 1616.12 [1603.34, 1629.44] | +595.38 [+582.18, +608.18] |
| `output_tokens` | 415.58 [393.46, 437.77] | 362.36 [345.24, 380.40] | -53.22 [-79.89, -26.04] |
| `cost_usd` | 0.00310 [0.00299, 0.00321] | 0.00343 [0.00334, 0.00352] | +0.00033 [+0.00019, +0.00047] |
| `actions` | 3.92 [3.72, 4.10] | 3.04 [3.00, 3.10] | -0.88 [-1.08, -0.68] |
| `messages` | 4.00 [4.00, 4.00] | 4.00 [4.00, 4.00] | +0.00 [+0.00, +0.00] |

The taxonomy is mechanically derived from the shared-state event log;
no language-model judge assigns failure labels. Categories may co-occur.
Interpretation is limited to this controlled deployment task and model.
