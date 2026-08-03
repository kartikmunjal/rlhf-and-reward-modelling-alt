# ppo_grpo_v2_pilot — feasibility result

**Status: FAIL**

Primary data: 64 disjoint problems and 256 sampled completions.

| Gate | Observed | Threshold | Result |
|---|---:|---:|---:|
| Numeric parse rate | 1.0000 | >= 0.5000 | PASS |
| Numeric exact rate (minimum) | 0.7461 | >= 0.1000 | PASS |
| Truncation rate | 0.0000 | <= 0.2500 | PASS |
| Groups with reward contrast | 0.3281 | >= 0.5000 | FAIL |

Reward mean: 0.7611; reward variance: 0.168650; tagged exact rate: 0.7461.

A failed pilot is retained as a result and cannot be converted into a pass by changing its frozen thresholds.
