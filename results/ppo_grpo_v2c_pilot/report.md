# ppo_grpo_v2c_pilot — feasibility result

**Status: PASS**

Primary data: 64 disjoint problems and 256 sampled completions.

| Gate | Observed | Threshold | Result |
|---|---:|---:|---:|
| Numeric parse rate | 1.0000 | >= 0.5000 | PASS |
| Numeric exact rate (minimum) | 0.1055 | >= 0.1000 | PASS |
| Numeric exact rate (maximum) | 0.1055 | <= 0.6500 | PASS |
| Truncation rate | 0.0156 | <= 0.2500 | PASS |
| Groups with reward contrast | 0.6250 | >= 0.5000 | PASS |

Reward mean: 0.1830; reward variance: 0.081289; tagged exact rate: 0.1055.

A failed pilot is retained as a result and cannot be converted into a pass by changing its frozen thresholds.
