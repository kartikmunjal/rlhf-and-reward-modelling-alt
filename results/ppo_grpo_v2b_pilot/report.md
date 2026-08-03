# ppo_grpo_v2b_pilot — feasibility result

**Status: FAIL**

Primary data: 64 disjoint problems and 256 sampled completions.

| Gate | Observed | Threshold | Result |
|---|---:|---:|---:|
| Numeric parse rate | 1.0000 | >= 0.5000 | PASS |
| Numeric exact rate (minimum) | 0.0820 | >= 0.1000 | FAIL |
| Numeric exact rate (maximum) | 0.0820 | <= 0.6500 | PASS |
| Truncation rate | 0.0625 | <= 0.2500 | PASS |
| Groups with reward contrast | 0.7812 | >= 0.5000 | PASS |

Reward mean: 0.1705; reward variance: 0.065195; tagged exact rate: 0.0820.

A failed pilot is retained as a result and cannot be converted into a pass by changing its frozen thresholds.
