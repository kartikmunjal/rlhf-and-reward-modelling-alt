# Alignment-Aware Inference Serving Results

Study: `inference_serving_v1`

All estimates below are generated from raw trials; intervals are 95% bootstrap CIs.
The original Hugging Face Stage-1 rows predate the ledger's `phase` field; they are treated as confirmatory because HF pilot/smoke measurements were written to separate ledgers.

## Stage 1: vLLM minus Hugging Face

| Metric | Difference [95% CI] | N trials | Holm p |
|---|---:|---:|---:|
| output_tokens_per_second | 2755 [2701, 2813] | 10 | 0.008399 |
| request_throughput | 43.04 [42.22, 43.93] | 10 | 0.008399 |
| ttft_ms_p50 | -300.2 [-310.6, -291.3] | 10 | 0.008399 |
| ttft_ms_p95 | -240.4 [-253.5, -229.1] | 10 | 0.008399 |
| itl_ms_p50 | -17 [-17.09, -16.91] | 10 | 0.008399 |
| itl_ms_p95 | -16.36 [-16.49, -16.25] | 10 | 0.008399 |
| peak_gpu_memory_bytes | 1.39e+10 [1.39e+10, 1.39e+10] | 10 | 0.008399 |

Preregistered throughput criterion: **PASS**.

## Stage 2: GPTQ minus FP16

| Metric | Difference [95% CI] | N trials | Holm p |
|---|---:|---:|---:|
| output_tokens_per_second | 212.7 [122.8, 295.9] | 10 | 0.0144 |
| request_throughput | 3.323 [1.982, 4.681] | 10 | 0.0144 |
| ttft_ms_p50 | -14.37 [-26.91, -0.3236] | 10 | 0.1672 |
| ttft_ms_p95 | -12.7 [-27.22, 1.606] | 10 | 0.1672 |
| itl_ms_p50 | -0.2405 [-0.3328, -0.1664] | 10 | 0.008399 |
| itl_ms_p95 | -0.305 [-0.513, -0.1584] | 10 | 0.008399 |
| peak_gpu_memory_bytes | -4.614e+07 [-4.614e+07, -4.614e+07] | 10 | 0.009499 |

Preregistered throughput criterion: **PASS**.

## Stage 3: speculative minus ordinary

| Metric | Difference [95% CI] | N trials | Holm p |
|---|---:|---:|---:|
| output_tokens_per_second | 142.5 [-236.2, 493.1] | 10 | 0.9723 |
| request_throughput | 2.227 [-4.083, 7.717] | 10 | 0.9723 |
| ttft_ms_p50 | -50.89 [-69.4, -32.72] | 10 | 0.009499 |
| ttft_ms_p95 | 40.65 [0.1818, 97.35] | 10 | 0.4905 |
| itl_ms_p50 | 7.959 [7.056, 9.448] | 10 | 0.008399 |
| itl_ms_p95 | 9.691 [7.075, 13.16] | 10 | 0.008399 |
| peak_gpu_memory_bytes | -4.614e+08 [-4.614e+08, -4.614e+08] | 10 | 0.009499 |

Preregistered throughput criterion: **FAIL**.

## Stage 2: frozen-judge quality

Coverage: 168/168 (100.0%; Wilson 95% CI [97.8%, 100.0%]).

| Axis | GPTQ - FP16 [95% CI] | N pairs | Equivalent |
|---|---:|---:|---:|
| relevance | -0.08929 [-0.2321, 0.05952] | 168 | no |
| consistency | 0.02976 [-0.1786, 0.244] | 168 | no |

Preregistered quality-equivalence criterion: **FAIL**.

## Stage 3: draft-token acceptance by alignment state

| Target | Acceptance rate [95% CI] | N trials |
|---|---:|---:|
| base | 71.4% [71.3%, 71.6%] | 10 |
| sft | 71.5% [71.5%, 71.6%] | 10 |
| dpo | 75.2% [75.2%, 75.3%] | 10 |

| Planned contrast | Acceptance-rate difference [95% CI] | N paired trials | Holm p |
|---|---:|---:|---:|
| base minus sft | -0.12% [-0.24%, 0.02%] | 10 | 0.1363 |
| sft minus dpo | -3.71% [-3.81%, -3.62%] | 10 | 0.0036 |

## Scope boundary

No comparable GPT-2-medium GRPO checkpoint exists; existing PPO/GRPO adapters target Qwen2.5-0.5B and a new GRPO run is out of scope.

Serving measurements ran on NVIDIA RTX A5000; the draft alone was trained on an RTX 3070. These measurements validate this concrete stack only and are not multi-GPU or datacenter-scale claims.
