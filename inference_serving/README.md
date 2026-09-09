# Alignment-Aware Inference Serving

This extension connects the repository's real GPT-2-medium summarization
artifacts to serving systems: Hugging Face versus vLLM, locked GPTQ
quantization with frozen-judge quality evaluation, and speculative decoding
with a newly materialized GPT-2-small draft.

The design is frozen in [`preregistration.md`](preregistration.md) and
[`study_config.json`](study_config.json). In particular, GRPO is intentionally
absent because the repository has no comparable GPT-2-medium GRPO artifact;
the existing GRPO adapters use Qwen2.5-0.5B.

No performance result is claimed until the WSL2 environment, exact artifacts,
raw per-request timing records, acceptance counters, and generated analysis
are complete.

## Reproducible execution

Run the Linux serving stack inside WSL2; vLLM is not supported by the native
Windows Python environment used for training. Install with
`bash scripts/setup_inference_wsl.sh "$PWD"`. The script creates an isolated
separate `.venv-vllm` and `.venv-gptq` environments, installs the
preregistered versions, saves both resolved package locks, and verifies GPU
visibility. The separation is required because the locked releases have
incompatible protobuf constraints; GPTQ creates the artifact and vLLM serves
it, so no method or version is substituted. Draft training uses
`scripts/train_gpt2_small_draft.py`; it is resumable and writes its validation
metrics from the locked data split. Model servers are launched only through
`scripts/run_vllm_server.sh`, and every measurement is appended to the raw
trial ledger before generated analysis is run.

The speculative pilot must cover k=2, 4, and 6 with the locked trial count.
`scripts/select_speculative_tokens.py` then creates a one-time selection file;
it refuses to overwrite that decision. Only the selected value is allowed in
the confirmatory held-out run.

## Local execution blocker

The WSL2 path on the available RTX 3070 host was declared unattainable before
any study run. See
[`environment_blocker_and_alternatives.md`](environment_blocker_and_alternatives.md)
for the observed failure, claim boundary, and ranked execution alternatives.
The frozen preregistration has not been edited and no performance result is
claimed.

The approved hybrid execution change is separately frozen in
[`protocol_amendment_001_hybrid.json`](protocol_amendment_001_hybrid.json).
It assigns draft training only to native Windows/RTX 3070 and assigns every
quantization and comparative serving measurement to one on-demand Ubuntu/RTX
A5000 pod. All scientific choices and thresholds remain unchanged.

<!-- INFERENCE-SERVING-RESULTS:START -->
## Alignment-Aware Inference Serving Extension

This completed, preregistered extension serves the repository's real GPT-2-medium DPO artifact, quantizes the same base/SFT/DPO family with 4-bit GPTQ, and uses a trained GPT-2-small draft for speculative decoding. All comparative serving measurements ran on the same NVIDIA RTX A5000; only draft training ran on the RTX 3070.

**Stage 1 — PASS.** At concurrency 32, vLLM delivered 3809.1 output tokens/s versus 1054.6 for Hugging Face `generate()`: a paired gain of 2754.5 (95% CI 2701.1 to 2812.6; N_trials=10). This is a 3.61x throughput ratio. vLLM's larger device-memory reading reflects its preallocated KV cache, so it is not a model-weight footprint comparison.

**Stage 2 performance — PASS.** GPTQ delivered 4021.7 versus 3809.1 output tokens/s for FP16, a paired gain of 212.7 (95% CI 122.8 to 295.9; N_trials=10). **Stage 2 quality equivalence — FAIL.** Across 168/168 held-out articles, GPTQ minus FP16 was -0.089 (95% CI -0.232 to 0.060) for relevance and 0.030 (95% CI -0.179 to 0.244) for consistency. Both lower bounds had to remain at or above the frozen −0.15-point margin; neither did.

**Stage 3 — FAIL on throughput.** The pilot selected k=2 from locked k=2/4/6 candidates (median output tokens/s: k=2: 3062.5, k=4: 2717.0, k=6: 2558.0). On held-out DPO trials, speculative decoding delivered 4164.3 versus 4021.7 output tokens/s, a paired difference of 142.5 (95% CI -236.2 to 493.1; N_trials=10); its interval includes zero.

Draft-token acceptance was base 71.42% (95% CI 71.30%–71.55%), SFT 71.53% (71.47%–71.60%), and DPO 75.25% (75.18%–75.32%), each over N_trials=10. Base minus SFT was -0.12% (95% CI -0.24% to 0.02%); SFT minus DPO was -3.71% (-3.81% to -3.62%; Holm-adjusted p=0.0036). The preregistered expectation that acceptance would decrease with post-training is therefore rejected for these artifacts.

Interpretation is deliberately narrow: N_training_seeds=1; the 355M target is a systems-validation workload; vLLM used its V1 runner for draft-model speculation; and these A5000 results do not establish datacenter-scale or multi-GPU behavior. GRPO remains excluded because the available GRPO artifact uses a different Qwen base. A failed base-server startup caused by `ninja` not being on the subprocess PATH was preserved as an audit log and occurred before measurement.

See [`inference_serving/`](inference_serving/), [`results/inference_serving_v1/report.md`](results/inference_serving_v1/report.md), and [`results/inference_serving_v1/metrics.json`](results/inference_serving_v1/metrics.json).
<!-- INFERENCE-SERVING-RESULTS:END -->
