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
