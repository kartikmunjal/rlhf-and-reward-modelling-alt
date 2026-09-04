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
`.venv-serving`, installs the preregistered vLLM and GPTQModel versions, saves
the resolved package lock, and verifies GPU visibility. Draft training uses
`scripts/train_gpt2_small_draft.py`; it is resumable and writes its validation
metrics from the locked data split. Model servers are launched only through
`scripts/run_vllm_server.sh`, and every measurement is appended to the raw
trial ledger before generated analysis is run.

The speculative pilot must cover k=2, 4, and 6 with the locked trial count.
`scripts/select_speculative_tokens.py` then creates a one-time selection file;
it refuses to overwrite that decision. Only the selected value is allowed in
the confirmatory held-out run.
