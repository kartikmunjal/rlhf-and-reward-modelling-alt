# Alignment-aware inference serving v1 preregistration

Status: **locked before draft training, quantization, pilot selection, or serving
benchmark execution on 2026-09-04**. The machine-readable source of truth is
[`study_config.json`](study_config.json).

## Research questions

1. Does vLLM improve saturated throughput over the repository's naive
   Hugging Face generation path for the actual GPT-2-medium DPO summarizer?
2. What throughput, latency, memory, and frozen-judge quality tradeoff follows
   from 4-bit GPTQ quantization of that same checkpoint?
3. Can a deployable GPT-2-small model distilled from the unaligned
   GPT-2-medium distribution accelerate the target through speculative
   decoding?
4. With the draft fixed, does acceptance change from base to SFT to DPO?

## Artifact scope and explicit GRPO omission

The comparison contains unaligned GPT-2-medium, the existing merged
CNN/DailyMail SFT checkpoint, and the existing merged judge-preference DPO
checkpoint. These form a coherent base → task adaptation → preference
optimization sequence with one tokenizer and architecture family.

**GRPO is excluded because no comparable GPT-2-medium GRPO checkpoint exists
in this repository.** The PPO/GRPO artifacts that do exist are
Qwen2.5-0.5B LoRA adapters and are not a fair model-, task-, or checkpoint-form
comparison. Extending this study to GRPO would require a new alignment run,
which is out of scope. No GRPO result will be manufactured to complete a
matrix.

The earlier repository narrative mentioned a 117M/355M scaling comparison but
did not contain a deployable distilled checkpoint. This study therefore trains
and preserves a real GPT-2-small draft artifact. It is described as a new
artifact gap closure, not as a recovered historical checkpoint.

## Leakage controls and fixed workload

The already locked 200-article CNN/DailyMail final-evaluation partition is
ranked deterministically by article ID and the study seed. Thirty-two articles
form a pilot partition used only for engine feasibility, quantizer calibration
support, and speculative-token-count selection. The remaining 168 articles are
untouched until configurations are frozen and provide confirmatory serving and
quality comparisons.

All engines receive byte-identical prompt text and token-identical inputs.
Greedy decoding, prompt length, completion caps, hardware, and target weights
are fixed within comparisons. Warm-up traffic is excluded. Engine startup and
model loading are reported separately from steady-state request performance.

## Draft training

GPT-2-small is initialized from the pinned upstream revision and distilled
from the pinned unaligned GPT-2-medium teacher, not from SFT or DPO. This avoids
baking privileged closeness to either aligned target into the draft. Training
uses 50,000 deterministically selected, contamination-filtered CNN/DailyMail
training examples for one epoch. The loss is 0.8 temperature-scaled token KL
at T=2 plus 0.2 hard-label causal-LM loss. A separate 2,000-example validation
partition measures token KL and NLL before and after training. Both must
improve for draft training to pass.

## Stage 1: serving engine

Naive Hugging Face generation and vLLM FP16 serve the identical DPO checkpoint.
Each concurrency level (1, 8, 32) uses 10 benchmark trials of 128 requests,
after 16 warm-up requests. The primary claim is positive vLLM minus Hugging
Face output-token throughput at concurrency 32 with a paired trial-bootstrap
95% interval whose lower bound is above zero. TTFT, ITL, request throughput,
and peak GPU allocation are required diagnostics and may disagree with the
throughput result.

## Stage 2: GPTQ

GPTQ is locked at 4 bits, group size 128, symmetric weights, and disabled
activation-order reordering. Calibration uses no confirmatory article. GPTQ
passes the systems criterion only if its throughput improvement over FP16 has
a positive paired-trial interval. It passes the quality criterion only if the
lower bound of quantized-minus-FP16 frozen-Claude score is at least -0.15
points for both relevance and consistency on the 168 held-out articles.
ROUGE-L and a loadable existing reward ensemble are secondary. If the locked
GPTQ stack cannot execute GPT-2, incompatibility is a result; another
quantizer will not be substituted after outcomes are visible.

## Stage 3: speculative decoding

The pilot compares 2, 4, and 6 speculative tokens against quantized DPO and
selects the highest median throughput, breaking ties toward the smaller value.
That choice is frozen before held-out execution. The primary claim requires a
positive paired-trial throughput interval for speculative minus ordinary GPTQ
DPO. Drafted and accepted token counters and their ratio are reported.

The fixed draft is then paired separately with quantized base, SFT, and DPO.
Acceptance-rate contrasts base-minus-SFT and SFT-minus-DPO test whether each
post-training layer changes predictability. The directional hypothesis is
decreasing acceptance because the draft was distilled from the unaligned
teacher, but all directions are reported.

## Statistics and claim boundaries

Every throughput, latency, memory, quality, and acceptance estimate reports
N_trials and a 95% interval. Performance resampling clusters by whole benchmark
trial; quality resampling clusters by article. Two thousand deterministic
bootstrap replicates are used, with Holm correction within each stage's metric
family. Validity rates use Wilson intervals.

This is a single-RTX-3070, single-draft-seed systems study on a 355M model. It
does not establish datacenter-scale serving behavior, multi-GPU scaling, or a
general ordering of alignment algorithms. A speedup without accepted-token
accounting or a quality audit is not considered a successful result.
