[![CI](https://github.com/kartikmunjal/rlhf-and-reward-modelling-alt/actions/workflows/ci.yml/badge.svg)](https://github.com/kartikmunjal/rlhf-and-reward-modelling-alt/actions/workflows/ci.yml)

# RLHF Pipeline: Reward Modeling, Policy Optimization, and Scaling

This repository is a research portfolio for reward modeling, post-training,
evaluation, serving, safety, and agent coordination. Its evidence policy is
simple: a numerical finding belongs in the root overview only when a committed
metrics artifact and named generator reproduce it. Detailed reports—not prose
copied into this page—are the units of record.

<!-- REPOSITORY-EVIDENCE-MAP:START -->
## Evidence map

The repository contains both executable research infrastructure and completed
experiments. The table below lists only studies with committed result artifacts;
historical notebook demonstrations and `--show_expected` outputs are not treated
as empirical findings. See the [two-page research note](docs/portfolio_research_note.md)
for the cross-project argument and limitations.

| Study | Primary artifact-backed finding | Evidence boundary |
|---|---|---|
| [SummEval judge](llm_judge_summeval/) | Claude–human Spearman rho: relevance 0.540 [0.490, 0.592], consistency 0.587 [0.534, 0.642]; N=1274 | 80 source articles; pairwise coverage missed its frozen floor |
| [Summarization training](summarization_finetune/) | SFT improved relevance 0.995 [0.884, 1.111] and consistency 0.596 [0.460, 0.722]; N=198 | One training seed; DPO failed the locked two-axis rule |
| [Inference serving](inference_serving/) | vLLM added 2754.5 [2701.1, 2812.6] output tokens/s over HF; N=10 | One A5000; GPTQ quality equivalence and speculation throughput failed |
| [Recursive self-improvement](recursive_self_improvement/) | 100% self labels changed round-4 win rate by -0.100 [-0.166, -0.036] vs 0%; N=250, Holm p=0.0220 | One training seed; self-likelihood is a narrow self-reliance proxy |
| [Safety and fairness](results/safety_classifier_v1/report.md) | Any-label FPR 1.47% [1.29%, 1.68%] on N=14,256; adjacent-benign FPR 0.00% [0.00%, 6.02%] on N=60 | V1 complete; V2 selection remains interim (5/12 evaluated) |
| [Multi-agent coordination](results/miscoordination_v1/report.md) | Shared ledger changed miscoordination by -14.0% [-24.0%, -4.0%]; N=50 matched pairs | Controlled task; global success was already saturated |
| [PPO vs GRPO v1](results/ppo_grpo_v1/report.md) | GRPO−PPO exact match 0.000 [0.000, 0.000]; N_trials=3 | Complete negative result at a shared performance floor; protocol deviation retained |
| [PPO vs GRPO v2](results/ppo_grpo_v2/report.md) | GRPO−PPO exact match 0.290 [0.253, 0.333]; N_trials=3, 400 evaluations/trial | Synthetic arithmetic, Qwen LoRA; not pooled with GPT-2 studies |

### How to read the repository

1. Start with the [research note](docs/portfolio_research_note.md).
2. Inspect each study's preregistration, generated report, metrics JSON, and provenance hashes.
3. Use `src/` for reusable model/training components, `eval/` for agent harnesses,
   and `scripts/` for named data, training, analysis, and publication entry points.
4. Use the extension status registry below to distinguish completed evidence,
   partial studies, reusable infrastructure, and preregistered work in progress.

Regenerate this section and the research note with:

```bash
python scripts/publish_repository_research_note.py
```
<!-- REPOSITORY-EVIDENCE-MAP:END -->

## Canonical result artifacts

| Study | Design | Generated report | Metrics |
|---|---|---|---|
| SummEval LLM judge | [`llm_judge_summeval/`](llm_judge_summeval/) | [`report.md`](results/summeval_judge_v1/report.md) | [`metrics.json`](results/summeval_judge_v1/metrics.json) |
| Summarization SFT and judge-DPO | [`summarization_finetune/`](summarization_finetune/) | [`report.md`](results/summarization_finetune_v1/report.md) | [`metrics.json`](results/summarization_finetune_v1/metrics.json) |
| Alignment-aware inference serving | [`inference_serving/`](inference_serving/) | [`report.md`](results/inference_serving_v1/report.md) | [`metrics.json`](results/inference_serving_v1/metrics.json) |
| Recursive self-improvement | [`recursive_self_improvement/`](recursive_self_improvement/) | [`report.md`](results/recursive_self_improvement_v1/report.md) | [`metrics.json`](results/recursive_self_improvement_v1/metrics.json) |
| Safety classifier and fairness | [`design`](docs/safety_classifier_extension.md) | [`report.md`](results/safety_classifier_v1/report.md) | [`metrics.json`](results/safety_classifier_v1/metrics.json) |
| Multi-agent miscoordination | [`preregistration`](docs/miscoordination_v1_preregistered_plan.md) | [`report.md`](results/miscoordination_v1/report.md) | [`metrics.json`](results/miscoordination_v1/metrics.json) |
| PPO vs GRPO v1 | [`preregistration`](docs/ppo_grpo_v1_preregistered_plan.md) | [`report.md`](results/ppo_grpo_v1/report.md) | [`metrics.json`](results/ppo_grpo_v1/metrics.json) |
| PPO vs GRPO v2 | [`preregistration`](docs/ppo_grpo_v2_preregistered_plan.md) | [`report.md`](results/ppo_grpo_v2/report.md) | [`metrics.json`](results/ppo_grpo_v2/metrics.json) |

The generated [portfolio research note](docs/portfolio_research_note.md) connects
these studies and records their shared limitations. Pilot reports remain in the
repository for provenance but are not promoted to headline evidence.

## Extension status registry

| Extension | Topic | Evidence status |
|---:|---|---|
| 1 | Constitutional AI / RLAIF | Infrastructure and demonstrations; no canonical standalone result artifact |
| 2 | Reward-model ensembling and hacking diagnostics | Reusable infrastructure; downstream use is reported only inside linked completed studies |
| 3 | Process vs outcome reward models | Infrastructure/demo; no canonical standalone result artifact |
| 4 | LoRA fine-tuning | Reusable method component used by completed studies; not a standalone finding |
| 5 | Synthetic SFT data | Infrastructure/demo; no canonical standalone result artifact |
| 6 | Small-vs-medium model scaling | Historical implementation; no canonical result report |
| 7 | AgentBench-Mini | Harness/infrastructure; historical headline values are not portfolio evidence |
| 8 | Iterative DPO | Superseded by the artifact-backed recursive self-improvement study |
| 9 | Agentic SFT | Infrastructure/demo; no canonical standalone result artifact |
| 10 | GAIA-style evaluation | Mock-tool harness/demo; no canonical standalone result artifact |
| 11 | TTS RLHF | Infrastructure and expected-preview tooling; no executed canonical report |
| 12 | Distributed/scaling analysis | Derived and single-device systems analysis; no multi-node execution claim |
| 13 | Multi-agent systems | Artifact-backed miscoordination study completed; older coordinator headlines are noncanonical |
| 14 | Code-execution agent | Infrastructure/demo; no canonical standalone result artifact |
| 15 | Rubric, programmatic, and composite rewards | Infrastructure/preview studies; no canonical standalone result artifact |
| 16 | Programmatic-verifier reward hacking | Preregistered and implemented; model runs blocked pending a full-VM sandbox gate |

Historical expected/example numbers are intentionally absent from this README.
They remain recoverable from Git history, including the checkpoint tag
`pre-readme-cleanup-2026-11-19`, but must not be cited as measured findings.

## Extension 16: current execution boundary

The programmatic-verifier study has a frozen design, deterministic task builder,
mutation audit, sandbox gates, GRPO pipeline, statistical analysis, and blinded
manual-audit tooling. No model outcome has been observed. A standard RunPod Pod
could not create the nested per-sample containers required by the protocol, so
the prospective platform switch to a full Linux GPU VM is recorded in
[`EXECUTION_REVIEW.md`](verifier_reward_hacking/EXECUTION_REVIEW.md). Training
cannot begin until the unchanged gate suite passes on that VM.

## Reproduction

Create a Python 3.11 environment and install the package:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c constraints-ci.txt -e ".[dev]"
python -m pytest tests/ -q
```

Regenerate the evidence map and research note:

```bash
python scripts/publish_repository_research_note.py
git diff --exit-code -- README.md docs/portfolio_research_note.md
```

GPU training and API-backed studies have separate runbooks, frozen configs, and
environment manifests. The CPU CI suite does not download models or make API
calls.

## Repository map

```text
src/                         reusable data, training, reward, and evaluation code
scripts/                     named experiment, analysis, and publication commands
tests/                       CPU-safe unit, integrity, and provenance tests
configs/                     frozen and operational configurations
results/                     committed reports, metrics, and provenance artifacts
docs/                        preregistrations, runbooks, and generated research note
llm_judge_summeval/          frozen SummEval judge study
summarization_finetune/      SFT and judge-derived DPO study
inference_serving/           serving, quantization, and speculative decoding study
recursive_self_improvement/ iterative and self-generated-signal study
verifier_reward_hacking/     Extension 16 design and execution review
```

## Interpretation rules

- Confidence intervals describe the sampling unit stated in each report; they
  do not automatically capture training-seed uncertainty.
- Public benchmark slices retain contamination risk and are reported separately
  where preregistered.
- A failed success criterion remains a failed result even when a secondary
  metric improves.
- Infrastructure, expected previews, and derived scaling arithmetic are not
  empirical model findings.
- No result here establishes frontier-scale behavior, vendor-wide performance,
  production sandbox security, or general algorithm superiority.

## License and attribution

Dataset licenses and upstream revisions are recorded in each study's data card
or preregistration. Model and API provider terms remain applicable to downloaded
weights and generated outputs.
