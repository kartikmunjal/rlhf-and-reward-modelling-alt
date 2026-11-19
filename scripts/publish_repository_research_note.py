"""Publish the repository-level research note and README evidence map.

Every empirical number is loaded from a committed metrics artifact. This script
is the sole generator for the note and the delimited README overview.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTE = ROOT / "docs" / "portfolio_research_note.md"
README = ROOT / "README.md"
START = "<!-- REPOSITORY-EVIDENCE-MAP:START -->"
END = "<!-- REPOSITORY-EVIDENCE-MAP:END -->"


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def interval(metric: dict, digits: int = 3) -> str:
    lo, hi = metric["ci95"]
    estimate = metric.get("estimate", metric.get("value"))
    return f"{estimate:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def pct_interval(metric: dict, digits: int = 1) -> str:
    lo, hi = metric["ci95"]
    estimate = metric.get("estimate", metric.get("value"))
    return (
        f"{100 * estimate:.{digits}f}% "
        f"[{100 * lo:.{digits}f}%, {100 * hi:.{digits}f}%]"
    )


def render() -> tuple[str, str]:
    judge = load("results/summeval_judge_v1/metrics.json")
    summary = load("results/summarization_finetune_v1/metrics.json")
    serving = load("results/inference_serving_v1/metrics.json")
    recursive = load("results/recursive_self_improvement_v1/metrics.json")
    safety = load("results/safety_classifier_v1/metrics.json")
    agents = load("results/miscoordination_v1/metrics.json")
    rl = load("results/ppo_grpo_v2/metrics.json")
    rl_v1 = load("results/ppo_grpo_v1/metrics.json")
    safety_v2 = load("results/safety_v2_interim_2026-07-26/matrix_status.json")

    rel = judge["pointwise"]["anthropic"]["axes"]["relevance"]["judge_human"]
    con = judge["pointwise"]["anthropic"]["axes"]["consistency"]["judge_human"]
    sft_rel = summary["comparisons"]["sft_minus_base"]["anthropic"]["relevance"]
    sft_con = summary["comparisons"]["sft_minus_base"]["anthropic"]["consistency"]
    dpo_rel = summary["comparisons"]["dpo_minus_sft"]["anthropic"]["relevance"]
    dpo_con = summary["comparisons"]["dpo_minus_sft"]["anthropic"]["consistency"]
    vllm = serving["stage1"]["metrics"]["output_tokens_per_second"]
    gptq = serving["stage2_performance"]["metrics"]["output_tokens_per_second"]
    spec = serving["stage3_performance"]["metrics"]["output_tokens_per_second"]
    self100 = recursive["stage3_vs_zero_percent"]["100"]
    round6 = recursive["stage1_round_gains"]["iterative_dpo_round_6"]
    fpr = safety["fairness"]["overall_test_fpr"]
    adjacent = safety["fairness"]["adjacent_benign_fpr"]
    miscoord = agents["analysis"]["paired"]["any_miscoordination"]
    actions = agents["analysis"]["paired"]["actions"]
    grpo = rl["accuracy"]["grpo_minus_ppo"]
    grpo_v1 = rl_v1["accuracy"]["grpo_minus_ppo"]
    evaluated_v2 = sum(x["status"] == "complete" for x in safety_v2["trials"])

    overview = f"""{START}
## Evidence map

The repository contains both executable research infrastructure and completed
experiments. The table below lists only studies with committed result artifacts;
historical notebook demonstrations and `--show_expected` outputs are not treated
as empirical findings. See the [two-page research note](docs/portfolio_research_note.md)
for the cross-project argument and limitations.

| Study | Primary artifact-backed finding | Evidence boundary |
|---|---|---|
| [SummEval judge](llm_judge_summeval/) | Claude–human Spearman rho: relevance {interval(rel)}, consistency {interval(con)}; N={rel['n_observations']} | 80 source articles; pairwise coverage missed its frozen floor |
| [Summarization training](summarization_finetune/) | SFT improved relevance {interval(sft_rel)} and consistency {interval(sft_con)}; N={sft_rel['n_trials']} | One training seed; DPO failed the locked two-axis rule |
| [Inference serving](inference_serving/) | vLLM added {interval(vllm, 1)} output tokens/s over HF; N={vllm['n_trials']} | One A5000; GPTQ quality equivalence and speculation throughput failed |
| [Recursive self-improvement](recursive_self_improvement/) | 100% self labels changed round-4 win rate by {interval(self100)} vs 0%; N={self100['n_trials']}, Holm p={self100['holm_adjusted_p_value']:.4f} | One training seed; self-likelihood is a narrow self-reliance proxy |
| [Safety and fairness](results/safety_classifier_v1/report.md) | Any-label FPR {pct_interval(fpr, 2)} on N={fpr['n_examples']:,}; adjacent-benign FPR {pct_interval(adjacent, 2)} on N={adjacent['n_examples']} | V1 complete; V2 selection remains interim ({evaluated_v2}/{safety_v2['n_trials_planned']} evaluated) |
| [Multi-agent coordination](results/miscoordination_v1/report.md) | Shared ledger changed miscoordination by {pct_interval(miscoord)}; N={agents['analysis']['paired']['n_matched_pairs']} matched pairs | Controlled task; global success was already saturated |
| [PPO vs GRPO v1](results/ppo_grpo_v1/report.md) | GRPO−PPO exact match {interval(grpo_v1)}; N_trials={grpo_v1['n_paired_trials']} | Complete negative result at a shared performance floor; protocol deviation retained |
| [PPO vs GRPO v2](results/ppo_grpo_v2/report.md) | GRPO−PPO exact match {interval(grpo)}; N_trials={grpo['n_trials']}, {grpo['n_evaluations_per_trial']} evaluations/trial | Synthetic arithmetic, Qwen LoRA; not pooled with GPT-2 studies |

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
{END}"""

    note = f"""# Research Note: Measuring Alignment Systems Without Hiding Their Failures

**Repository:** `rlhf-and-reward-modelling-alt`<br>
**Scope:** reward modeling, post-training, evaluation, serving, safety, and agent coordination<br>
**Evidence policy:** every number below is generated from the linked committed metrics artifact; illustrative notebook outputs are excluded.

## Research question

How can an alignment pipeline be evaluated as a connected system rather than as
a collection of successful training demos? This repository studies the chain
from reward construction to policy optimization, deployment, and downstream
failure diagnosis. Its central methodological claim is modest: proxy rewards
and automated judges are useful only when their validity, uncertainty, and
failure boundaries are measured independently of the optimization that uses
them.

The work uses preregistered thresholds, held-out evaluation, deterministic
bootstrap or Wilson intervals, paired designs where possible, artifact hashes,
and explicit retention of negative results. The common pattern is: validate a
signal, optimize against it, then test whether optimization exploits weaknesses
in that signal.

## Connected experimental program

### 1. Validate the measurement instrument

The SummEval study froze a Claude pointwise judge before held-out evaluation.
Against expert ratings, Spearman correlation was **{interval(rel)}** for
relevance and **{interval(con)}** for consistency, each on
**N={rel['n_observations']}** summaries clustered over {rel['n_articles']}
source articles. Both passed the preregistered threshold and exceeded ROUGE-L
by confidence intervals that excluded zero. A GPT-5-mini cross-provider audit
provided a second measurement channel. The pairwise variant did not meet its
frozen coverage floor, so its position-bias results were retained as
exploratory rather than promoted to confirmatory evidence.

This matters because the judge becomes a fixed instrument in the next study;
its prompt is not retuned after model outputs are observed.

### 2. Close the training–evaluation loop

On CNN/DailyMail summarization, GPT-2-medium LoRA-SFT improved the frozen judge
by **{interval(sft_rel)}** relevance points and **{interval(sft_con)}**
consistency points on **N={sft_rel['n_trials']}** paired articles. Judge-derived
DPO then improved relevance by **{interval(dpo_rel)}**, but consistency changed
by only **{interval(dpo_con)}**. Because the locked success criterion required
both axes to improve with intervals excluding zero, DPO **failed**. Its length
shift interval included zero and length-controlled analysis did not rescue the
consistency claim. The appropriate conclusion is not that DPO improved human
quality; it is that this judge-optimization loop produced axis-specific evidence
and failed its joint criterion. N_training_seeds={summary['training_seeds']} is
the main generalization limit.

### 3. Measure serving trade-offs on the trained artifact

The inference study moved the same GPT-2-medium family from training into
deployment. On an RTX A5000 at concurrency 32, vLLM produced
**{vllm['left']['estimate']:.1f}** output tokens/s versus
**{vllm['right']['estimate']:.1f}** for Hugging Face generation, a paired gain
of **{interval(vllm, 1)}** across **N_trials={vllm['n_trials']}**. Four-bit GPTQ
added **{interval(gptq, 1)}** tokens/s, but failed the frozen quality-equivalence
margin. Speculative decoding with a separately trained GPT-2-small draft added
**{interval(spec, 1)}** tokens/s; its interval crossed zero, so the throughput
claim failed. DPO nevertheless had higher draft-token acceptance than SFT,
rejecting the preregistered expectation that post-training would monotonically
reduce predictability. These are single-GPU systems measurements, not claims
about multi-node inference.

### 4. Probe iterative and self-generated optimization

The recursive study replaced earlier illustrative iterative-DPO figures with
executed artifacts. Eight rounds showed a local plateau rule firing at round
{recursive['stage1_plateau_round']}, followed by a round-6 gain of
**{interval(round6)}** on **N={round6['n_trials']}** prompts. This is evidence
that the particular short-window rule can declare a local plateau before a
later improvement; one seed cannot show how often that happens.

A separate 20-cell experiment varied self-generated preference labels from
0% to 100%. At round 4, 100% self labels changed external-evaluator win rate by
**{interval(self100)}**, Holm-adjusted p={self100['holm_adjusted_p_value']:.4f},
on **N={self100['n_trials']}** paired prompts. Lower mixtures were not
distinguishable from control. The fully self-labeled policy became shorter,
while KL and reward-ensemble disagreement rose, arguing against a verbosity
exploit and supporting a narrower diagnosis of drift under a weakly aligned
self-likelihood signal. This does not model deliberative self-judgment or
frontier-scale recursive improvement.

### 5. Test safety and coordination failures

The completed safety baseline reports category-level precision, recall, and F1
plus fairness diagnostics. Any-label false-positive rate was
**{pct_interval(fpr, 2)}** on **N={fpr['n_examples']:,}** negative Jigsaw rows.
The adjacent-benign set observed **{adjacent['n_flagged']}/{adjacent['n_examples']}**
flags, **{pct_interval(adjacent, 2)}**; the wide upper bound is important and
prevents interpreting zero observed errors as zero risk. The larger V2 matrix
has trained all planned cells but only {evaluated_v2}/{safety_v2['n_trials_planned']}
evaluations are published, so it has no final selected model.

In the multi-agent study, a shared ledger changed any-miscoordination by
**{pct_interval(miscoord)}** across **N={agents['analysis']['paired']['n_matched_pairs']}**
matched pairs and reduced actions by **{interval(actions, 2)}**. Global task
success remained saturated at 100% in both conditions: the result is about
process efficiency, principally redundant work, not final-answer accuracy.
Unobserved severe failures retain nonzero Wilson upper bounds.

Finally, the preregistered PPO–GRPO v2 comparison supplied reward signal before
policy optimization, correcting an earlier floor-effect study without erasing
it. On a locked arithmetic task, GRPO minus PPO exact match was
**{interval(grpo)}**, with **N_trials={grpo['n_trials']}** and
{grpo['n_evaluations_per_trial']} evaluations per trial. The result is
task-specific and uses a Qwen LoRA family; it is deliberately not merged into
the GPT-2 capability–compute curve.

## What the repository establishes

The strongest contribution is the experimental discipline connecting these
studies. A judge is validated before it evaluates SFT and DPO; trained artifacts
are then subjected to serving and quantization tests; a frozen external ensemble
checks whether self-generated optimization is real or circular; and agent
coordination labels come mechanically from event logs rather than another LLM.
Failed criteria remain visible: pairwise judge coverage, DPO's consistency axis,
GPTQ quality equivalence, speculative throughput, and the first PPO–GRPO floor
effect.

The evidence does **not** establish frontier-scale learning laws, multi-node
training behavior, universal GRPO superiority, or safe recursive improvement.
Most language-model training studies use one training seed, prompt-bootstrap
intervals do not capture seed variation, and several environments are controlled
proxies. The clearest next confirmatory experiment is a separately preregistered
multi-seed plateau study: run full trajectories first, then compare stopping
rules offline so that a rule's false-plateau rate and compute regret can be
measured without selectively truncating evidence.

## Reproduction and audit trail

- Repository overview: [`README.md`](../README.md)
- Judge: [`results/summeval_judge_v1/report.md`](../results/summeval_judge_v1/report.md)
- Summarization: [`results/summarization_finetune_v1/report.md`](../results/summarization_finetune_v1/report.md)
- Serving: [`results/inference_serving_v1/report.md`](../results/inference_serving_v1/report.md)
- Recursive study: [`results/recursive_self_improvement_v1/report.md`](../results/recursive_self_improvement_v1/report.md)
- Safety: [`results/safety_classifier_v1/report.md`](../results/safety_classifier_v1/report.md)
- Coordination: [`results/miscoordination_v1/report.md`](../results/miscoordination_v1/report.md)
- PPO–GRPO: [`results/ppo_grpo_v2/report.md`](../results/ppo_grpo_v2/report.md)

Regenerate this note and the README evidence map with
`python scripts/publish_repository_research_note.py`. Detailed reports and
metrics remain the unit of record; this note is a generated synthesis.
"""
    return overview, note


def render_readme(overview: str) -> str:
    """Render the concise repository index; empirical cells come from ``render``."""
    return f"""[![CI](https://github.com/kartikmunjal/rlhf-and-reward-modelling-alt/actions/workflows/ci.yml/badge.svg)](https://github.com/kartikmunjal/rlhf-and-reward-modelling-alt/actions/workflows/ci.yml)

# RLHF Pipeline: Reward Modeling, Policy Optimization, and Scaling

This repository is a research portfolio for reward modeling, post-training,
evaluation, serving, safety, and agent coordination. Its evidence policy is
simple: a numerical finding belongs in the root overview only when a committed
metrics artifact and named generator reproduce it. Detailed reports—not prose
copied into this page—are the units of record.

{overview}

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
"""


def main() -> None:
    overview, note = render()
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text(note.rstrip() + "\n", encoding="utf-8")
    README.write_text(render_readme(overview), encoding="utf-8")
    print(NOTE.relative_to(ROOT))
    print(README.relative_to(ROOT))


if __name__ == "__main__":
    main()
