# Research Note: Measuring Alignment Systems Without Hiding Their Failures

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
Against expert ratings, Spearman correlation was **0.540 [0.490, 0.592]** for
relevance and **0.587 [0.534, 0.642]** for consistency, each on
**N=1274** summaries clustered over 80
source articles. Both passed the preregistered threshold and exceeded ROUGE-L
by confidence intervals that excluded zero. A GPT-5-mini cross-provider audit
provided a second measurement channel. The pairwise variant did not meet its
frozen coverage floor, so its position-bias results were retained as
exploratory rather than promoted to confirmatory evidence.

This matters because the judge becomes a fixed instrument in the next study;
its prompt is not retuned after model outputs are observed.

### 2. Close the training–evaluation loop

On CNN/DailyMail summarization, GPT-2-medium LoRA-SFT improved the frozen judge
by **0.995 [0.884, 1.111]** relevance points and **0.596 [0.460, 0.722]**
consistency points on **N=198** paired articles. Judge-derived
DPO then improved relevance by **0.101 [0.010, 0.197]**, but consistency changed
by only **0.061 [-0.040, 0.162]**. Because the locked success criterion required
both axes to improve with intervals excluding zero, DPO **failed**. Its length
shift interval included zero and length-controlled analysis did not rescue the
consistency claim. The appropriate conclusion is not that DPO improved human
quality; it is that this judge-optimization loop produced axis-specific evidence
and failed its joint criterion. N_training_seeds=1 is
the main generalization limit.

### 3. Measure serving trade-offs on the trained artifact

The inference study moved the same GPT-2-medium family from training into
deployment. On an RTX A5000 at concurrency 32, vLLM produced
**3809.1** output tokens/s versus
**1054.6** for Hugging Face generation, a paired gain
of **2754.5 [2701.1, 2812.6]** across **N_trials=10**. Four-bit GPTQ
added **212.7 [122.8, 295.9]** tokens/s, but failed the frozen quality-equivalence
margin. Speculative decoding with a separately trained GPT-2-small draft added
**142.5 [-236.2, 493.1]** tokens/s; its interval crossed zero, so the throughput
claim failed. DPO nevertheless had higher draft-token acceptance than SFT,
rejecting the preregistered expectation that post-training would monotonically
reduce predictability. These are single-GPU systems measurements, not claims
about multi-node inference.

### 4. Probe iterative and self-generated optimization

The recursive study replaced earlier illustrative iterative-DPO figures with
executed artifacts. Eight rounds showed a local plateau rule firing at round
2, followed by a round-6 gain of
**0.032 [0.016, 0.049]** on **N=250** prompts. This is evidence
that the particular short-window rule can declare a local plateau before a
later improvement; one seed cannot show how often that happens.

A separate 20-cell experiment varied self-generated preference labels from
0% to 100%. At round 4, 100% self labels changed external-evaluator win rate by
**-0.100 [-0.166, -0.036]**, Holm-adjusted p=0.0220,
on **N=250** paired prompts. Lower mixtures were not
distinguishable from control. The fully self-labeled policy became shorter,
while KL and reward-ensemble disagreement rose, arguing against a verbosity
exploit and supporting a narrower diagnosis of drift under a weakly aligned
self-likelihood signal. This does not model deliberative self-judgment or
frontier-scale recursive improvement.

### 5. Test safety and coordination failures

The completed safety baseline reports category-level precision, recall, and F1
plus fairness diagnostics. Any-label false-positive rate was
**1.47% [1.29%, 1.68%]** on **N=14,256** negative Jigsaw rows.
The adjacent-benign set observed **0/60**
flags, **0.00% [0.00%, 6.02%]**; the wide upper bound is important and
prevents interpreting zero observed errors as zero risk. The larger V2 matrix
has trained all planned cells but only 5/12
evaluations are published, so it has no final selected model.

In the multi-agent study, a shared ledger changed any-miscoordination by
**-14.0% [-24.0%, -4.0%]** across **N=50**
matched pairs and reduced actions by **-0.88 [-1.08, -0.68]**. Global task
success remained saturated at 100% in both conditions: the result is about
process efficiency, principally redundant work, not final-answer accuracy.
Unobserved severe failures retain nonzero Wilson upper bounds.

Finally, the preregistered PPO–GRPO v2 comparison supplied reward signal before
policy optimization, correcting an earlier floor-effect study without erasing
it. On a locked arithmetic task, GRPO minus PPO exact match was
**0.290 [0.253, 0.333]**, with **N_trials=3** and
400 evaluations per trial. The result is
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
