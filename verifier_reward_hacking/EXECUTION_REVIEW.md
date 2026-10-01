# Extension 16 Execution Review

## Purpose of this memo

This memo records the requested study, the implementation completed so far,
the RunPod feasibility test, the exact infrastructure blocker, and the
recommended paths forward. It is an execution review, not a results report.
No pilot, eligibility-model run, GRPO training run, or confirmatory evaluation
has started.

## Original ask

The requested extension asks whether a language-model policy optimized by GRPO
against an incomplete programmatic verifier learns to exploit that verifier,
and whether three verifier-hardening strategies reduce exploitation at a fixed
training budget.

The frozen design compares:

- C0: the unmodified base policy;
- C1: GRPO trained against three visible tests;
- C2: GRPO trained against mutation-hardened tests;
- C3: GRPO trained against deterministic property tests; and
- C4: GRPO trained against the conjunction of the visible and property tests.

The primary outcomes are hidden-test pass rate, visible-pass/hidden-fail
exploit rate, and the visible-minus-hidden gap. The study also tests whether
pre-training mutation-kill rate predicts later exploitability. The full locked
scope, task counts, seeds, budgets, inference rules, manual audits, and
statistical decision rules are in
[`PREREGISTRATION.md`](PREREGISTRATION.md) and
[`study_config.json`](study_config.json).

## What was implemented

The implementation was committed and pushed in three checkpoints:

- `87d04ae`: deterministic programmatic tasks, pinned HumanEval ingestion,
  AST mutation generation, verifier audit, rootless-container runner, and
  blocking sandbox gates;
- `5f6a3e4`: resumable base-model QA, frozen eligibility selection, C1–C4
  sandboxed rewards, manifest-first GRPO training, and held-out evaluation;
- `6f6b5c6`: hierarchical bootstrap analysis, Holm-adjusted comparisons,
  mutation-kill correlation, rollout preservation, deterministic blinded audit
  sampling, and task-clustered manual-audit analysis.

The focused automated suite passed before cloud execution. The repository and
the user's Downloads clone were synchronized through `6f6b5c6`. Hidden tests,
reference solutions, generated model code, checkpoints, and raw trajectory
ledgers remain excluded from Git by design.

The approved pre-run execution choices were:

- a short C1/seed-2025 debugging pilot;
- a bounded four-configuration dev grid using only the preregistered learning
  rate and KL-coefficient degrees of freedom; and
- all confirmatory conditions on one on-demand Linux GPU platform.

## What was tried on RunPod

A standard RunPod PyTorch Pod was deployed with one NVIDIA RTX A5000, a
container disk, and persistent storage mounted at `/workspace`. SSH access,
the GPU, Python, and persistent storage all worked.

Before copying data or running a model, the environment was checked for the
locked isolation requirements. The diagnostic established that:

- neither Docker nor Podman was present;
- the process was already running inside the provider's outer container;
- the process did not have the kernel capabilities needed for nested container
  creation; and
- `unshare -Ur true` failed with `Operation not permitted`.

This last result is decisive. Installing a Docker or Podman executable inside
the Pod would not grant the missing namespace and host capabilities. A custom
PyTorch image could package more binaries, but it would still run inside the
same restricted outer-container boundary unless RunPod explicitly supplied a
privileged Docker-in-Docker or equivalent VM product.

No study artifacts were generated on the Pod and no model outcome was
observed. The failed infrastructure check therefore does not contaminate task
selection, hyperparameter selection, or confirmatory inference.

## Why this is a blocking issue

The preregistration requires every piece of model-generated Python to execute
in a newly disposable Linux container with no network, no credentials, a
read-only root, resource limits, and host-held tests. It also declares a failed
sandbox gate to be a hard stop.

Running generated programs directly in the training Pod, in a plain Python
subprocess, or under a weaker temporary-directory wrapper would change a
locked safety condition. Splitting generation onto RunPod and execution onto a
different service would also break the frozen same-platform boundary. Neither
change can be adopted merely because the original provider is inconvenient.

The holdback is therefore not GPU memory, model compatibility, persistent
storage, or an incorrect RunPod deployment setting. It is the absence of the
kernel-level control required to construct the preregistered per-sample
isolation boundary.

## Recommended alternatives

### 1. Full Linux GPU virtual machine — recommended

Use a genuine VM with administrator control over Docker, rather than a managed
GPU container. The first action on any candidate VM remains the blocking
sandbox-gate script; provider marketing or the presence of a `docker` command
is not treated as proof.

Suitable shapes include:

- Google Compute Engine G2 with one NVIDIA L4. Google documents G2 as an
  accelerator-optimized VM family and provides an official GPU-VM creation
  workflow: <https://docs.cloud.google.com/compute/docs/gpus/create-gpu-vm-g-series>.
- Lambda On-Demand Cloud with one NVIDIA A10. Lambda documents this as a
  Linux-based GPU virtual machine with one 24 GB A10:
  <https://docs.lambda.ai/public-cloud/on-demand/>.
- AWS EC2 `g5.xlarge` with one NVIDIA A10G. AWS documents G5 as a VM family
  using 24 GB A10G GPUs: <https://aws.amazon.com/ec2/instance-types/g5/>.

The VM should be on-demand, use Ubuntu, have sufficient local disk for model
caches and checkpoints, and permit Docker with the locked CPU, memory,
process, filesystem, network, and timeout controls. All conditions must remain
on the same instance type and software image.

### 2. RunPod only with explicit nested-isolation support

RunPod remains viable only if support can provide a product or deployment mode
that explicitly enables nested containers or equivalent per-job isolation and
passes the committed gate suite unchanged. Recreating the same standard
PyTorch Pod, increasing disk, or choosing another normal template is not an
adequate fix.

### 3. Formal preregistration amendment — not recommended while a VM is viable

A dated, pre-outcome amendment could replace per-sample containers with a
different audited sandbox such as a dedicated microVM service. That would need
a new threat model, new blocking gates, and an explicit statement that the
execution design changed. A plain subprocess sandbox is not an acceptable
research-grade substitute for this study.

## Recommended next action

**2026-10-01 — Prospective platform note:** execution is switching from a standard RunPod Pod to a full Linux GPU VM because the tested RunPod environment blocked the nested user namespaces and per-sample containers required by the frozen sandbox protocol; no model or task-selection outcome had been observed at this decision point.

Provision one full Linux GPU VM, verify that Docker can launch an unprivileged,
network-disabled, read-only child container, and run
`scripts/run_verifier_sandbox_gates.py`. Only after its manifest reports a
pass should the base-model eligibility screen begin. If the gate fails, stop
the instance and preserve the failure record; do not weaken the frozen test.

## Honest current status

The research design and execution software are implemented and versioned. The
candidate task pool exists, but the base-model eligibility gate has not run and
the final corpus is not frozen. There are no empirical Extension 16 findings
yet. The only observed cloud result is an infrastructure incompatibility:
standard RunPod Pods, as tested, did not expose the nested isolation required
by this preregistered study.
