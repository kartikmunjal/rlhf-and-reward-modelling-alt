# Environment Blocker and Execution Alternatives

Status: **WSL2/RTX-3070 execution path unattainable on the available host**  
Recorded: 2026-09-07, before draft training, quantization, pilot selection, or
serving measurements.

This memo records an infrastructure failure. It does not alter the frozen
preregistration and contains no performance finding.

## What was attempted

The Windows 11 host is an ASUS system with an Intel Core i7-10700K and NVIDIA
GeForce RTX 3070. The following work completed:

1. Enabled the `Microsoft-Windows-Subsystem-Linux` and
   `VirtualMachinePlatform` optional components.
2. Installed Microsoft's signed WSL 2.7.13 x64 package.
3. Confirmed `hypervisorlaunchtype Auto`, VM Monitor Mode Extensions, SLAT,
   DEP, and automatic Tailscale/OpenSSH services.
4. Attempted to register Ubuntu under WSL2.

Ubuntu registration failed with
`HCS_E_HYPERV_NOT_INSTALLED`. `systeminfo` and
`Win32_Processor.VirtualizationFirmwareEnabled` both reported firmware
virtualization disabled. This agrees with Microsoft's documented requirement
that hardware-assisted virtualization be enabled in BIOS/UEFI for WSL2.

The operator then enabled Intel VMX in ASUS UEFI. The machine could not boot
normally and displayed an enabling/startup problem. Disabling VMX restored
normal Windows operation. The exact firmware/startup error was not captured,
so no narrower causal claim is justified. Repeated firmware changes were
stopped to avoid risking the only available GPU host.

Sources:

- [Microsoft Hyper-V hardware requirements](https://learn.microsoft.com/en-us/windows-server/virtualization/hyper-v/host-hardware-requirements)
- [Microsoft WSL manual installation requirements](https://learn.microsoft.com/en-us/windows/wsl/install-manual)
- [vLLM quickstart and Linux prerequisite](https://docs.vllm.ai/en/latest/getting_started/quickstart/)

## Consequence for the locked study

The frozen design names WSL2 Ubuntu, vLLM, and the RTX 3070. vLLM's supported
CUDA installation requires Linux. Native Windows cannot execute that design,
and Docker Desktop would still depend on the same unavailable virtualization
layer. WSL1 cannot provide the required CUDA/vLLM path.

Therefore no vLLM result, GPTQ trade-off, or speculative-decoding result has
been produced. This is an environment limitation, not a negative model or
serving result.

## Ranked alternatives

### 1. Recommended: short-lived Linux GPU pod

Use an SSH-accessible Ubuntu pod with one 24 GB RTX 4090 and persistent storage
for the duration of the study. RunPod is the lowest-friction candidate because
it offers on-demand Linux GPU pods, per-second billing, and current RTX 4090
listings. A 24 GB device gives ample margin for the 355M target, 117M draft,
GPTQ calibration, and vLLM KV cache.

This preserves the substantive design:

- actual base/SFT/DPO artifacts;
- real GPT-2-small distillation;
- Hugging Face versus vLLM;
- locked GPTQModel quantization;
- frozen-judge quality validation;
- vLLM speculative decoding and acceptance counters.

It changes the platform and GPU. Before looking at any outputs, add a
timestamped protocol amendment that replaces only `WSL2/RTX 3070` with the
exact pod image and GPU. Preserve all datasets, seeds, trial counts, thresholds,
comparisons, and GRPO exclusion. Results must be described as 4090-specific,
not as validation of 3070 capacity.

RunPod currently advertises RTX 4090 Community Cloud from $0.34/GPU-hour and
Secure Cloud from $0.74/GPU-hour, subject to availability. For budgeting, cost
must be calculated from the provider's displayed rate multiplied by actual pod
uptime; it must not be presented later as a benchmark result.

Source: [RunPod RTX 4090 pricing and specifications](https://www.runpod.io/gpu-models/rtx-4090)

### 2. Acceptable fallback: dedicated GCP L4 VM

A `g2-standard-4` Linux VM supplies one 24 GB L4 and a stable, dedicated
runtime. Google currently lists that complete VM at approximately
$0.7068/hour before disk and network charges. This is more reproducible than a
notebook runtime and preserves the same software study, but account setup,
quota, image configuration, and cost controls are heavier than a GPU pod.

Source: [Google Cloud accelerator-optimized VM pricing](https://cloud.google.com/products/compute/pricing/accelerator-optimized)

### 3. Pilot/debug only: Google Colab

Colab can test installation, artifact loading, and a reduced dry run on Linux.
It is not recommended for the confirmatory matrix: Google explicitly states
that GPU types, usage limits, idle timeouts, and maximum VM lifetime vary and
are not guaranteed; typical runtimes can terminate within 12 hours. That makes
the ten-trial serving matrix and persistent raw ledgers unnecessarily fragile.

Source: [Google Colab resource-limit FAQ](https://research.google.com/colaboratory/faq.html)

### 4. Not equivalent: native Windows inference engine

Using llama.cpp, ONNX Runtime, or another Windows-native runtime could form a
separate engineering study, but it would replace vLLM scheduling, GPTQ with a
different quantization representation, and vLLM's speculative-decoding
instrumentation. It would not answer the preregistered question and must not be
reported as if it did. It is therefore not recommended for this extension.

### 5. Not recommended: another VMX attempt or dual boot

Firmware repair or a native Ubuntu dual boot might eventually expose the RTX
3070 to Linux, but both require physical access and introduce avoidable boot
and data-availability risk. They offer no methodological advantage over a
temporary Linux GPU pod.

## Recommended execution decision

Provision one persistent Ubuntu RTX 4090 pod; record its immutable image,
driver, CUDA, Python, vLLM, GPTQModel, PyTorch, CPU, RAM, and disk details; add
a platform-only preregistration amendment; then run the existing pipeline
unchanged. Keep the Windows RTX 3070 available for unrelated native-PyTorch
work, but do not combine timings across the two GPUs.

The alternative is honest and defensible: the local path was attempted and
failed for a documented firmware constraint, while the research estimands and
precommitted statistical tests remain unchanged on a supported Linux system.
