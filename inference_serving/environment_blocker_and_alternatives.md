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

### 1. Cheapest and recommended: hybrid RTX 3070 training plus A5000 serving

Train the real GPT-2-small draft with native PyTorch on the existing Windows
RTX 3070, which does not require WSL and adds no cloud-compute cost. After the
draft passes its preregistered validation gate, transfer it together with the
base/SFT/DPO targets to one on-demand RunPod Community Cloud Ubuntu pod with a
24 GB RTX A5000. Perform every quantization, serving, quality-generation, and
speculative-decoding measurement on that single cloud GPU.

RunPod is the lowest-friction candidate because it offers SSH-accessible Linux
GPU pods and per-second billing. Its current GPU list shows the 24 GB RTX A5000
from $0.16/GPU-hour, subject to availability. This is cheaper than the listed
16 GB A4000 and provides more VRAM. The target is only 355M parameters and the
draft 117M, so 24 GB gives comfortable margin for GPTQ calibration, both models
during speculative decoding, concurrency testing, and vLLM KV cache.

This preserves the substantive design:

- actual base/SFT/DPO artifacts;
- real GPT-2-small distillation;
- Hugging Face versus vLLM;
- locked GPTQModel quantization;
- frozen-judge quality validation;
- vLLM speculative decoding and acceptance counters.

It changes the execution topology and serving GPU. Before looking at any
outputs, add a timestamped protocol amendment recording native-Windows RTX
3070 draft training and the exact A5000 pod image for serving. Preserve all
datasets, seeds, trial counts, thresholds, comparisons, and the GRPO exclusion.
Never combine timings across the two GPUs: the RTX 3070 produces only the draft
artifact; every comparative timing is measured on the same A5000.

A planning allowance of 6--12 cloud hours corresponds to roughly $1--$2 of
compute at the advertised starting rate. A $5 cap provides room for setup and
storage. These are prospective budgeting calculations, not measured runtime or
performance results. Actual cost must be generated from the provider's billed
pod uptime, storage, and displayed rate. Use an on-demand pod, not Spot, because
interruptions would compromise timing trials; delete both pod and volume after
results and provenance are copied out.

Source: [RunPod current GPU models and pod pricing](https://www.runpod.io/gpu-models)

### 2. Faster but more expensive: RTX 4090 pod

If wall-clock time becomes more important than minimum cost, run the entire
pipeline on a 24 GB RTX 4090 Ubuntu pod. RunPod currently lists Community Cloud
from $0.34/GPU-hour and Secure Cloud from $0.74/GPU-hour. This simplifies the
topology and accelerates draft training, but it is unnecessary for the small
models in this study and is not the preferred budget option.

Source: [RunPod RTX 4090 pricing and specifications](https://www.runpod.io/gpu-models/rtx-4090)

### 3. Acceptable fallback: dedicated GCP L4 VM

A `g2-standard-4` Linux VM supplies one 24 GB L4 and a stable, dedicated
runtime. Google currently lists that complete VM at approximately
$0.7068/hour before disk and network charges. This is more reproducible than a
notebook runtime and preserves the same software study, but account setup,
quota, image configuration, and cost controls are heavier than a GPU pod.

Source: [Google Cloud accelerator-optimized VM pricing](https://cloud.google.com/products/compute/pricing/accelerator-optimized)

### 4. Pilot/debug only: Google Colab

Colab can test installation, artifact loading, and a reduced dry run on Linux.
It is not recommended for the confirmatory matrix: Google explicitly states
that GPU types, usage limits, idle timeouts, and maximum VM lifetime vary and
are not guaranteed; typical runtimes can terminate within 12 hours. That makes
the ten-trial serving matrix and persistent raw ledgers unnecessarily fragile.

Source: [Google Colab resource-limit FAQ](https://research.google.com/colaboratory/faq.html)

### 5. Not equivalent: native Windows inference engine

Using llama.cpp, ONNX Runtime, or another Windows-native runtime could form a
separate engineering study, but it would replace vLLM scheduling, GPTQ with a
different quantization representation, and vLLM's speculative-decoding
instrumentation. It would not answer the preregistered question and must not be
reported as if it did. It is therefore not recommended for this extension.

### 6. Not recommended: another VMX attempt or dual boot

Firmware repair or a native Ubuntu dual boot might eventually expose the RTX
3070 to Linux, but both require physical access and introduce avoidable boot
and data-availability risk. They offer no methodological advantage over a
temporary Linux GPU pod.

## Recommended execution decision

Train and validate the draft on the existing Windows RTX 3070, then provision
one on-demand persistent Ubuntu RTX A5000 pod for every serving-stage operation.
Record both environments, including immutable image, driver, CUDA, Python,
vLLM, GPTQModel, PyTorch, CPU, RAM, disk, commands, and artifact hashes. Add a
platform-only preregistration amendment before training and run the substantive
pipeline unchanged. Do not combine timings across GPUs.

The alternative is honest and defensible: the local path was attempted and
failed for a documented firmware constraint, while the research estimands and
precommitted statistical tests remain unchanged on a supported Linux system.
