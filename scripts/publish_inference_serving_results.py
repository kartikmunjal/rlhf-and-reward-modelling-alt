#!/usr/bin/env python3
"""Regenerate inference-serving result summaries from analysis artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "inference_serving_v1"
START = "<!-- INFERENCE-SERVING-RESULTS:START -->"
END = "<!-- INFERENCE-SERVING-RESULTS:END -->"


def _ci(row: dict, digits: int = 1) -> str:
    return f"{row['estimate']:.{digits}f} (95% CI {row['ci95'][0]:.{digits}f} to {row['ci95'][1]:.{digits}f}; N_trials={row['n_trials']})"


def _replace(text: str, start: str, end: str, replacement: str) -> str:
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        return before + replacement + after
    return text.rstrip() + "\n\n" + replacement + "\n"


def build(metrics: dict, selection: dict) -> str:
    stage1 = metrics["stage1"]["metrics"]["output_tokens_per_second"]
    stage2 = metrics["stage2_performance"]["metrics"]["output_tokens_per_second"]
    stage3 = metrics["stage3_performance"]["metrics"]["output_tokens_per_second"]
    quality = metrics["stage2_quality"]
    accept = metrics["stage3_acceptance"]
    contrasts = metrics["stage3_acceptance_differences"]
    gpu = ", ".join(metrics["execution_hardware"]["serving_gpu_names"])
    pilot = selection["candidates"]
    k = selection["selected_speculative_tokens"]
    lines = [
        START,
        "## Alignment-Aware Inference Serving Extension",
        "",
        "This completed, preregistered extension serves the repository's real GPT-2-medium DPO artifact, quantizes the same base/SFT/DPO family with 4-bit GPTQ, and uses a trained GPT-2-small draft for speculative decoding. All comparative serving measurements ran on the same " + gpu + "; only draft training ran on the RTX 3070.",
        "",
        f"**Stage 1 — PASS.** At concurrency 32, vLLM delivered {stage1['left']['estimate']:.1f} output tokens/s versus {stage1['right']['estimate']:.1f} for Hugging Face `generate()`: a paired gain of {_ci(stage1)}. This is a {stage1['left']['estimate'] / stage1['right']['estimate']:.2f}x throughput ratio. vLLM's larger device-memory reading reflects its preallocated KV cache, so it is not a model-weight footprint comparison.",
        "",
        f"**Stage 2 performance — PASS.** GPTQ delivered {stage2['left']['estimate']:.1f} versus {stage2['right']['estimate']:.1f} output tokens/s for FP16, a paired gain of {_ci(stage2)}. **Stage 2 quality equivalence — FAIL.** Across {quality['coverage']['valid']}/{quality['coverage']['total']} held-out articles, GPTQ minus FP16 was {quality['axes']['relevance']['estimate']:.3f} (95% CI {quality['axes']['relevance']['ci95'][0]:.3f} to {quality['axes']['relevance']['ci95'][1]:.3f}) for relevance and {quality['axes']['consistency']['estimate']:.3f} (95% CI {quality['axes']['consistency']['ci95'][0]:.3f} to {quality['axes']['consistency']['ci95'][1]:.3f}) for consistency. Both lower bounds had to remain at or above the frozen −0.15-point margin; neither did.",
        "",
        f"**Stage 3 — FAIL on throughput.** The pilot selected k={k} from locked k=2/4/6 candidates (median output tokens/s: " + ", ".join(f"k={name}: {row['median_output_tokens_per_second']:.1f}" for name, row in sorted(pilot.items(), key=lambda item: int(item[0]))) + f"). On held-out DPO trials, speculative decoding delivered {stage3['left']['estimate']:.1f} versus {stage3['right']['estimate']:.1f} output tokens/s, a paired difference of {_ci(stage3)}; its interval includes zero.",
        "",
        f"Draft-token acceptance was base {accept['base']['estimate']:.2%} (95% CI {accept['base']['ci95'][0]:.2%}–{accept['base']['ci95'][1]:.2%}), SFT {accept['sft']['estimate']:.2%} ({accept['sft']['ci95'][0]:.2%}–{accept['sft']['ci95'][1]:.2%}), and DPO {accept['dpo']['estimate']:.2%} ({accept['dpo']['ci95'][0]:.2%}–{accept['dpo']['ci95'][1]:.2%}), each over N_trials=10. Base minus SFT was {contrasts['base_minus_sft']['estimate']:.2%} (95% CI {contrasts['base_minus_sft']['ci95'][0]:.2%} to {contrasts['base_minus_sft']['ci95'][1]:.2%}); SFT minus DPO was {contrasts['sft_minus_dpo']['estimate']:.2%} ({contrasts['sft_minus_dpo']['ci95'][0]:.2%} to {contrasts['sft_minus_dpo']['ci95'][1]:.2%}; Holm-adjusted p={contrasts['sft_minus_dpo']['holm_adjusted_p']:.4f}). The preregistered expectation that acceptance would decrease with post-training is therefore rejected for these artifacts.",
        "",
        "Interpretation is deliberately narrow: N_training_seeds=1; the 355M target is a systems-validation workload; vLLM used its V1 runner for draft-model speculation; and these A5000 results do not establish datacenter-scale or multi-GPU behavior. GRPO remains excluded because the available GRPO artifact uses a different Qwen base. A failed base-server startup caused by `ninja` not being on the subprocess PATH was preserved as an audit log and occurred before measurement.",
        "",
        "See [`inference_serving/`](inference_serving/), [`results/inference_serving_v1/report.md`](results/inference_serving_v1/report.md), and [`results/inference_serving_v1/metrics.json`](results/inference_serving_v1/metrics.json).",
        END,
    ]
    return "\n".join(lines)


def main() -> None:
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    selection = json.loads((RESULTS / "speculative_selection.json").read_text(encoding="utf-8"))
    section = build(metrics, selection)

    module_readme = ROOT / "inference_serving" / "README.md"
    module_text = module_readme.read_text(encoding="utf-8")
    module_readme.write_text(_replace(module_text, START, END, section), encoding="utf-8")

    main_readme = ROOT / "README.md"
    main_text = main_readme.read_text(encoding="utf-8")
    if START in main_text and END in main_text:
        updated = _replace(main_text, START, END, section)
    else:
        old_start = "## Alignment-Aware Inference Serving Extension"
        enclosing_end = "<!-- SUMMARIZATION-FINETUNE-RESULTS:END -->"
        before, rest = main_text.split(old_start, 1)
        _, after = rest.split(enclosing_end, 1)
        updated = before + section + "\n" + enclosing_end + after
    main_readme.write_text(updated, encoding="utf-8")

    print(f"Updated {module_readme.relative_to(ROOT)} and {main_readme.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
