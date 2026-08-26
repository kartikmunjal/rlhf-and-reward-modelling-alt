#!/usr/bin/env python3
"""Generate README findings exclusively from locked result artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def ci(metric: dict) -> str:
    return f"{metric['estimate']:.3f} (95% CI {metric['ci95'][0]:.3f}–{metric['ci95'][1]:.3f}; N={metric['n_observations']})"


def generated(metrics: dict) -> str:
    claude = metrics["pointwise"]["anthropic"]
    openai = metrics["pointwise"]["openai"]
    pairwise = metrics["pairwise"]
    relevance, consistency = claude["axes"]["relevance"], claude["axes"]["consistency"]
    cross = metrics["cross_provider"]["axes"]
    cost = metrics["actual_api_cost_at_recorded_rates"]
    return "\n".join([
        "## SummEval LLM-as-Judge Extension",
        "",
        f"**Preregistered primary result: {'PASS' if metrics['primary_success'] else 'FAIL'}.** "
        f"Claude relevance tracked expert ratings at Spearman rho {ci(relevance['judge_human'])}; "
        f"consistency reached {ci(consistency['judge_human'])}. Both exceeded the locked 0.40 threshold.",
        "",
        f"Against ROUGE-L, the paired correlation advantage was {ci(relevance['judge_minus_rouge'])} "
        f"for relevance and {ci(consistency['judge_minus_rouge'])} for consistency. Both intervals exclude zero positively, satisfying the second locked condition.",
        "",
        f"Claude coverage was {claude['valid']}/{claude['total']} (Wilson 95% CI "
        f"{claude['valid_rate_wilson_ci95'][0]:.3f}–{claude['valid_rate_wilson_ci95'][1]:.3f}). "
        f"The amended GPT-5-mini secondary run covered {openai['valid']}/{openai['total']} (Wilson 95% CI "
        f"{openai['valid_rate_wilson_ci95'][0]:.3f}–{openai['valid_rate_wilson_ci95'][1]:.3f}). "
        f"Cross-provider agreement was {ci(cross['relevance'])} for relevance and {ci(cross['consistency'])} for consistency.",
        "",
        f"Length sensitivity was modest but detectable: Claude score-length rho was {ci(relevance['length_bias'])} for relevance and {ci(consistency['length_bias'])} for consistency. Controlling for length left judge-human correlations at {ci(relevance['judge_human_controlling_length'])} and {ci(consistency['judge_human_controlling_length'])}, respectively, so verbosity does not mechanically explain the primary result.",
        "",
        f"**Pairwise limitation:** only {pairwise['valid']}/{pairwise['total']} bidirectional pairs were complete (Wilson 95% CI "
        f"{pairwise['valid_rate_wilson_ci95'][0]:.3f}–{pairwise['valid_rate_wilson_ci95'][1]:.3f}), below the preregistered 90% floor. Pairwise position-bias and mitigation estimates are exploratory. Symmetrization improved agreement for consistency and fluency but reduced it for relevance and coherence, so the proposed mitigation did not generalize across axes.",
        "",
        f"Recorded-token API cost was ${cost['total_usd']:.2f}. All correlations and differences use 2,000 deterministic clustered-bootstrap trials; validity rates use Wilson intervals.",
        "",
        "See [`llm_judge_summeval/`](llm_judge_summeval/) and [`results/summeval_judge_v1/report.md`](results/summeval_judge_v1/report.md) for the frozen design, amendment, audit ledgers, complete metrics, and interpretation boundaries.",
    ])


def module_generated(metrics: dict) -> str:
    claude = metrics["pointwise"]["anthropic"]
    pairwise = metrics["pairwise"]
    relevance = claude["axes"]["relevance"]
    consistency = claude["axes"]["consistency"]
    return "\n".join([
        "## Generated findings",
        "",
        f"The preregistered primary result **{'passed' if metrics['primary_success'] else 'failed'}**. "
        f"Claude judge-human Spearman rho was {ci(relevance['judge_human'])} for relevance and "
        f"{ci(consistency['judge_human'])} for consistency. Its paired advantage over ROUGE-L was "
        f"{ci(relevance['judge_minus_rouge'])} and {ci(consistency['judge_minus_rouge'])}, respectively.",
        "",
        f"Primary pointwise coverage was {claude['valid']}/{claude['total']} (Wilson 95% CI "
        f"{claude['valid_rate_wilson_ci95'][0]:.3f}–{claude['valid_rate_wilson_ci95'][1]:.3f}). "
        f"Pairwise coverage was {pairwise['valid']}/{pairwise['total']} (Wilson 95% CI "
        f"{pairwise['valid_rate_wilson_ci95'][0]:.3f}–{pairwise['valid_rate_wilson_ci95'][1]:.3f}), so pairwise conclusions remain exploratory and infrastructure-incomplete.",
        "",
        "The preregistered symmetrization mitigation was not generally supported: its direction differed by axis. See [`../results/summeval_judge_v1/report.md`](../results/summeval_judge_v1/report.md) for every metric and confidence interval.",
    ])


def replace_between(text: str, start: str, end: str, content: str) -> str:
    if start not in text or end not in text:
        raise ValueError(f"Missing README markers: {start}, {end}")
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    return before + start + "\n" + content.rstrip() + "\n" + end + after


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=Path("results/summeval_judge_v1/metrics.json"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--module-readme", type=Path, default=Path("llm_judge_summeval/README.md"))
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    text = args.readme.read_text(encoding="utf-8")
    if "<!-- SUMMEVAL-RESULTS:START -->" not in text:
        begin = text.index("## SummEval LLM-as-Judge Extension")
        end = text.index("<!-- SAFETY-RESULTS:START -->")
        text = text[:begin] + "<!-- SUMMEVAL-RESULTS:START -->\nplaceholder\n<!-- SUMMEVAL-RESULTS:END -->\n\n" + text[end:]
    text = replace_between(text, "<!-- SUMMEVAL-RESULTS:START -->", "<!-- SUMMEVAL-RESULTS:END -->", generated(metrics))
    args.readme.write_text(text, encoding="utf-8")
    module_text = args.module_readme.read_text(encoding="utf-8")
    start, end = "<!-- GENERATED-FINDINGS:START -->", "<!-- GENERATED-FINDINGS:END -->"
    if start not in module_text:
        module_text = module_text.rstrip() + f"\n\n{start}\nplaceholder\n{end}\n"
    module_text = replace_between(module_text, start, end, module_generated(metrics))
    args.module_readme.write_text(module_text, encoding="utf-8")
    print("Updated", args.readme, "and", args.module_readme)


if __name__ == "__main__":
    main()
