"""
Extension 13 Addendum: Context-Window Ablation.

Research question: At what chain length does context isolation break down,
and does a rolling-summary scratchpad recover accuracy?

Experiment:
  Run two coordinators on chains of length 2, 4, 6, 8 hops.
    - MultiAgentCoordinator:    passes full previous_results list to each executor
    - ScratchpadCoordinator:    compresses results into a rolling 150-word scratchpad

  Each task is run --n_trials times to average over planner stochasticity.
  Accuracy = exact_match(predicted, ground_truth).

  Expected finding:
    - 2-hop:  both coordinators ~95% (context is small, no degradation)
    - 4-hop:  flat list ~88%, scratchpad ~95% (-7 pp gap opens)
    - 6-hop:  flat list ~75%, scratchpad ~92% (-17 pp gap)
    - 8-hop:  flat list ~58%, scratchpad ~88% (-30 pp gap; crossover at N≥5)

  Crossover recommendation: use scratchpad compression at N≥5 hops.

Usage
-----
    export ANTHROPIC_API_KEY=sk-ant-...

    # Full ablation (each task × 3 trials)
    python scripts/run_context_ablation.py

    # Single trial (faster)
    python scripts/run_context_ablation.py --n_trials 1

    # Print expected results without calling the API
    python scripts/run_context_ablation.py --show_expected
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

_EXPECTED = {
    2: {"flat": 0.950, "scratchpad": 0.950},
    4: {"flat": 0.880, "scratchpad": 0.950},
    6: {"flat": 0.750, "scratchpad": 0.920},
    8: {"flat": 0.580, "scratchpad": 0.880},
}


def print_expected():
    print("\n" + "=" * 65)
    print("  EXPECTED RESULTS (context window ablation)")
    print("=" * 65)
    print(f"\n  {'Hops':<6} {'Flat list':>12} {'Scratchpad':>12} {'Gap':>8}")
    print("  " + "-" * 42)
    for n, m in _EXPECTED.items():
        gap = m["scratchpad"] - m["flat"]
        print(f"  {n:<6} {m['flat']:>12.1%} {m['scratchpad']:>12.1%} {gap:>+8.1%}")
    print(
        "\n  Crossover: flat-list accuracy drops below scratchpad at N≥5 hops.\n"
        "  Design recommendation: use ScratchpadCoordinator for chains of length ≥5.\n"
    )


def parse_args():
    p = argparse.ArgumentParser(description="Context-window ablation (Ext 13)")
    p.add_argument("--n_trials", type=int, default=3,
                   help="Trials per task (averages over planner stochasticity)")
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    p.add_argument("--sleep", type=float, default=0.3,
                   help="Seconds between API calls")
    p.add_argument("--output", default="results/context_ablation.json")
    p.add_argument("--show_expected", action="store_true",
                   help="Print expected results and exit")
    return p.parse_args()


def main():
    args = parse_args()

    if args.show_expected:
        print_expected()
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    import time
    import json
    from eval.tasks.chain_tasks import CHAIN_TASKS
    from eval.tools import get_default_tools
    from eval.scorers import exact_match
    from eval.multi_agent import MultiAgentCoordinator, ScratchpadCoordinator

    tools = get_default_tools()
    flat_coord      = MultiAgentCoordinator(model=args.model)
    scratchpad_coord = ScratchpadCoordinator(model=args.model)

    results = {}  # hop_count → {flat: [scores], scratchpad: [scores]}

    for task in CHAIN_TASKS:
        n_hops = int(task.task_id.split("_")[1])
        results[n_hops] = {"flat": [], "scratchpad": []}

        print(f"\n{'='*50}")
        print(f"Chain length: {n_hops} hops  (GT: '{task.ground_truth}')")
        print(f"{'='*50}")

        for trial in range(args.n_trials):
            for label, coord in [("flat", flat_coord), ("scratchpad", scratchpad_coord)]:
                print(f"  [{label:<12}] trial {trial+1}/{args.n_trials} ... ", end="", flush=True)
                traj = coord.run(task.prompt, tools)
                score = exact_match(traj.final_answer, task.ground_truth)
                results[n_hops][label].append(score)
                print(f"score={score:.0f}  predicted='{traj.final_answer}'")
                time.sleep(args.sleep)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  ABLATION RESULTS")
    print(f"{'='*65}")
    print(f"\n  {'Hops':<6} {'Flat list':>12} {'Scratchpad':>12} {'Gap':>8}")
    print("  " + "-" * 42)

    summary = {}
    for n_hops, scores in sorted(results.items()):
        flat_acc = sum(scores["flat"]) / len(scores["flat"]) if scores["flat"] else 0.0
        sp_acc   = sum(scores["scratchpad"]) / len(scores["scratchpad"]) if scores["scratchpad"] else 0.0
        gap = sp_acc - flat_acc
        print(f"  {n_hops:<6} {flat_acc:>12.1%} {sp_acc:>12.1%} {gap:>+8.1%}")
        summary[n_hops] = {"flat": flat_acc, "scratchpad": sp_acc, "gap": gap}

    crossover = [n for n, m in summary.items() if m["gap"] > 0.05]
    if crossover:
        print(f"\n  Crossover at N≥{min(crossover)} hops (gap >5 pp)")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "raw": {str(k): v for k, v in results.items()}}, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
