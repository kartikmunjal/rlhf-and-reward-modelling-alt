"""
Reward Hacking Detection Analysis (Extension 2 Addendum).

Research Question: Can a two-signal heuristic (length z-score + KL divergence
trend) detect verbose-bias reward hacking before it reaches pathological levels?

Experiment
----------
1. Simulate two training traces: one clean, one with hacking starting at step 10.
2. Run RewardHackingDetector on both traces.
3. Show detection latency: how many steps after hacking starts does the detector fire?
4. Simulate ensemble RM (Extension 2) reducing the hacking rate.
5. Show that the detector fires later (or not at all) when ensemble is used.

Expected finding
----------------
  - Clean trace:   detector stays quiet throughout (0 false positives)
  - Hacking trace: warning fires ~3-4 steps after hack_start, hard stop ~5-6 steps after
  - With ensemble: hacking growth rate is halved → detection delay increases by ~3 steps
  - Design rule: couple ensemble penalty (λ=0.3) with detector threshold (z=2.5, KL=0.15)
                 for defence in depth

Usage
-----
    python scripts/run_reward_hacking_analysis.py
    python scripts/run_reward_hacking_analysis.py --show_expected
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

_EXPECTED = {
    "clean_warnings":          0,
    "clean_hard_stops":        0,
    "hacking_first_warning":   13,   # step (hack starts at 10, ~3-step latency)
    "hacking_first_hard_stop": 15,   # step
    "ensemble_first_warning":  16,   # 3 extra steps due to slowed hacking rate
    "detection_delay_gain":     3,   # extra steps of safe training with ensemble
}


def print_expected():
    print("\n" + "=" * 62)
    print("  EXPECTED RESULTS (reward hacking detection analysis)")
    print("=" * 62)
    print(f"\n  {'Metric':<40} {'Value':>8}")
    print("  " + "-" * 50)
    for k, v in _EXPECTED.items():
        print(f"  {k:<40} {v:>8}")
    print(
        "\n  Design rule: combine ensemble penalty (λ=0.3) with detector\n"
        "  thresholds z=2.5 / KL=0.15 for defence-in-depth.\n"
        "  Detector alone catches hacking; ensemble buys ~3 extra safe steps.\n"
    )


def parse_args():
    p = argparse.ArgumentParser(description="Reward hacking detection (Ext 2 addendum)")
    p.add_argument("--show_expected", action="store_true",
                   help="Print expected results and exit")
    p.add_argument("--hack_start", type=int, default=10,
                   help="Training step where hacking begins (default 10)")
    p.add_argument("--n_steps", type=int, default=20,
                   help="Total training steps to simulate (default 20)")
    p.add_argument("--output", default="results/reward_hacking_analysis.json")
    return p.parse_args()


def run_detector(trace, label: str, **detector_kwargs):
    from src.analysis.reward_hacking_detector import RewardHackingDetector
    detector = RewardHackingDetector(**detector_kwargs)
    statuses = []
    first_warning   = None
    first_hard_stop = None

    print(f"\n  {'Step':>5}  {'Length z':>9}  {'KL div':>8}  {'KL run':>7}  {'Status'}")
    print("  " + "-" * 55)

    for step, lengths, scores in trace:
        status = detector.update(step, lengths, scores)
        statuses.append(status)

        flag = ""
        if status.hard_stop:
            flag = "HARD STOP"
            if first_hard_stop is None:
                first_hard_stop = step
        elif status.warning:
            flag = "WARNING"
            if first_warning is None:
                first_warning = step

        print(
            f"  {step:>5}  {status.length_z:>9.2f}  {status.kl_divergence:>8.3f}  "
            f"{status.kl_trend_steps:>7}  {flag}"
        )

    print(f"\n  [{label}] First warning: step {first_warning}  |  "
          f"First hard stop: step {first_hard_stop}")
    return statuses, first_warning, first_hard_stop


def main():
    args = parse_args()

    if args.show_expected:
        print_expected()
        return

    from src.analysis.reward_hacking_detector import simulate_training

    print(f"\nReward Hacking Detection Analysis")
    print(f"Hack starts at step {args.hack_start} / {args.n_steps} total steps\n")

    clean_trace, hacking_trace = simulate_training(
        n_steps=args.n_steps,
        hack_start=args.hack_start,
    )

    # ── 1. Clean trace ─────────────────────────────────────────────────────────
    print("=" * 62)
    print("  CLEAN TRACE (no reward hacking)")
    print("=" * 62)
    _, clean_warn, clean_stop = run_detector(clean_trace, "clean")

    # ── 2. Hacking trace ───────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print(f"  HACKING TRACE (verbose-bias starts step {args.hack_start})")
    print("=" * 62)
    _, hack_warn, hack_stop = run_detector(hacking_trace, "hacking")

    # ── 3. Ensemble effect simulation ─────────────────────────────────────────
    # With ensemble penalty λ=0.3, length growth rate is halved
    # Simulate by multiplying hack step multiplier by 0.5
    import random
    rng = random.Random(42)
    ensemble_trace = []
    base_len = 150
    base_score = 0.5
    for step in range(args.n_steps):
        if step < args.hack_start:
            lengths = [int(rng.gauss(base_len + step * 0.3, 20)) for _ in range(50)]
            scores  = [rng.gauss(base_score + step * 0.01, 0.5) for _ in range(50)]
        else:
            hack_step = step - args.hack_start
            # Halved growth rate vs unmitigated hacking
            lengths = [int(rng.gauss(base_len + hack_step * 5, 25)) for _ in range(50)]
            scores  = [rng.gauss(base_score + hack_step * 0.04, 0.4) for _ in range(50)]
        ensemble_trace.append((step, lengths, scores))

    print()
    print("=" * 62)
    print("  ENSEMBLE TRACE (λ=0.3 penalty, half growth rate)")
    print("=" * 62)
    _, ens_warn, ens_stop = run_detector(ensemble_trace, "ensemble")

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  SUMMARY")
    print("=" * 62)
    print(f"\n  {'Condition':<28} {'First warning':>14} {'First hard stop':>16}")
    print("  " + "-" * 62)
    print(f"  {'Clean (no hacking)':<28} {'—':>14} {'—':>16}")
    print(f"  {'Hacking (unmitigated)':<28} {str(hack_warn) if hack_warn else '—':>14} "
          f"{str(hack_stop) if hack_stop else '—':>16}")
    print(f"  {'Hacking + ensemble (λ=0.3)':<28} {str(ens_warn) if ens_warn else '—':>14} "
          f"{str(ens_stop) if ens_stop else '—':>16}")

    if hack_warn is not None and ens_warn is not None:
        gain = ens_warn - hack_warn
        print(f"\n  Detection delay gain from ensemble: +{gain} steps of safe training")

    print(
        "\n  Design rule: ensemble RM (λ=0.3) + detector (z=2.5, KL=0.15) gives\n"
        "  defence-in-depth. Ensemble slows hacking; detector catches what slips through.\n"
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    import json
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "clean":    {"first_warning": clean_warn, "first_hard_stop": clean_stop},
            "hacking":  {"first_warning": hack_warn,  "first_hard_stop": hack_stop},
            "ensemble": {"first_warning": ens_warn,   "first_hard_stop": ens_stop},
            "detection_delay_gain": (ens_warn - hack_warn)
                if (hack_warn is not None and ens_warn is not None) else None,
        }, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
