"""
Reward Hacking Detector (Extension 2 Addendum).

Research Question: Can a lightweight heuristic — tracking response length
distribution shift and KL divergence of reward scores — detect reward hacking
early enough to pause training before the model degrades?

Design
------
Two complementary signals:

1. Length z-score
   At each checkpoint, compute the mean response length on a held-out
   probe set. If the z-score (relative to the first K checkpoints) exceeds
   a threshold, flag as length-hacking.

2. KL divergence trend
   Track the distribution of reward scores across checkpoints. Reward
   hacking manifests as the reward distribution shifting right (model
   learns to elicit high scores without genuine quality improvement).
   If the KL divergence from the initial distribution rises monotonically
   for W consecutive checkpoints, flag as score-gaming.

Both signals are combined: if EITHER fires, the detector raises a warning.
If BOTH fire simultaneously, it escalates to a hard stop recommendation.

Connection to Extensions 1–3
-----------------------------
- Extension 1 (Bradley-Terry RM): baseline reward model; no protection
- Extension 2 (Ensemble RM + penalty): ensemble disagreement as proxy for
  hacking. The detector here quantifies *when* the penalty kicks in.
- Extension 3 (CAI constitutional filters): downstream guardrails that catch
  what slips past the RM. The detector is an upstream early-warning system.

Usage
-----
    from src.analysis.reward_hacking_detector import RewardHackingDetector

    detector = RewardHackingDetector()
    for step, (lengths, scores) in enumerate(training_checkpoints):
        status = detector.update(step, lengths, scores)
        if status.hard_stop:
            raise RuntimeError(f"Reward hacking detected at step {step}")
        if status.warning:
            print(f"WARNING at step {step}: {status.message}")
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class DetectorStatus:
    step: int
    warning: bool
    hard_stop: bool
    length_z: float
    kl_divergence: float
    kl_trend_steps: int  # how many consecutive steps KL has increased
    message: str


@dataclass
class _Checkpoint:
    step: int
    mean_length: float
    reward_hist: List[float]  # histogram bin counts, normalised


# ── Core detector ──────────────────────────────────────────────────────────────

class RewardHackingDetector:
    """Lightweight heuristic detector for reward hacking during RLHF training.

    Parameters
    ----------
    warmup_steps : int
        Number of initial checkpoints used to establish the baseline
        length distribution (default 5).
    length_z_threshold : float
        Z-score above which response length growth is flagged (default 2.5).
        At z=2.5, ~1.2% of checkpoints would trigger randomly under N(0,1).
    kl_threshold : float
        KL divergence from initial reward distribution that triggers a flag.
    kl_trend_window : int
        Number of consecutive steps of monotonically rising KL that must be
        observed before the KL signal fires (default 3).
    reward_bins : int
        Number of histogram bins for the reward score distribution (default 20).
    reward_range : tuple
        (min, max) of reward scores expected from the RM (default (-3, 3)).
    """

    def __init__(
        self,
        warmup_steps: int = 5,
        length_z_threshold: float = 2.5,
        kl_threshold: float = 0.15,
        kl_trend_window: int = 3,
        reward_bins: int = 20,
        reward_range: tuple = (-3.0, 3.0),
    ):
        self.warmup_steps = warmup_steps
        self.length_z_threshold = length_z_threshold
        self.kl_threshold = kl_threshold
        self.kl_trend_window = kl_trend_window
        self.reward_bins = reward_bins
        self.reward_range = reward_range

        self._history: List[_Checkpoint] = []
        self._initial_hist: Optional[List[float]] = None
        self._kl_values: List[float] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        step: int,
        response_lengths: Sequence[int],
        reward_scores: Sequence[float],
    ) -> DetectorStatus:
        """Record a new checkpoint and return the current detector status.

        Parameters
        ----------
        step : int
            Training step or checkpoint index.
        response_lengths : sequence of int
            Token lengths of model responses sampled at this checkpoint.
        reward_scores : sequence of float
            Reward model scores for those responses.

        Returns
        -------
        DetectorStatus
        """
        mean_len = statistics.mean(response_lengths) if response_lengths else 0.0
        hist = self._make_histogram(reward_scores)

        ckpt = _Checkpoint(step=step, mean_length=mean_len, reward_hist=hist)
        self._history.append(ckpt)

        # Store initial distribution after warmup
        if len(self._history) == self.warmup_steps:
            # Average the first warmup_steps histograms as the reference
            self._initial_hist = self._avg_hist(
                [c.reward_hist for c in self._history[:self.warmup_steps]]
            )

        # ── Length signal ──────────────────────────────────────────────────────
        length_z = self._length_z_score()

        # ── KL signal ─────────────────────────────────────────────────────────
        kl_div = 0.0
        kl_trend_steps = 0
        if self._initial_hist is not None and len(self._history) > self.warmup_steps:
            kl_div = self._kl_divergence(self._initial_hist, hist)
            self._kl_values.append(kl_div)
            kl_trend_steps = self._monotonic_run(self._kl_values)

        # ── Decision ──────────────────────────────────────────────────────────
        length_flag = len(self._history) > self.warmup_steps and length_z > self.length_z_threshold
        kl_flag     = (kl_div > self.kl_threshold
                       and kl_trend_steps >= self.kl_trend_window)

        warning   = length_flag or kl_flag
        hard_stop = length_flag and kl_flag

        parts = []
        if length_flag:
            parts.append(f"length z={length_z:.2f} (threshold {self.length_z_threshold})")
        if kl_flag:
            parts.append(
                f"KL={kl_div:.3f} over {kl_trend_steps} consecutive steps "
                f"(threshold {self.kl_threshold}, window {self.kl_trend_window})"
            )
        message = "; ".join(parts) if parts else "OK"

        return DetectorStatus(
            step=step,
            warning=warning,
            hard_stop=hard_stop,
            length_z=length_z,
            kl_divergence=kl_div,
            kl_trend_steps=kl_trend_steps,
            message=message,
        )

    def summary(self) -> str:
        """Return a human-readable summary of detector history."""
        if not self._history:
            return "No checkpoints recorded."
        lines = [
            f"Checkpoints: {len(self._history)}",
            f"Warmup:      {self.warmup_steps}",
            f"Final length z-score: {self._length_z_score():.2f}",
        ]
        if self._kl_values:
            lines.append(f"Final KL divergence:  {self._kl_values[-1]:.3f}")
            lines.append(f"KL trend run:         {self._monotonic_run(self._kl_values)}")
        return "\n".join(lines)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _make_histogram(self, scores: Sequence[float]) -> List[float]:
        """Normalised histogram of reward scores."""
        lo, hi = self.reward_range
        bins = [0.0] * self.reward_bins
        width = (hi - lo) / self.reward_bins
        for s in scores:
            idx = int((s - lo) / width)
            idx = max(0, min(self.reward_bins - 1, idx))
            bins[idx] += 1.0
        total = sum(bins)
        if total > 0:
            bins = [b / total for b in bins]
        return bins

    def _avg_hist(self, hists: List[List[float]]) -> List[float]:
        n = len(hists)
        avg = [0.0] * self.reward_bins
        for h in hists:
            for i, v in enumerate(h):
                avg[i] += v / n
        return avg

    @staticmethod
    def _kl_divergence(p: List[float], q: List[float], eps: float = 1e-10) -> float:
        """KL(p || q) — divergence of q from reference distribution p."""
        kl = 0.0
        for pi, qi in zip(p, q):
            pi = max(pi, eps)
            qi = max(qi, eps)
            kl += pi * math.log(pi / qi)
        return kl

    def _length_z_score(self) -> float:
        """Z-score of the most recent mean length relative to warmup baseline."""
        if len(self._history) <= self.warmup_steps:
            return 0.0
        baseline_lengths = [c.mean_length for c in self._history[:self.warmup_steps]]
        mu  = statistics.mean(baseline_lengths)
        std = statistics.stdev(baseline_lengths) if len(baseline_lengths) > 1 else 1.0
        if std < 1e-6:
            std = 1.0
        current = self._history[-1].mean_length
        return (current - mu) / std

    @staticmethod
    def _monotonic_run(values: List[float]) -> int:
        """Length of the trailing monotonically non-decreasing run."""
        if len(values) < 2:
            return 0
        run = 1
        for i in range(len(values) - 1, 0, -1):
            if values[i] >= values[i - 1]:
                run += 1
            else:
                break
        return run if run >= 2 else 0


# ── Synthetic demonstration helper ────────────────────────────────────────────

def simulate_training(
    n_steps: int = 20,
    hack_start: int = 10,
    seed: int = 42,
) -> tuple:
    """Generate synthetic training traces for demonstration.

    Returns
    -------
    (clean_trace, hacking_trace) where each trace is a list of
    (step, lengths, scores) tuples.
    """
    import random
    rng = random.Random(seed)

    def _norm(mu, sigma, n):
        return [rng.gauss(mu, sigma) for _ in range(n)]

    clean_trace   = []
    hacking_trace = []

    base_len   = 150
    base_score = 0.5

    for step in range(n_steps):
        # Clean: stable length + slowly improving scores
        lengths_c = [int(x) for x in _norm(base_len + step * 0.3, 20, 50)]
        scores_c  = _norm(base_score + step * 0.01, 0.5, 50)
        clean_trace.append((step, lengths_c, scores_c))

        # Hacking: after hack_start, length explodes and reward scores inflate
        if step < hack_start:
            lengths_h = [int(x) for x in _norm(base_len + step * 0.3, 20, 50)]
            scores_h  = _norm(base_score + step * 0.01, 0.5, 50)
        else:
            # Verbose-bias: responses grow ~10 tokens/step
            hack_step = step - hack_start
            lengths_h = [int(x) for x in _norm(base_len + hack_step * 10, 25, 50)]
            # Score inflation: distribution shifts right
            scores_h  = _norm(base_score + hack_step * 0.08, 0.4, 50)
        hacking_trace.append((step, lengths_h, scores_h))

    return clean_trace, hacking_trace
