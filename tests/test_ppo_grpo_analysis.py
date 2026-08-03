import numpy as np

from scripts.analyze_ppo_grpo_v1 import bootstrap_seed_metric, wilson_interval


def test_seed_bootstrap_tracks_trial_count_and_constant_interval():
    result = bootstrap_seed_metric([2.0, 2.0, 2.0], np.random.default_rng(7), 100)
    assert result == {"estimate": 2.0, "ci95": [2.0, 2.0], "n_trials": 3}


def test_wilson_zero_success_interval_is_non_degenerate():
    low, high = wilson_interval(0, 400)
    assert low == 0.0
    assert 0.009 < high < 0.010
