import numpy as np

from verifier_reward_hacking.statistics import hierarchical_bootstrap_mean, hierarchical_paired_bootstrap, holm_adjust, paired_task_bootstrap, spearman_task_bootstrap, task_bootstrap_mean


def test_bootstraps_are_deterministic_and_report_sample_sizes():
    first = task_bootstrap_mean([0, 1, 1], 100, 4)
    second = task_bootstrap_mean([0, 1, 1], 100, 4)
    assert first == second
    assert first["n_trials"] == 1 and first["n_tasks_per_trial"] == 3
    values = np.array([[0, 1, 1], [1, 1, 0], [1, 1, 1]])
    absolute = hierarchical_bootstrap_mean(values, 100, 4)
    contrast = hierarchical_paired_bootstrap(values, np.zeros_like(values), 100, 4)
    assert absolute["n_trials"] == contrast["n_trials"] == 3


def test_paired_and_spearman_directionality():
    paired = paired_task_bootstrap([1, 1, 1, 1], [0, 0, 0, 0], 100, 7)
    assert paired["ci95"][0] > 0
    rank = spearman_task_bootstrap([0, 1, 2, 3], [3, 2, 1, 0], 100, 7)
    assert rank["estimate"] == -1.0


def test_holm_is_monotone_in_sorted_p_values():
    adjusted = holm_adjust([0.03, 0.01, 0.2])
    assert adjusted[1] <= adjusted[0] <= adjusted[2]
