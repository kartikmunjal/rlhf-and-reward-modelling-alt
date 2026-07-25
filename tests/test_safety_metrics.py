import numpy as np

from src.safety.metrics import classification_report, fairness_report, select_thresholds


def test_threshold_selection_uses_grid_and_resolves_ties_upward():
    truth = np.array([[0], [0], [1], [1]])
    probability = np.array([[0.1], [0.2], [0.8], [0.9]])
    threshold = select_thresholds(truth, probability, np.array([0.3, 0.5, 0.7]))
    assert threshold.tolist() == [0.7]


def test_classification_report_contains_ci_and_trial_count():
    truth = np.array([[0, 0, 0], [1, 0, 1], [1, 1, 0], [0, 1, 0]])
    probability = np.array(
        [[0.1, 0.1, 0.1], [0.9, 0.2, 0.8], [0.8, 0.9, 0.2], [0.2, 0.8, 0.1]]
    )
    report = classification_report(
        truth, probability, np.array([0.5, 0.5, 0.5]), n_bootstrap=20, n_trials=1
    )
    assert report["n_trials"] == 1
    assert report["per_category"]["hate_harassment"]["f1"]["value"] == 1.0
    assert len(report["per_category"]["sexualized"]["precision"]["ci95"]) == 2


def test_fairness_gap_is_adjacent_minus_base():
    base_truth = np.zeros((4, 3))
    base_probability = np.array(
        [[0.9, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
    )
    benign_probability = np.array(
        [[0.9, 0.1, 0.1], [0.8, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
    )
    report = fairness_report(
        base_truth,
        base_probability,
        benign_probability,
        np.array(["a", "a", "b", "b"]),
        np.array([0.5, 0.5, 0.5]),
        n_bootstrap=20,
    )
    assert report["overall_test_fpr"]["value"] == 0.25
    assert report["adjacent_benign_fpr"]["value"] == 0.5
    assert report["adjacent_benign_fpr_gap"]["value"] == 0.25
