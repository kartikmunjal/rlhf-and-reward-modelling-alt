import numpy as np

from src.safety.v2_metrics import (
    binary_diagnostic_report,
    expected_calibration_error,
    identity_fairness_report,
    threshold_sensitivity,
)


def test_expected_calibration_error_is_zero_for_perfect_probabilities():
    truth = np.array([0, 0, 1, 1])
    probability = np.array([0.0, 0.0, 1.0, 1.0])
    ece, bins = expected_calibration_error(truth, probability, n_bins=2)
    assert ece == 0.0
    assert sum(item["n"] for item in bins) == 4


def test_threshold_sensitivity_reports_each_category_and_threshold():
    truth = np.array([[0, 0, 0], [1, 1, 1]])
    probability = np.array([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]])
    result = threshold_sensitivity(truth, probability, [0.25, 0.5])
    assert set(result) == {"hate_harassment", "sexualized", "harmful_violent"}
    assert len(result["hate_harassment"]) == 2


def test_binary_diagnostic_reports_functionality_slices():
    report = binary_diagnostic_report(
        np.array([0, 1, 0, 1]),
        np.array([0.1, 0.9, 0.2, 0.8]),
        0.5,
        np.array(["a", "a", "b", "b"]),
        n_bootstrap=10,
    )
    assert report["overall"]["accuracy"]["value"] == 1.0
    assert set(report["by_slice"]) == {"a", "b"}


def test_identity_fairness_reports_rates_and_conservative_gap():
    truth = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]])
    probability = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [0.9, 0.1, 0.1]]
    )
    memberships = {
        "group_a": np.array([1, 0, 1, 0]),
        "group_b": np.array([0, 1, 0, 1]),
    }
    report = identity_fairness_report(
        truth, probability, np.array([0.5, 0.5, 0.5]), memberships, n_bootstrap=10
    )
    assert report["groups"]["group_a"]["fpr"]["value"] == 1.0
    assert report["groups"]["group_b"]["fpr"]["value"] == 0.0
    assert report["fpr_gap_worst_minus_best"]["value"] == 1.0
