from src.analysis.scaling_laws import (
    EXAMPLE_RLHF_SCALING_POINTS,
    ScalingPoint,
    fit_all_families,
    fit_power_law,
    format_fit_table,
)


def test_fit_power_law_recovers_positive_alpha():
    points = [
        ScalingPoint("a", 100, 1.10, "toy"),
        ScalingPoint("b", 1_000, 0.80, "toy"),
        ScalingPoint("c", 10_000, 0.65, "toy"),
    ]

    fit = fit_power_law(points, family="toy", irreducible_loss=0.5)

    assert fit.alpha > 0
    assert fit.r_squared > 0.9
    assert fit.predict(20_000) < fit.predict(1_000)


def test_synthetic_example_points_fit_multiple_families():
    fits = fit_all_families(EXAMPLE_RLHF_SCALING_POINTS)
    families = {fit.family for fit in fits}

    assert {"reward_model", "sft", "dpo", "iterative_dpo"}.issubset(families)
    assert "| Family |" in format_fit_table(fits)
