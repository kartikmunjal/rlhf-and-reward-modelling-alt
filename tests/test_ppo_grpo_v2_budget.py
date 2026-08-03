import pytest

from src.training.ppo_grpo_v2 import _assert_budget


CONFIG = {
    "compute_budget": {"updates_per_rollout_group": 2, "generations_per_group": 4},
    "runtime_assertions": {
        "optimizer_steps_must_equal": 200,
        "rollout_groups_must_equal": 100,
        "generated_completions_must_equal": 400,
    },
}


def test_ppo_budget_assertion_counts_two_updates_per_group():
    trajectory = [{"objective/scores": 0.1} for _ in range(100)]
    assert _assert_budget("ppo", trajectory, CONFIG)["optimizer_steps"] == 200


def test_grpo_budget_assertion_requires_100_reward_groups_and_step_200():
    trajectory = [{"reward": 0.1, "step": 2 * index + 1} for index in range(100)]
    trajectory.append({"step": 200})
    assert _assert_budget("grpo", trajectory, CONFIG)["rollout_groups"] == 100


def test_budget_assertion_rejects_v1_double_reuse_shape():
    trajectory = [{"reward": 0.1, "step": 4 * index + 1} for index in range(50)]
    trajectory.append({"step": 200})
    with pytest.raises(RuntimeError):
        _assert_budget("grpo", trajectory, CONFIG)
