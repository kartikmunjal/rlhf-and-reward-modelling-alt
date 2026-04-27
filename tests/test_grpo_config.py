from src.training.grpo import GRPOTrainingConfig


def test_grpo_defaults_match_repo_recommendation() -> None:
    cfg = GRPOTrainingConfig()

    assert cfg.num_generations == 4
    assert cfg.beta == 0.1
    assert cfg.epsilon == 0.2
