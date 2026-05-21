"""
Smoke tests for training configuration dataclasses.

These tests verify that default hyperparameters are within expected ranges and
that configs can be instantiated without side effects. They run in milliseconds
and require no GPU or model downloads.
"""
from src.training.reward_ensemble import EnsembleTrainingConfig
from src.training.sft_lora import LoRASFTConfig
from src.training.dpo_lora import LoRADPOConfig


class TestEnsembleTrainingConfig:
    def test_default_instantiation(self):
        cfg = EnsembleTrainingConfig()
        assert cfg.k == 3
        assert cfg.base_seed == 0

    def test_seeds_are_distinct(self):
        cfg = EnsembleTrainingConfig(k=5, base_seed=10)
        seeds = [cfg.base_seed + i for i in range(cfg.k)]
        assert len(set(seeds)) == cfg.k, "Each ensemble member must have a unique seed"

    def test_learning_rate_is_reasonable(self):
        cfg = EnsembleTrainingConfig()
        assert 1e-6 < cfg.learning_rate < 1e-3, (
            f"Learning rate {cfg.learning_rate} is outside typical range for RM training"
        )

    def test_gradient_accumulation_positive(self):
        cfg = EnsembleTrainingConfig()
        assert cfg.gradient_accumulation_steps >= 1

    def test_warmup_ratio_in_unit_interval(self):
        cfg = EnsembleTrainingConfig()
        assert 0.0 < cfg.warmup_ratio < 1.0

    def test_custom_k_respected(self):
        cfg = EnsembleTrainingConfig(k=7)
        assert cfg.k == 7


class TestLoRASFTConfig:
    def test_default_instantiation(self):
        cfg = LoRASFTConfig()
        assert "gpt2" in cfg.model_name

    def test_lora_rank_positive(self):
        cfg = LoRASFTConfig()
        assert cfg.lora_r > 0

    def test_lora_alpha_positive(self):
        cfg = LoRASFTConfig()
        assert cfg.lora_alpha > 0

    def test_target_modules_non_empty(self):
        cfg = LoRASFTConfig()
        assert len(cfg.target_modules) > 0

    def test_lora_rank_less_than_hidden_dim(self):
        cfg = LoRASFTConfig()
        # r=16 should be << 768 (GPT-2 hidden dim) — the whole point of LoRA
        assert cfg.lora_r < 256, f"lora_r={cfg.lora_r} seems too large for low-rank intent"


class TestLoRADPOConfig:
    def test_default_instantiation(self):
        cfg = LoRADPOConfig()
        assert cfg.beta > 0

    def test_beta_in_expected_range(self):
        cfg = LoRADPOConfig()
        # DPO beta is typically in [0.01, 0.5]; values outside this range
        # indicate either too-aggressive or near-zero KL regularisation
        assert 0.01 <= cfg.beta <= 0.5, (
            f"DPO beta={cfg.beta} outside expected regularisation range [0.01, 0.5]"
        )

    def test_learning_rate_higher_than_full_dpo(self):
        # LoRA adapters (tiny parameter space) should use a larger LR than full fine-tuning
        cfg = LoRADPOConfig()
        assert cfg.learning_rate >= 5e-5, (
            "LoRA adapter LR should be higher than full fine-tuning LR (~1e-5)"
        )

    def test_lora_config_fields_present(self):
        cfg = LoRADPOConfig()
        assert cfg.lora_r > 0
        assert cfg.lora_alpha > 0
        assert len(cfg.target_modules) > 0
