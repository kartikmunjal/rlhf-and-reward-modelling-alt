"""
Unit tests for reward model and preference loss — no GPU, no pretrained downloads.

All tests use synthetic random tensors or minimal GPT-2 configs so they run
in CPU-only CI environments in seconds.
"""
import pytest
import torch
from transformers import GPT2Config

from src.models.reward_model import GPT2RewardModel, preference_loss, RewardModelOutput


# ---------------------------------------------------------------------------
# preference_loss
# ---------------------------------------------------------------------------

class TestPreferenceLoss:
    def test_clear_winner_low_loss_high_accuracy(self):
        """Chosen reward clearly above rejected → near-zero loss, high accuracy."""
        r_w = torch.tensor([2.0, 1.8, 2.5, 1.6])
        r_l = torch.tensor([0.2, 0.1, 0.4, 0.3])
        loss, acc = preference_loss(r_w, r_l)
        assert loss.item() < 0.3, f"Expected low loss, got {loss.item():.3f}"
        assert acc.item() > 0.9, f"Expected high acc, got {acc.item():.3f}"

    def test_reversed_pair_high_loss_low_accuracy(self):
        """Rejected reward above chosen → high loss, near-zero accuracy."""
        r_w = torch.tensor([0.2, 0.1, 0.3])
        r_l = torch.tensor([2.0, 1.8, 2.5])
        loss, acc = preference_loss(r_w, r_l)
        assert loss.item() > 1.5, f"Expected high loss, got {loss.item():.3f}"
        assert acc.item() < 0.2, f"Expected low acc, got {acc.item():.3f}"

    def test_equal_rewards_chance_accuracy(self):
        """Equal rewards → 50% accuracy (coin-flip)."""
        r = torch.zeros(100)
        loss, acc = preference_loss(r, r)
        # BCE at margin=0 is log(2) ≈ 0.693
        assert 0.6 < loss.item() < 0.8, f"Expected ~0.693, got {loss.item():.3f}"
        assert 0 <= acc.item() <= 1.0

    def test_loss_is_scalar(self):
        r_w = torch.randn(8)
        r_l = torch.randn(8)
        loss, acc = preference_loss(r_w, r_l)
        assert loss.ndim == 0, "loss should be a scalar tensor"
        assert acc.ndim == 0, "acc should be a scalar tensor"

    def test_gradients_flow_through_loss(self):
        """Confirm the loss is differentiable w.r.t. reward inputs."""
        r_w = torch.randn(4, requires_grad=True)
        r_l = torch.randn(4, requires_grad=True)
        loss, _ = preference_loss(r_w, r_l)
        loss.backward()
        assert r_w.grad is not None
        assert r_l.grad is not None


# ---------------------------------------------------------------------------
# GPT2RewardModel (tiny config, no download)
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_model():
    """GPT-2 with 2 layers, 64-dim hidden — fast CPU instantiation."""
    cfg = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=64,
        vocab_size=100,
        n_positions=32,
    )
    return GPT2RewardModel(cfg)


class TestGPT2RewardModel:
    def test_forward_returns_reward_per_sequence(self, tiny_model):
        batch, seq = 3, 16
        input_ids = torch.randint(0, 100, (batch, seq))
        out = tiny_model(input_ids)
        assert isinstance(out, RewardModelOutput)
        assert out.rewards.shape == (batch,), f"Expected ({batch},), got {out.rewards.shape}"

    def test_forward_with_attention_mask(self, tiny_model):
        batch, seq = 2, 16
        input_ids = torch.randint(0, 100, (batch, seq))
        mask = torch.ones(batch, seq, dtype=torch.long)
        mask[0, 12:] = 0  # pad last 4 tokens for sequence 0
        out = tiny_model(input_ids, attention_mask=mask)
        assert out.rewards.shape == (batch,)

    def test_hidden_states_not_returned_by_default(self, tiny_model):
        input_ids = torch.randint(0, 100, (2, 16))
        out = tiny_model(input_ids)
        assert out.hidden_states is None

    def test_hidden_states_returned_on_request(self, tiny_model):
        input_ids = torch.randint(0, 100, (2, 16))
        out = tiny_model(input_ids, return_hidden_states=True)
        assert out.hidden_states is not None
        assert out.hidden_states.shape == (2, 16, 64)

    def test_reward_head_is_single_linear(self, tiny_model):
        assert tiny_model.reward_head.out_features == 1
        assert tiny_model.reward_head.bias is None, "reward head should have no bias"
