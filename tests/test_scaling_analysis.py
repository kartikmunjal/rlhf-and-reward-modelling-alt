"""
Unit tests for scaling analysis — pure arithmetic, no GPU, no model downloads.

Verifies the memory derivations from first principles:
  bf16 weights    : 2P bytes
  fp32 gradients  : 4P bytes
  fp32 master wts : 4P bytes
  Adam m + v      : 8P bytes
  ─────────────────────────────
  Total (mixed)   : 18P bytes/param under DDP
  FSDP FULL_SHARD : 18P/N bytes/param with N GPUs
"""
import pytest

from src.analysis.scaling_analysis import (
    ModelSpec,
    MemoryBreakdown,
    compute_memory_breakdown,
    compute_fsdp_per_gpu,
    lora_memory_savings,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gpt2_small():
    return ModelSpec(
        name="gpt2-small",
        num_params=117_000_000,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        vocab_size=50_257,
        seq_len=512,
    )


@pytest.fixture
def gpt2_medium():
    return ModelSpec(
        name="gpt2-medium",
        num_params=355_000_000,
        hidden_dim=1024,
        num_layers=24,
        num_heads=16,
        vocab_size=50_257,
        seq_len=512,
    )


# ---------------------------------------------------------------------------
# compute_memory_breakdown
# ---------------------------------------------------------------------------

class TestMemoryBreakdown:
    def test_returns_memory_breakdown_instance(self, gpt2_small):
        result = compute_memory_breakdown(gpt2_small)
        assert isinstance(result, MemoryBreakdown)

    def test_18_bytes_per_param_rule(self, gpt2_small):
        """Total mixed-precision DDP memory = 18 bytes/param."""
        result = compute_memory_breakdown(gpt2_small)
        expected_gb = 18 * gpt2_small.num_params / 1e9
        assert abs(result.total_ddp_mixed_gb - expected_gb) < 0.1, (
            f"Expected {expected_gb:.2f} GB (18 bytes/param), "
            f"got {result.total_ddp_mixed_gb:.2f} GB"
        )

    def test_mixed_precision_exceeds_weights_alone(self, gpt2_small):
        result = compute_memory_breakdown(gpt2_small)
        assert result.total_ddp_mixed_gb > result.weights_bf16_gb

    def test_larger_model_more_memory(self, gpt2_small, gpt2_medium):
        small = compute_memory_breakdown(gpt2_small)
        medium = compute_memory_breakdown(gpt2_medium)
        assert medium.total_ddp_mixed_gb > small.total_ddp_mixed_gb

    def test_activation_checkpointing_reduces_activation_memory(self, gpt2_small):
        result = compute_memory_breakdown(gpt2_small)
        assert result.activations_checkpointed_gb < result.activations_full_gb

    def test_all_memory_fields_non_negative(self, gpt2_small):
        result = compute_memory_breakdown(gpt2_small)
        for field_name, value in vars(result).items():
            assert value >= 0, f"Field '{field_name}' should be >= 0, got {value}"


# ---------------------------------------------------------------------------
# compute_fsdp_per_gpu
# ---------------------------------------------------------------------------

class TestFSDPPerGpu:
    def test_full_shard_with_more_gpus_saves_memory(self, gpt2_medium):
        breakdown = compute_memory_breakdown(gpt2_medium)
        one_gpu  = compute_fsdp_per_gpu(breakdown, num_gpus=1, sharding_strategy="FULL_SHARD")
        eight_gpu = compute_fsdp_per_gpu(breakdown, num_gpus=8, sharding_strategy="FULL_SHARD")
        assert eight_gpu["total_peak_gb"] < one_gpu["total_peak_gb"], (
            "ZeRO-3 (FULL_SHARD) should reduce per-GPU peak memory as N increases"
        )

    def test_full_shard_beats_no_shard(self, gpt2_medium):
        breakdown = compute_memory_breakdown(gpt2_medium)
        no_shard   = compute_fsdp_per_gpu(breakdown, num_gpus=4, sharding_strategy="NO_SHARD")
        full_shard = compute_fsdp_per_gpu(breakdown, num_gpus=4, sharding_strategy="FULL_SHARD")
        assert full_shard["total_peak_gb"] < no_shard["total_peak_gb"], (
            "FULL_SHARD (ZeRO-3) should beat NO_SHARD (ZeRO-0) for 4+ GPUs"
        )

    def test_shard_grad_op_between_no_and_full(self, gpt2_medium):
        breakdown = compute_memory_breakdown(gpt2_medium)
        no_shard    = compute_fsdp_per_gpu(breakdown, num_gpus=4, sharding_strategy="NO_SHARD")
        shard_grad  = compute_fsdp_per_gpu(breakdown, num_gpus=4, sharding_strategy="SHARD_GRAD_OP")
        full_shard  = compute_fsdp_per_gpu(breakdown, num_gpus=4, sharding_strategy="FULL_SHARD")
        assert no_shard["total_peak_gb"] >= shard_grad["total_peak_gb"] >= full_shard["total_peak_gb"]

    def test_single_gpu_no_shard_matches_ddp(self, gpt2_small):
        """1 GPU, NO_SHARD = DDP baseline (no sharding benefit, no overhead)."""
        ddp = compute_memory_breakdown(gpt2_small)
        fsdp = compute_fsdp_per_gpu(ddp, num_gpus=1, sharding_strategy="NO_SHARD",
                                    use_checkpointing=False)
        assert abs(fsdp["total_peak_gb"] - ddp.peak_mixed_gb) < 0.5


# ---------------------------------------------------------------------------
# lora_memory_savings
# ---------------------------------------------------------------------------

class TestLoraMemorySavings:
    def test_trainable_ratio_under_one_percent(self, gpt2_medium):
        result = lora_memory_savings(gpt2_medium, lora_r=16)
        assert result["trainable_ratio"] < 0.01, (
            f"LoRA r=16 should train <1% of GPT-2-medium params, "
            f"got {result['trainable_ratio']:.2%}"
        )

    def test_higher_rank_more_trainable_params(self, gpt2_medium):
        low  = lora_memory_savings(gpt2_medium, lora_r=4)
        high = lora_memory_savings(gpt2_medium, lora_r=64)
        assert high["lora_trainable_params"] > low["lora_trainable_params"]

    def test_optimizer_savings_are_positive(self, gpt2_medium):
        result = lora_memory_savings(gpt2_medium, lora_r=16)
        assert result["optim_savings_gb"] > 0, (
            "LoRA should save optimizer memory vs full fine-tuning"
        )

    def test_lora_total_less_than_full_ft_total(self, gpt2_medium):
        result = lora_memory_savings(gpt2_medium, lora_r=16)
        assert result["total_lora_gb"] < result["total_full_ft_gb"]
