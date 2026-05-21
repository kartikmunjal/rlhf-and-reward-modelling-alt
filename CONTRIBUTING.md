# Development Guide

How to set up the environment, run the full pipeline end-to-end, and navigate the notebooks.

---

## Setup

```bash
git clone https://github.com/kartikmunjal/rlhf-reward-modeling
cd rlhf-reward-modeling
python -m venv .venv && source .venv/bin/activate

# GPU training (recommended)
pip install torch --index-url https://download.pytorch.org/whl/cu121
# CPU only (for tests and scaling analysis scripts)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[dev]"        # installs the package + pytest + black + ruff
```

---

## Running Tests

```bash
pytest tests/ -v
```

All tests run on CPU and complete in under 30 seconds. No GPU or model downloads are required.

---

## Full Pipeline — End to End

The core pipeline runs in five sequential steps. Each step saves a checkpoint that the next step
consumes. Estimated times are for `gpt2-medium` on a single A100 40 GB.

### Step 1 — Supervised Fine-Tuning

```bash
python scripts/train_sft.py
# Checkpoint: checkpoints/sft/
# Time: ~2h
```

This produces the behavioral cloning baseline and the frozen reference policy used by PPO and DPO.

### Step 2 — Reward Model

```bash
python scripts/train_reward_model.py \
    --sft_checkpoint checkpoints/sft \
    --output_dir checkpoints/reward_model
# Time: ~1h
```

### Step 3a — PPO (on-policy, requires RM)

```bash
python scripts/train_ppo.py \
    --sft_checkpoint checkpoints/sft \
    --rm_checkpoint checkpoints/reward_model
# Time: ~5–6h
# Note: PPO is sensitive to KL coefficient. Use beta=0.2 with adaptive KL control.
# See README Finding #5 for what happens without it.
```

### Step 3b — DPO (recommended starting point)

```bash
python scripts/train_dpo.py \
    --sft_checkpoint checkpoints/sft
# Time: ~1.5h
# No reward model needed; trains directly from preference pairs.
```

### Step 3c — GRPO

```bash
python scripts/train_grpo.py \
    --sft_checkpoint checkpoints/sft
# Group-normalized reward — no value head, lower memory than PPO.
```

### Evaluation

```bash
python scripts/evaluate.py \
    --sft checkpoints/sft \
    --dpo checkpoints/dpo \
    --n_prompts 500
```

---

## Extensions

Most extensions have a `--show_expected` flag that prints expected results without running
a full training job. Use this to verify the scaffold before committing GPU time.

```bash
# Confidence filtering flywheel (preview)
python scripts/run_confidence_flywheel.py --show_expected

# Reward ensemble lambda sweep
python scripts/run_ensemble_lambda_sweep.py --show_expected

# Scaling analysis (CPU, no GPU needed — runs in seconds)
python scripts/analyze_scaling.py

# CAI preference generation (requires Anthropic API key)
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/generate_cai_preferences.py --num_pairs 2000 --output data/cai_preferences.jsonl
```

---

## Notebook Order

For a guided walkthrough of the full pipeline:

| Notebook | Content |
|----------|---------|
| `01_data_exploration.ipynb` | Dataset statistics, preference pair analysis |
| `02_sft_training.ipynb` | SFT loss curves, generation samples |
| `03_reward_modeling.ipynb` | RM accuracy, length bias, calibration |
| `04_ppo_training.ipynb` | PPO reward/KL curves, verbose bias emergence |
| `05_dpo_training.ipynb` | DPO loss, comparison to SFT |
| `06_ppo_vs_dpo_comparison.ipynb` | Side-by-side reward, KL, win rate |

For extensions, use the [Reading Guide](README.md#reading-guide) to navigate by topic.

---

## Hardware Requirements

| Experiment | Min GPU VRAM | Approximate Time |
|------------|-------------|-----------------|
| SFT (gpt2-medium) | 16 GB | ~2h |
| Reward model | 16 GB | ~1h |
| DPO | 16 GB | ~1.5h |
| PPO | 24 GB | ~5–6h |
| FSDP (2-GPU) | 2 × 16 GB | ~3h |
| Scaling analysis, tests | CPU only | < 1 min |

Gradient accumulation is configured by default (`gradient_accumulation_steps=4`) to fit on
smaller GPUs. Reduce `batch_size` further if needed.
