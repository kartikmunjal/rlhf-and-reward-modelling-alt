# RLHF Pipeline: Reward Modeling & PPO vs DPO

An end-to-end implementation of the three-stage RLHF (Reinforcement Learning from Human Feedback) pipeline on GPT-2, trained on Anthropic's `hh-rlhf` preference dataset.

The goal is a clear, runnable demonstration of the full alignment stack — not just calling a trainer, but understanding *why* each component exists and *what happens* when you remove it.

---

## The Pipeline

```
Anthropic hh-rlhf dataset
        │
        ▼
┌───────────────────┐
│   Stage 1: SFT    │  Fine-tune GPT-2 on chosen responses
│   (behavioral     │  → checkpoints/sft/
│    cloning)       │  → becomes π_ref (frozen reference)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Stage 2: Reward  │  Train Bradley-Terry reward model
│  Model Training   │  Loss: -log σ(r_chosen - r_rejected)
│                   │  → checkpoints/reward_model/
└────┬──────────────┘
     │
     ├──────────────────────────────────────┐
     ▼                                      ▼
┌──────────────┐                    ┌──────────────┐
│  Stage 3a:   │                    │  Stage 3b:   │
│     PPO      │                    │     DPO      │
│              │                    │              │
│ 4 models     │                    │ 2 models     │
│ On-policy    │                    │ Offline data │
│ Needs RM     │                    │ No RM needed │
└──────────────┘                    └──────────────┘
```

---

## Results

Evaluated on 500 prompts from the hh-rlhf test split (gpt2-medium backbone):

| Model | Mean Reward | Reward Std | Win Rate vs SFT | KL from Ref | Train Time |
|-------|-------------|------------|-----------------|-------------|------------|
| SFT (baseline) | 0.212 | 0.318 | — | 0.000 | 2h 10m |
| PPO (β=0.2) | 0.681 | 0.241 | 71.2% | 4.821 | 5h 45m |
| DPO (β=0.1) | 0.543 | 0.274 | 63.4% | 1.734 | 1h 30m |

**Key insight**: DPO achieves ~70% of PPO's reward gain with ~36% of the KL divergence — significantly better alignment efficiency at a fraction of the compute cost.

---

## Repository Structure

```
.
├── src/
│   ├── data/
│   │   └── preprocessing.py     # SFTDataset, PreferenceDataset, DPODataset
│   ├── models/
│   │   └── reward_model.py      # GPT2RewardModel + Bradley-Terry preference_loss
│   ├── training/
│   │   ├── sft.py               # Supervised fine-tuning
│   │   ├── reward.py            # Reward model training loop
│   │   ├── ppo.py               # PPO with trl + custom reward scoring
│   │   └── dpo.py               # DPO loss from scratch + trl trainer
│   └── evaluation/
│       └── metrics.py           # win_rate, reward_stats, kl_divergence
├── scripts/
│   ├── train_sft.py
│   ├── train_reward_model.py
│   ├── train_ppo.py
│   ├── train_dpo.py
│   └── evaluate.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sft_training.ipynb
│   ├── 03_reward_modeling.ipynb
│   ├── 04_ppo_training.ipynb
│   ├── 05_dpo_training.ipynb
│   └── 06_ppo_vs_dpo_comparison.ipynb
└── configs/
    ├── sft_config.yaml
    ├── reward_config.yaml
    ├── ppo_config.yaml
    └── dpo_config.yaml
```

---

## Quick Start

### 1. Environment

```bash
git clone https://github.com/kartikmunjal/rlhf-and-reward-modelling.git
cd rlhf-and-reward-modelling
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

**GPU requirements**:
- gpt2 (124M): free Colab T4 (15 GB)
- gpt2-medium (355M): Colab Pro A100 or any GPU with ≥ 22 GB VRAM

### 2. Run the full pipeline

```bash
# Stage 1: SFT
python scripts/train_sft.py --num_samples 10000 --epochs 3

# Stage 2: Reward Model (initialised from SFT checkpoint)
python scripts/train_reward_model.py --num_samples 10000 --epochs 2

# Stage 3a: PPO
python scripts/train_ppo.py --num_samples 5000

# Stage 3b: DPO (no reward model needed)
python scripts/train_dpo.py --beta 0.1 --num_samples 10000

# Evaluate all three policies
python scripts/evaluate.py \
    --ppo_checkpoint checkpoints/ppo \
    --dpo_checkpoint checkpoints/dpo \
    --num_eval 500
```

### 3. Smoke test (no GPU)

```bash
python scripts/train_sft.py --model gpt2 --num_samples 200 --epochs 1 --no_fp16
python scripts/train_reward_model.py --num_samples 200 --epochs 1 --no_fp16
```

### 4. Notebooks (Google Colab)

Each notebook has an "Open in Colab" badge. Run them in order:

| Notebook | Topic |
|----------|-------|
| [01_data_exploration](notebooks/01_data_exploration.ipynb) | Dataset statistics and conversation structure |
| [02_sft_training](notebooks/02_sft_training.ipynb) | SFT: behavioral cloning on chosen responses |
| [03_reward_modeling](notebooks/03_reward_modeling.ipynb) | Bradley-Terry reward model training |
| [04_ppo_training](notebooks/04_ppo_training.ipynb) | PPO: on-policy RL with reward model signal |
| [05_dpo_training](notebooks/05_dpo_training.ipynb) | DPO: offline preference learning, full derivation |
| [06_ppo_vs_dpo_comparison](notebooks/06_ppo_vs_dpo_comparison.ipynb) | Quantitative and qualitative comparison |

---

## Conceptual Stack

### Stage 1: Supervised Fine-Tuning (SFT)

**Why**: Base GPT-2 has never seen Human/Assistant dialogue. Without SFT, RL starts from a broken distribution and cannot make meaningful progress.

**Objective**: Standard causal LM loss, masked to response tokens only:

$$\mathcal{L}_{SFT} = -\sum_{t \in \text{response}} \log \pi_\theta(y_t \mid y_{<t}, x)$$

**Key detail**: Prompt tokens get label `-100` (PyTorch's ignore index), so the gradient flows only through the response.

---

### Stage 2: Reward Modeling

**Why**: RL requires a scalar reward signal. We can't query humans at RL training time — we train a *proxy* that predicts human preferences from pairs.

**Model**: GPT-2 transformer backbone + `Linear(n_embd → 1, bias=False)` head pooling the last non-padding token hidden state.

**Bradley-Terry preference loss**:

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\!\left[\log \sigma\!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right]$$

This maximises the margin between chosen and rejected rewards. The model is initialised from the SFT checkpoint (not raw GPT-2) because the SFT model already understands the conversation format.

**Evaluation**: Pairwise accuracy on held-out pairs. Random baseline = 0.50; well-trained RM = 0.70+.

---

### Stage 3a: Proximal Policy Optimization (PPO)

**Why PPO and not REINFORCE**: Standard policy gradient has high variance and can take catastrophically large steps. PPO's clipped surrogate objective limits update magnitude:

$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\left(\rho_t A_t,\; \text{clip}(\rho_t, 1{-}\varepsilon, 1{+}\varepsilon)\, A_t\right)\right]$$

where $\rho_t = \pi_\theta(a_t|s_t) / \pi_{\theta_{old}}(a_t|s_t)$.

**KL penalty** (essential for alignment):

$$\mathcal{L}^{PPO+KL} = \mathcal{L}^{CLIP} - \beta \cdot \mathbb{KL}[\pi_\theta \| \pi_{ref}]$$

Without the KL term, the policy will exploit the reward model's blind spots (reward hacking). An adaptive KL controller adjusts $\beta$ to keep KL near `target_kl`.

**Training infrastructure**: 4 models required simultaneously — policy (trained), value network (separate linear head), reference policy (frozen SFT), reward model (frozen).

---

### Stage 3b: Direct Preference Optimization (DPO)

**Key insight**: The KL-constrained RLHF objective has a closed-form optimal policy:

$$\pi^*(y|x) \propto \pi_{ref}(y|x) \exp\!\left(\frac{r(x,y)}{\beta}\right)$$

Inverting this gives the reward as a function of the optimal policy:

$$r^*(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

Substituting into the Bradley-Terry loss, $\log Z(x)$ cancels (same for both responses to the same prompt), yielding:

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x,y_w,y_l)}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

**What this means**: the log-ratio $\log(\pi_\theta/\pi_{ref})$ acts as an *implicit reward*. DPO learns this implicit reward without ever instantiating an explicit reward model or generating rollouts.

---

## Key Hyperparameters

### SFT
- `learning_rate=2e-5` — standard for full fine-tuning
- `max_length=512` — covers ~95% of examples without truncation
- `warmup_ratio=0.05` — prevents early training instability

### Reward Model
- Initialise from SFT checkpoint (not raw GPT-2)
- `learning_rate=1e-5` — more conservative than SFT
- Monitor `pairwise_accuracy` alongside loss (random = 0.5, target ≥ 0.7)

### PPO
- `init_kl_coef=0.2`, `target_kl=6.0` — adaptive controller keeps KL in reasonable range
- `ppo_epochs=4` — standard; too many causes stale-rollout overfitting
- `clip_range=0.2` — the ε parameter; rarely needs changing

### DPO
- `beta=0.1` — the paper default; higher β → more conservative, lower reward
- `learning_rate=5e-7` — ~100× smaller than SFT; we are nudging, not retraining
- `max_prompt_length=256` — ensures response gets sufficient sequence budget

---

## References

- [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) — Original RLHF paper from OpenAI
- [Anthropic hh-rlhf dataset (Bai et al., 2022)](https://arxiv.org/abs/2204.05862) — The preference dataset used here
- [DPO (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290) — Direct Preference Optimization
- [PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) — Proximal Policy Optimization
- [TRL library](https://github.com/huggingface/trl) — HuggingFace Transformer Reinforcement Learning
