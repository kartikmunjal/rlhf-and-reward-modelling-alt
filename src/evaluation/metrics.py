"""
Evaluation metrics for comparing SFT, PPO, and DPO policies.

Three complementary metrics
----------------------------
1. **Win rate** (proxy for human preference)
   Given the reward model as a proxy judge, we compare two models by generating
   responses to the same prompts and checking which gets the higher reward score.
   Win rate is the fraction of prompts where model A beats model B.
   Limitation: biased toward whatever the reward model rewards, which may not
   fully reflect human intent.

2. **Reward statistics** (distribution analysis)
   Mean and std of reward scores across a prompt set.  Tells us whether a policy
   found higher-reward regions *and* whether it has collapsed to a narrow mode
   (reward hacking symptom: very high mean, very low std).

3. **KL divergence from reference**
   Measures how far the policy has drifted from the SFT reference model.
   High KL + high reward → possible reward hacking.
   Low KL + high reward → genuine alignment improvement.
   Computed as E[log π_θ(y|x) − log π_ref(y|x)] over a sample of outputs.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2LMHeadModel
from tqdm.auto import tqdm

from src.models.reward_model import GPT2RewardModel


@torch.no_grad()
def compute_win_rate(
    policy_a: GPT2LMHeadModel,
    policy_b: GPT2LMHeadModel,
    reward_model: GPT2RewardModel,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int = 128,
    batch_size: int = 8,
) -> Dict[str, float]:
    """Estimate win rate of policy A vs policy B using the reward model as judge.

    For each prompt we:
      1. Generate a response from A and from B.
      2. Score both with the reward model.
      3. A "wins" if r(A_response) > r(B_response).

    Returns
    -------
    dict with keys:
      "win_rate_a"  — fraction of prompts where A wins
      "win_rate_b"  — fraction of prompts where B wins
      "tie_rate"    — fraction of prompts where |r_A − r_B| < 0.01
      "mean_reward_a", "mean_reward_b"
    """
    policy_a.eval(); policy_b.eval(); reward_model.eval()
    policy_a.to(device); policy_b.to(device); reward_model.to(device)

    wins_a, wins_b, ties = 0, 0, 0
    rewards_a_all, rewards_b_all = [], []

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        enc = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=256,
        ).to(device)

        # Generate from A
        out_a = policy_a.generate(**enc, **gen_kwargs)
        # Generate from B
        out_b = policy_b.generate(**enc, **gen_kwargs)

        def score_batch(outputs):
            texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            enc_r = tokenizer(
                texts, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(device)
            return reward_model(**enc_r).rewards

        r_a = score_batch(out_a)
        r_b = score_batch(out_b)

        rewards_a_all.extend(r_a.cpu().tolist())
        rewards_b_all.extend(r_b.cpu().tolist())

        diff = r_a - r_b
        wins_a += (diff > 0.01).sum().item()
        wins_b += (diff < -0.01).sum().item()
        ties += (diff.abs() <= 0.01).sum().item()

    n = len(prompts)
    return {
        "win_rate_a": wins_a / n,
        "win_rate_b": wins_b / n,
        "tie_rate": ties / n,
        "mean_reward_a": sum(rewards_a_all) / n,
        "mean_reward_b": sum(rewards_b_all) / n,
    }


@torch.no_grad()
def compute_reward_stats(
    policy: GPT2LMHeadModel,
    reward_model: GPT2RewardModel,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int = 128,
    batch_size: int = 8,
) -> Dict[str, float]:
    """Compute mean, std, min, max of reward scores over a set of prompts."""
    policy.eval(); reward_model.eval()
    policy.to(device); reward_model.to(device)

    all_rewards = []
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    for i in tqdm(range(0, len(prompts), batch_size), desc="Scoring"):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        out = policy.generate(**enc, **gen_kwargs)
        texts = tokenizer.batch_decode(out, skip_special_tokens=True)
        enc_r = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        rewards = reward_model(**enc_r).rewards
        all_rewards.extend(rewards.cpu().tolist())

    t = torch.tensor(all_rewards)
    return {
        "mean": t.mean().item(),
        "std": t.std().item(),
        "min": t.min().item(),
        "max": t.max().item(),
        "p25": t.quantile(0.25).item(),
        "p75": t.quantile(0.75).item(),
    }


@torch.no_grad()
def compute_kl_divergence(
    policy: GPT2LMHeadModel,
    ref_policy: GPT2LMHeadModel,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int = 64,
    n_samples: int = 200,
) -> float:
    """Estimate KL(π_θ ‖ π_ref) via Monte Carlo sampling.

    For each prompt x, we sample y ~ π_θ(·|x) and estimate:

        KL(π_θ ‖ π_ref | x)  ≈  log π_θ(y|x) − log π_ref(y|x)

    We average over (x, y) pairs for the final estimate.

    High KL indicates the policy has drifted significantly from the reference.
    """
    policy.eval(); ref_policy.eval()
    policy.to(device); ref_policy.to(device)

    kl_vals = []
    prompts = prompts[:n_samples]

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )

    for prompt in tqdm(prompts, desc="KL estimate"):
        enc = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(device)

        # Sample y from the policy
        with torch.no_grad():
            out = policy.generate(**enc, **gen_kwargs)

        full_ids = out  # (1, T)
        attn_mask = (full_ids != tokenizer.pad_token_id).long()

        # Compute log π_θ(y|x)
        logits_policy = policy(full_ids, attention_mask=attn_mask).logits
        # Compute log π_ref(y|x)
        logits_ref = ref_policy(full_ids, attention_mask=attn_mask).logits

        # Average log-prob over response tokens
        logp_policy = _mean_token_logp(logits_policy, full_ids, attn_mask)
        logp_ref = _mean_token_logp(logits_ref, full_ids, attn_mask)

        kl_vals.append((logp_policy - logp_ref).item())

    return sum(kl_vals) / len(kl_vals)


def _mean_token_logp(
    logits: torch.Tensor,       # (1, T, V)
    input_ids: torch.Tensor,    # (1, T)
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_ids = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()
    logp = F.log_softmax(shift_logits, dim=-1)
    token_logp = logp.gather(2, shift_ids.unsqueeze(-1)).squeeze(-1)
    return (token_logp * shift_mask).sum() / shift_mask.sum().clamp(min=1)


def generate_comparison_table(results: Dict[str, Dict]) -> str:
    """Format a markdown comparison table from evaluation results.

    Parameters
    ----------
    results:
        Mapping from model name (e.g. "SFT", "PPO", "DPO") to metric dicts.

    Returns
    -------
    Markdown-formatted table string.
    """
    headers = ["Model", "Mean Reward", "Std Reward", "Win vs SFT (%)", "KL from Ref"]
    rows = []
    for name, metrics in results.items():
        rows.append([
            name,
            f"{metrics.get('mean_reward', float('nan')):.4f}",
            f"{metrics.get('std_reward', float('nan')):.4f}",
            f"{metrics.get('win_rate_vs_sft', float('nan')) * 100:.1f}%",
            f"{metrics.get('kl_from_ref', float('nan')):.4f}",
        ])

    col_widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_widths) + " |"
    sep = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"

    lines = [fmt.format(*headers), sep]
    for row in rows:
        lines.append(fmt.format(*row))
    return "\n".join(lines)
