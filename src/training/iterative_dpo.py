"""
Extension 7 — Iterative DPO: Self-Improving Alignment Loop.

Problem with standard DPO
--------------------------
Standard (offline) DPO trains on a fixed dataset of (prompt, chosen, rejected)
pairs collected from the SFT model.  Once training starts, the policy diverges
from the SFT model — but the preference pairs still reflect the SFT distribution.
This *distribution shift* makes the gradient signal increasingly stale across
training.  The pairs were labeled for a policy that no longer exists.

The fix: on-policy preference pairs
-------------------------------------
Iterative DPO alternates between:

    Phase 1 — Rollout: sample 2 responses per prompt from the *current policy*,
              score them with the reward model, label the better one "chosen".

    Phase 2 — DPO update: train for K steps on fresh pairs.

    Phase 3 — Evaluate: measure win rate vs SFT reference on held-out prompts.

The pairs are always from the current policy distribution, so the gradient is
never stale.  This is the simplest form of a "self-improving alignment loop" —
the model generates data, gets scored, and updates itself.

Key ablation: buffer strategy
-------------------------------
How much historical data to keep changes the stability/variance tradeoff:

  buffer="current"  — only pairs from this iteration.  Low variance but noisier.
  buffer="rolling2" — last 2 iterations.  Balanced.
  buffer="full"     — all historical pairs.  High signal but drifts off-distribution.

We run all three and compare win-rate evolution across iterations.  This directly
reproduces the core research question from 2024 iterative DPO papers.

Expected results
-----------------
After 3 iterations:
  Iteration 1: win rate ~55-60%
  Iteration 2: win rate ~60-65%
  Iteration 3: win rate ~65-68%

  KL from SFT reference grows more slowly than PPO because each DPO step
  is conservative — this is the main efficiency advantage over PPO.

  Full historical buffer may *degrade* in iteration 3 due to distribution
  mismatch — a Goodhart's Law parallel for iterative methods.

Relation to frontier research
-------------------------------
Anthropic's agents team explicitly researches "learning from experience" and
"self-improvement".  Iterative DPO is the simplest instantiation: generate,
score, update, repeat.  The same loop, at scale, is how RLHF models are
continuously improved post-deployment.
"""

from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel
from trl import DPOConfig, DPOTrainer

from src.data.preprocessing import DPODataset


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class IterativeDPOConfig:
    sft_checkpoint: str = "checkpoints/sft"
    reward_checkpoint: str = "checkpoints/reward"
    output_dir: str = "checkpoints/iterative_dpo"

    # Loop control
    num_iterations: int = 3
    rollout_batch_size: int = 256    # prompts to roll out per iteration
    dpo_steps_per_iter: int = 200    # gradient steps of DPO per iteration
    eval_prompts: int = 200          # held-out prompts for win-rate evaluation

    # Buffer strategy: which historical preference pairs to keep
    buffer_strategy: Literal["current", "rolling2", "full"] = "rolling2"

    # DPO hyper-parameters
    beta: float = 0.1
    learning_rate: float = 5e-7
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_length: int = 512
    max_prompt_length: int = 256
    max_new_tokens: int = 128

    # Generation
    generation_temperature: float = 0.8
    generation_top_p: float = 0.9

    fp16: bool = True
    report_to: str = "none"


# ── Preference buffer ──────────────────────────────────────────────────────────

class PreferenceBuffer:
    """Manages rolling window of on-policy (prompt, chosen, rejected) triples.

    Parameters
    ----------
    strategy:
        "current"  — only keep pairs generated in the most recent iteration.
        "rolling2" — keep pairs from the last two iterations.
        "full"     — accumulate all pairs from all iterations.
    """

    def __init__(self, strategy: Literal["current", "rolling2", "full"] = "rolling2"):
        self.strategy = strategy
        # Pairs stored per iteration: {iter_idx: [{"prompt", "chosen", "rejected"}, ...]}
        self._store: Dict[int, List[Dict]] = {}

    def add(self, iteration: int, pairs: List[Dict]) -> None:
        """Add preference pairs from a completed rollout iteration."""
        self._store[iteration] = pairs

    def get_training_pairs(self, current_iteration: int) -> List[Dict]:
        """Return the set of pairs to train on given the buffer strategy."""
        if self.strategy == "current":
            return list(self._store.get(current_iteration, []))
        elif self.strategy == "rolling2":
            pairs = []
            for i in [current_iteration - 1, current_iteration]:
                pairs.extend(self._store.get(i, []))
            return pairs
        else:  # "full"
            pairs = []
            for i in sorted(self._store.keys()):
                pairs.extend(self._store[i])
            return pairs

    def total_pairs(self) -> int:
        return sum(len(v) for v in self._store.values())

    def __repr__(self) -> str:
        per_iter = {k: len(v) for k, v in self._store.items()}
        return f"PreferenceBuffer(strategy={self.strategy!r}, per_iter={per_iter})"


# ── Trajectory metrics ─────────────────────────────────────────────────────────

@dataclass
class IterationResult:
    iteration: int
    win_rate: float        # fraction of eval prompts where policy beats SFT
    mean_reward: float     # mean reward model score on policy responses
    kl_from_sft: float     # mean KL(policy || SFT ref) on eval prompts
    n_pairs: int           # number of preference pairs used in this DPO update
    buffer_strategy: str


# ── Main training loop ─────────────────────────────────────────────────────────

def run_iterative_dpo(cfg: IterativeDPOConfig) -> List[IterationResult]:
    """Execute the iterative DPO loop and return per-iteration metrics.

    For each iteration:
      1. Rollout: generate 2 responses per prompt from the current policy,
         score them with the reward model, construct preference pairs.
      2. Update: run K DPO gradient steps on the buffer.
      3. Eval: measure win rate, mean reward, and KL on held-out prompts.

    Parameters
    ----------
    cfg:
        Full training configuration.

    Returns
    -------
    List of IterationResult (one per iteration), suitable for plotting.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load tokenizer & models ───────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Policy model — will be updated each iteration
    policy = GPT2LMHeadModel.from_pretrained(cfg.sft_checkpoint).to(device)
    policy.config.pad_token_id = tokenizer.pad_token_id

    # SFT reference — frozen throughout (used for KL and DPO loss)
    sft_ref = GPT2LMHeadModel.from_pretrained(cfg.sft_checkpoint).to(device)
    for p in sft_ref.parameters():
        p.requires_grad_(False)

    # Reward model — frozen throughout (used to score rollout responses)
    from src.models.reward_model import GPT2RewardModel
    reward_model = GPT2RewardModel.from_pretrained(cfg.reward_checkpoint).to(device)
    for p in reward_model.parameters():
        p.requires_grad_(False)
    reward_model.eval()

    # ── Load prompt pool ──────────────────────────────────────────────────────
    rollout_prompts, eval_prompts = _load_prompt_pools(
        cfg.sft_checkpoint, tokenizer,
        n_rollout=cfg.rollout_batch_size,
        n_eval=cfg.eval_prompts,
    )

    buffer = PreferenceBuffer(strategy=cfg.buffer_strategy)
    results: List[IterationResult] = []

    os.makedirs(cfg.output_dir, exist_ok=True)

    for iteration in range(1, cfg.num_iterations + 1):
        print(f"\n{'='*60}")
        print(f"  Iterative DPO  |  Iteration {iteration}/{cfg.num_iterations}")
        print(f"  Buffer strategy: {cfg.buffer_strategy}")
        print(f"{'='*60}")

        # ── Phase 1: Rollout ──────────────────────────────────────────────────
        print(f"\n[{iteration}] Phase 1 — Rollout ({len(rollout_prompts)} prompts)")
        new_pairs = _rollout_and_label(
            policy=policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            prompts=rollout_prompts,
            cfg=cfg,
            device=device,
        )
        buffer.add(iteration, new_pairs)
        print(f"  Generated {len(new_pairs)} preference pairs  |  {buffer}")

        # ── Phase 2: DPO update ───────────────────────────────────────────────
        training_pairs = buffer.get_training_pairs(iteration)
        print(f"\n[{iteration}] Phase 2 — DPO update on {len(training_pairs)} pairs")

        if len(training_pairs) < 4:
            print("  Too few pairs, skipping DPO update.")
        else:
            policy = _run_dpo_update(
                policy=policy,
                ref_model=sft_ref,
                tokenizer=tokenizer,
                pairs=training_pairs,
                cfg=cfg,
                iteration=iteration,
            )

        # ── Phase 3: Evaluate ─────────────────────────────────────────────────
        print(f"\n[{iteration}] Phase 3 — Evaluation on {len(eval_prompts)} prompts")
        win_rate, mean_reward, kl = _evaluate(
            policy=policy,
            sft_ref=sft_ref,
            reward_model=reward_model,
            tokenizer=tokenizer,
            prompts=eval_prompts,
            cfg=cfg,
            device=device,
        )

        result = IterationResult(
            iteration=iteration,
            win_rate=win_rate,
            mean_reward=mean_reward,
            kl_from_sft=kl,
            n_pairs=len(training_pairs),
            buffer_strategy=cfg.buffer_strategy,
        )
        results.append(result)

        print(
            f"\n  [Iter {iteration}]  "
            f"win_rate={win_rate:.4f}  "
            f"mean_reward={mean_reward:.4f}  "
            f"kl={kl:.4f}  "
            f"n_pairs={len(training_pairs)}"
        )

        # Save policy checkpoint after each iteration
        iter_dir = os.path.join(cfg.output_dir, f"iter_{iteration}")
        policy.save_pretrained(iter_dir)
        tokenizer.save_pretrained(iter_dir)

    # Persist results JSON
    results_path = os.path.join(cfg.output_dir, "iterative_dpo_results.json")
    with open(results_path, "w") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


# ── Phase 1: rollout and label ────────────────────────────────────────────────

def _rollout_and_label(
    policy: GPT2LMHeadModel,
    reward_model,
    tokenizer,
    prompts: List[str],
    cfg: IterativeDPOConfig,
    device: torch.device,
) -> List[Dict]:
    """Generate 2 responses per prompt; label by reward model score."""
    policy.eval()
    pairs = []

    for prompt in tqdm(prompts, desc="  Rollout"):
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=cfg.max_prompt_length,
            truncation=True,
        ).to(device)

        gen_kwargs = dict(
            max_new_tokens=cfg.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

        with torch.no_grad():
            out_a = policy.generate(
                **enc, do_sample=True,
                temperature=cfg.generation_temperature,
                top_p=cfg.generation_top_p,
                **gen_kwargs,
            )
            out_b = policy.generate(
                **enc, do_sample=True,
                temperature=cfg.generation_temperature + 0.2,
                top_p=0.95,
                **gen_kwargs,
            )

        prompt_len = enc["input_ids"].shape[1]
        resp_a = tokenizer.decode(out_a[0][prompt_len:], skip_special_tokens=True)
        resp_b = tokenizer.decode(out_b[0][prompt_len:], skip_special_tokens=True)

        # Skip nearly identical responses
        words_a = set(resp_a.lower().split())
        words_b = set(resp_b.lower().split())
        overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
        if overlap > 0.90:
            continue

        # Score with reward model
        r_a = _score_response(reward_model, tokenizer, prompt + resp_a, device)
        r_b = _score_response(reward_model, tokenizer, prompt + resp_b, device)

        chosen, rejected = (resp_a, resp_b) if r_a >= r_b else (resp_b, resp_a)
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    return pairs


def _score_response(reward_model, tokenizer, text: str, device: torch.device) -> float:
    """Return scalar reward for a single text string."""
    enc = tokenizer(
        text, return_tensors="pt", max_length=512, truncation=True,
    ).to(device)
    with torch.no_grad():
        out = reward_model(**enc)
    return out.rewards.item()


# ── Phase 2: DPO update ───────────────────────────────────────────────────────

def _run_dpo_update(
    policy: GPT2LMHeadModel,
    ref_model: GPT2LMHeadModel,
    tokenizer,
    pairs: List[Dict],
    cfg: IterativeDPOConfig,
    iteration: int,
) -> GPT2LMHeadModel:
    """Run one round of DPO training on the given preference pairs."""
    # Convert to HF Dataset format expected by trl DPOTrainer
    hf_pairs = HFDataset.from_list([
        {
            "prompt": p["prompt"],
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        }
        for p in pairs
    ])

    iter_dir = os.path.join(cfg.output_dir, f"dpo_iter_{iteration}")

    dpo_config = DPOConfig(
        beta=cfg.beta,
        output_dir=iter_dir,
        max_steps=cfg.dpo_steps_per_iter,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        logging_steps=20,
        save_strategy="no",
        fp16=cfg.fp16,
        report_to=cfg.report_to,
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=hf_pairs,
        tokenizer=tokenizer,
    )
    trainer.train()
    return trainer.model


# ── Phase 3: Evaluation ───────────────────────────────────────────────────────

def _evaluate(
    policy: GPT2LMHeadModel,
    sft_ref: GPT2LMHeadModel,
    reward_model,
    tokenizer,
    prompts: List[str],
    cfg: IterativeDPOConfig,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Return (win_rate, mean_reward, mean_kl) on held-out prompts.

    win_rate : fraction of prompts where policy reward > SFT reward
    mean_reward : average reward model score for policy responses
    mean_kl : average token-level KL divergence from SFT reference
    """
    policy.eval()
    sft_ref.eval()

    rewards_policy = []
    rewards_sft = []
    kl_values = []

    for prompt in tqdm(prompts, desc="  Eval"):
        enc = tokenizer(
            prompt, return_tensors="pt",
            max_length=cfg.max_prompt_length, truncation=True,
        ).to(device)

        gen_kwargs = dict(
            max_new_tokens=cfg.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=cfg.generation_temperature,
            top_p=cfg.generation_top_p,
        )

        with torch.no_grad():
            out_policy = policy.generate(**enc, **gen_kwargs)
            out_sft = sft_ref.generate(**enc, **gen_kwargs)

        prompt_len = enc["input_ids"].shape[1]
        resp_policy = tokenizer.decode(out_policy[0][prompt_len:], skip_special_tokens=True)
        resp_sft = tokenizer.decode(out_sft[0][prompt_len:], skip_special_tokens=True)

        r_p = _score_response(reward_model, tokenizer, prompt + resp_policy, device)
        r_s = _score_response(reward_model, tokenizer, prompt + resp_sft, device)
        rewards_policy.append(r_p)
        rewards_sft.append(r_s)

        # KL: use the policy response, measure divergence from SFT
        kl = _compute_kl(policy, sft_ref, tokenizer, prompt, resp_policy, device, cfg)
        kl_values.append(kl)

    win_rate = sum(rp > rs for rp, rs in zip(rewards_policy, rewards_sft)) / len(prompts)
    mean_reward = sum(rewards_policy) / len(rewards_policy)
    mean_kl = sum(kl_values) / len(kl_values)
    return win_rate, mean_reward, mean_kl


def _compute_kl(
    policy: GPT2LMHeadModel,
    ref: GPT2LMHeadModel,
    tokenizer,
    prompt: str,
    response: str,
    device: torch.device,
    cfg: IterativeDPOConfig,
) -> float:
    """Compute mean per-token KL divergence KL(policy || ref) for a response."""
    text = prompt + response
    enc = tokenizer(
        text, return_tensors="pt",
        max_length=cfg.max_length, truncation=True,
    ).to(device)

    with torch.no_grad():
        logits_p = policy(**enc).logits          # (1, T, V)
        logits_r = ref(**enc).logits             # (1, T, V)

    log_p = F.log_softmax(logits_p, dim=-1)
    log_r = F.log_softmax(logits_r, dim=-1)
    p = torch.exp(log_p)

    # KL(p || r) = sum_v p(v) * (log p(v) - log r(v))
    kl_per_token = (p * (log_p - log_r)).sum(dim=-1)  # (1, T)
    return kl_per_token.mean().item()


# ── Prompt loading ─────────────────────────────────────────────────────────────

def _load_prompt_pools(
    sft_checkpoint: str,
    tokenizer,
    n_rollout: int,
    n_eval: int,
) -> Tuple[List[str], List[str]]:
    """Load prompts from hh-rlhf test split; split into rollout and eval pools."""
    from datasets import load_dataset
    from src.data.preprocessing import extract_prompt_and_response

    raw = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="test")
    total = n_rollout + n_eval
    raw = raw.select(range(min(total, len(raw))))

    prompts = []
    for ex in raw:
        prompt, _ = extract_prompt_and_response(ex["chosen"])
        if len(prompt.split()) < 5:
            continue
        prompts.append(prompt)

    random.shuffle(prompts)
    rollout = prompts[:n_rollout]
    eval_p = prompts[n_rollout: n_rollout + n_eval]
    print(f"Loaded {len(rollout)} rollout prompts, {len(eval_p)} eval prompts")
    return rollout, eval_p


# ── Comparison against PPO and single-pass DPO ───────────────────────────────

def compare_with_baselines(
    iterative_results: List[IterationResult],
    ppo_win_rate: float = 0.712,
    ppo_kl: float = 4.821,
    single_dpo_win_rate: float = 0.634,
    single_dpo_kl: float = 1.734,
) -> Dict:
    """Format a comparison table against PPO and single-pass DPO baselines."""
    final = iterative_results[-1]

    rows = [
        {"method": "SFT (baseline)", "win_rate": 0.500, "kl": 0.000,
         "notes": "reference point"},
        {"method": "Single-pass DPO", "win_rate": single_dpo_win_rate, "kl": single_dpo_kl,
         "notes": "offline, fixed data"},
        {"method": f"Iterative DPO iter 1 ({iterative_results[0].buffer_strategy})",
         "win_rate": iterative_results[0].win_rate, "kl": iterative_results[0].kl_from_sft,
         "notes": "on-policy pairs"},
    ]
    for r in iterative_results[1:]:
        rows.append({
            "method": f"Iterative DPO iter {r.iteration} ({r.buffer_strategy})",
            "win_rate": r.win_rate, "kl": r.kl_from_sft,
            "notes": "on-policy pairs",
        })
    rows.append({
        "method": "PPO",
        "win_rate": ppo_win_rate,
        "kl": ppo_kl,
        "notes": "on-policy rollouts + RM",
    })

    print(f"\n{'Method':<40} {'Win Rate':>10} {'KL':>8}  Notes")
    print("-" * 75)
    for row in rows:
        print(f"{row['method']:<40} {row['win_rate']:>10.4f} {row['kl']:>8.4f}  {row['notes']}")

    return {"rows": rows, "final_iterative": final.__dict__}
