"""
Rubric Reward Model training and evaluation (Extension 15).

Trains GPT2RewardModel using MSE loss against Claude-graded rubric scores
instead of the Bradley-Terry pairwise preference loss. Provides comparison
utilities to measure pairwise accuracy, length bias, and OOD calibration
against the standard BT reward model.

Key difference vs Bradley-Terry
---------------------------------
BT RM:     trained on pairs (chosen, rejected) → -log σ(r_chosen - r_rejected)
Rubric RM: trained on individual responses → MSE(predicted, rubric_normalized_score)

The rubric RM can be used for pairwise comparison at test time by computing
rubric_score(chosen) > rubric_score(rejected), but it wasn't trained on pairs.
This matters because the Conciseness criterion makes verbosity harmful to the
rubric score, while the BT RM has no such explicit disincentive.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class RubricRewardConfig:
    """Training configuration for the Rubric Reward Model."""

    # Data
    rubric_data_path: str = "data/rubric_scored_pairs.jsonl"
    output_dir: str = "checkpoints/rubric_reward"
    sft_checkpoint: str = "checkpoints/sft"

    # Training
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_length: int = 512
    fp16: bool = True
    log_every: int = 50

    # Evaluation split
    eval_fraction: float = 0.1  # fraction of rubric data held out for eval


# ── Loss function ──────────────────────────────────────────────────────────────

def rubric_mse_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """MSE loss for rubric regression.

    Parameters
    ----------
    predicted : (B,) scalar rewards from GPT2RewardModel
    target    : (B,) normalized rubric scores in [0, 1]

    Returns
    -------
    scalar MSE loss
    """
    return F.mse_loss(predicted, target)


# ── Training loop ──────────────────────────────────────────────────────────────

def train_rubric_reward_model(cfg: RubricRewardConfig) -> None:
    """Train GPT2RewardModel with MSE loss against rubric scores.

    The architecture is identical to the BT reward model. Only the loss changes:
    preference_loss → rubric_mse_loss. This isolates the effect of the training
    signal (pairwise rankings vs. absolute rubric grades) from architecture.
    """
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
    from torch.optim import AdamW

    from src.models.reward_model import GPT2RewardModel
    from src.data.rubric_preferences import RubricScoredDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2RewardModel.from_sft_checkpoint(cfg.sft_checkpoint).to(device)

    full_dataset = RubricScoredDataset.from_jsonl(cfg.rubric_data_path, tokenizer, cfg.max_length)

    # Split train / eval
    n_eval = max(1, int(len(full_dataset) * cfg.eval_fraction))
    n_train = len(full_dataset) - n_eval
    train_ds, eval_ds = torch.utils.data.random_split(full_dataset, [n_train, n_eval])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    eval_loader  = DataLoader(eval_ds,  batch_size=cfg.batch_size, shuffle=False)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    total_steps = len(train_loader) * cfg.num_epochs // cfg.gradient_accumulation_steps
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16 and device.type == "cuda")
    optimizer.zero_grad()
    global_step = 0

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_scores  = batch["target_score"].to(device)

            with torch.cuda.amp.autocast(enabled=cfg.fp16 and device.type == "cuda"):
                predicted = model(input_ids, attention_mask).rewards
                loss = rubric_mse_loss(predicted, target_scores)
                loss = loss / cfg.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * cfg.gradient_accumulation_steps
            n_batches  += 1

            if global_step % cfg.log_every == 0:
                print(f"  epoch {epoch+1} step {global_step}: loss={epoch_loss/n_batches:.4f}")

        eval_metrics = _eval_pass_mse(model, eval_loader, device)
        print(
            f"Epoch {epoch+1}/{cfg.num_epochs}: "
            f"train_loss={epoch_loss/n_batches:.4f}  "
            f"eval_loss={eval_metrics['eval_loss']:.4f}  "
            f"eval_pairwise_acc={eval_metrics.get('pairwise_acc', 0.0):.3f}"
        )

    os.makedirs(cfg.output_dir, exist_ok=True)
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"\nRubric RM saved to {cfg.output_dir}")


def _eval_pass_mse(model, loader, device) -> Dict:
    """MSE evaluation + pairwise accuracy on sequential (chosen, rejected) pairs."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            preds = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            ).rewards.cpu()
            all_preds.extend(preds.tolist())
            all_targets.extend(batch["target_score"].tolist())

    preds   = torch.tensor(all_preds)
    targets = torch.tensor(all_targets)
    mse     = F.mse_loss(preds, targets).item()

    # Pairwise accuracy: even-index = chosen, odd-index = rejected (if dataset alternates)
    pairwise_acc = 0.0
    n_pairs = len(all_preds) // 2
    if n_pairs > 0:
        correct = sum(
            all_preds[2*i] > all_preds[2*i + 1]
            for i in range(n_pairs)
        )
        pairwise_acc = correct / n_pairs

    return {"eval_loss": mse, "pairwise_acc": pairwise_acc}


# ── Length bias evaluation ─────────────────────────────────────────────────────

def evaluate_length_bias(
    model,
    tokenizer,
    prompts: List[str],
    responses: List[str],
    device: torch.device,
    filler: Optional[str] = None,
    max_length: int = 512,
) -> float:
    """Measure the average reward delta when appending filler text to responses.

    A reward model with length bias will score the padded version higher;
    an ideal reward model should score it the same or lower (padding adds nothing).

    Parameters
    ----------
    model : GPT2RewardModel (BT or Rubric)
    prompts, responses : matched lists
    filler : paragraph to append; defaults to the canonical verbose-bias probe

    Returns
    -------
    float: mean(score_padded - score_original)
      Positive → length bias present; near-zero or negative → robust
    """
    from src.data.rubric_preferences import FILLER_PARAGRAPH

    if filler is None:
        filler = FILLER_PARAGRAPH

    model.eval()
    deltas = []

    with torch.no_grad():
        for prompt, response in zip(prompts, responses):
            base_score   = _score_one(model, tokenizer, prompt, response,        device, max_length)
            padded_score = _score_one(model, tokenizer, prompt, response + filler, device, max_length)
            deltas.append(padded_score - base_score)

    return sum(deltas) / len(deltas) if deltas else 0.0


def _score_one(model, tokenizer, prompt: str, response: str, device, max_length: int) -> float:
    text = f"Human: {prompt}\n\nAssistant: {response}"
    enc  = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    reward = model(
        enc["input_ids"].to(device),
        enc["attention_mask"].to(device),
    ).rewards
    return reward.item()


# ── Joint comparison utility ───────────────────────────────────────────────────

def compare_rubric_vs_bt(
    rubric_model,
    bt_model_path: str,
    tokenizer,
    test_pairs: List[Dict],
    device: torch.device,
    ood_pairs: Optional[List[Dict]] = None,
) -> Dict:
    """Compute pairwise accuracy, length bias, and OOD calibration for both RMs.

    Parameters
    ----------
    rubric_model : trained GPT2RewardModel (rubric MSE)
    bt_model_path : path to trained BT reward model checkpoint
    test_pairs : list of {"prompt", "chosen", "rejected"} from hh-rlhf test set
    ood_pairs  : optional list of pairs from a different domain for OOD eval

    Returns
    -------
    dict with keys: bt_pairwise_acc, rubric_pairwise_acc,
                    bt_length_bias, rubric_length_bias,
                    bt_ood_spearman, rubric_ood_spearman (if ood_pairs provided)
    """
    from src.models.reward_model import GPT2RewardModel
    from src.data.rubric_preferences import PROBE_PROMPTS, PROBE_RESPONSES

    bt_model = GPT2RewardModel.from_pretrained(bt_model_path).to(device)
    bt_model.eval()
    rubric_model.eval()

    # ── Pairwise accuracy ──────────────────────────────────────────────────────
    bt_correct, rubric_correct, n = 0, 0, 0
    for pair in test_pairs:
        prompt   = pair.get("prompt", "")
        chosen   = pair.get("chosen", "")
        rejected = pair.get("rejected", "")

        bt_chosen   = _score_one(bt_model,     tokenizer, prompt, chosen,   device, 512)
        bt_rejected = _score_one(bt_model,     tokenizer, prompt, rejected, device, 512)
        r_chosen    = _score_one(rubric_model, tokenizer, prompt, chosen,   device, 512)
        r_rejected  = _score_one(rubric_model, tokenizer, prompt, rejected, device, 512)

        if bt_chosen > bt_rejected:
            bt_correct += 1
        if r_chosen > r_rejected:
            rubric_correct += 1
        n += 1

    bt_pairwise_acc     = bt_correct     / n if n else 0.0
    rubric_pairwise_acc = rubric_correct / n if n else 0.0

    # ── Length bias ────────────────────────────────────────────────────────────
    bt_bias     = evaluate_length_bias(bt_model,     tokenizer, PROBE_PROMPTS, PROBE_RESPONSES, device)
    rubric_bias = evaluate_length_bias(rubric_model, tokenizer, PROBE_PROMPTS, PROBE_RESPONSES, device)

    result: Dict = {
        "bt_pairwise_acc":     bt_pairwise_acc,
        "rubric_pairwise_acc": rubric_pairwise_acc,
        "bt_length_bias":      bt_bias,
        "rubric_length_bias":  rubric_bias,
    }

    # ── OOD Spearman correlation ───────────────────────────────────────────────
    if ood_pairs:
        try:
            from scipy.stats import spearmanr
            bt_scores, rubric_scores, human_ratings = [], [], []
            for pair in ood_pairs:
                prompt   = pair.get("prompt", "")
                response = pair.get("response", "")
                human    = float(pair.get("human_rating", 0.5))
                bt_scores.append(_score_one(bt_model,     tokenizer, prompt, response, device, 512))
                rubric_scores.append(_score_one(rubric_model, tokenizer, prompt, response, device, 512))
                human_ratings.append(human)
            result["bt_ood_spearman"]     = spearmanr(bt_scores,     human_ratings).correlation
            result["rubric_ood_spearman"] = spearmanr(rubric_scores, human_ratings).correlation
        except ImportError:
            pass

    return result
