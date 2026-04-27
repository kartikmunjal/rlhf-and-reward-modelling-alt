"""Lazy exports for training entrypoints."""

from __future__ import annotations

from importlib import import_module

__all__ = ["train_sft", "train_reward_model", "train_ppo", "train_grpo", "dpo_loss", "train_dpo"]


def __getattr__(name: str):
    if name == "train_sft":
        return import_module(".sft", __name__).train_sft
    if name == "train_reward_model":
        return import_module(".reward", __name__).train_reward_model
    if name == "train_ppo":
        return import_module(".ppo", __name__).train_ppo
    if name == "train_grpo":
        return import_module(".grpo", __name__).train_grpo
    if name == "dpo_loss":
        return import_module(".dpo", __name__).dpo_loss
    if name == "train_dpo":
        return import_module(".dpo", __name__).train_dpo
    raise AttributeError(name)
