"""Shared preprocessing utilities for SFT, reward modeling, PPO, DPO, and GRPO."""

from __future__ import annotations

import torch
import re
from dataclasses import dataclass
from typing import Optional
from torch.utils.data import DataLoader, Dataset


_ASSISTANT_SPLIT_RE = re.compile(r"(.*?)(?:\n\n)?Assistant:\s*(.*)", re.DOTALL)


def extract_prompt_and_response(transcript: str) -> tuple[str, str]:
    """Split an HH-RLHF transcript into prompt and preferred response."""
    match = _ASSISTANT_SPLIT_RE.search(transcript)
    if not match:
        return transcript.strip(), ""
    prompt = match.group(1).strip()
    response = match.group(2).strip()
    if not prompt.endswith("Assistant:"):
        prompt = f"{prompt}\n\nAssistant:"
    return prompt, response


def _load_hh_split(split: str, num_samples: Optional[int] = None):
    from datasets import load_dataset

    dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split=split)
    if num_samples and num_samples > 0:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    return dataset


class SFTDataset(Dataset):
    """Tokenized chosen responses for supervised fine-tuning."""

    def __init__(
        self,
        split: str,
        tokenizer,
        max_length: int = 512,
        num_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rows = list(_load_hh_split(split, num_samples))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.rows[idx]
        prompt, response = extract_prompt_and_response(example["chosen"])
        full_text = f"{prompt} {response}".strip()

        full = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        prompt_only = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )

        input_ids = full["input_ids"].squeeze(0)
        attention_mask = full["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        prompt_len = min(prompt_only["input_ids"].size(1), labels.size(0))
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class DPODataset(Dataset):
    """Prompt/chosen/rejected triples for offline preference optimization."""

    def __init__(
        self,
        split: str,
        tokenizer=None,
        max_length: int = 512,
        num_samples: Optional[int] = None,
    ):
        _ = tokenizer
        _ = max_length
        self.rows = list(_load_hh_split(split, num_samples))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, str]:
        example = self.rows[idx]
        prompt, chosen = extract_prompt_and_response(example["chosen"])
        _, rejected = extract_prompt_and_response(example["rejected"])
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }


@dataclass
class _PreferenceRow:
    chosen_input_ids: torch.Tensor
    chosen_attention_mask: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_attention_mask: torch.Tensor


class _PreferenceDataset(Dataset):
    def __init__(
        self,
        split: str,
        tokenizer,
        max_length: int = 512,
        num_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rows = list(_load_hh_split(split, num_samples))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> _PreferenceRow:
        example = self.rows[idx]
        chosen = self.tokenizer(
            example["chosen"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        rejected = self.tokenizer(
            example["rejected"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return _PreferenceRow(
            chosen_input_ids=chosen["input_ids"].squeeze(0),
            chosen_attention_mask=chosen["attention_mask"].squeeze(0),
            rejected_input_ids=rejected["input_ids"].squeeze(0),
            rejected_attention_mask=rejected["attention_mask"].squeeze(0),
        )


def build_preference_dataloader(
    split: str,
    tokenizer,
    batch_size: int,
    max_length: int = 512,
    num_samples: Optional[int] = None,
    shuffle: bool = True,
) -> DataLoader:
    """Build batched chosen/rejected tensors for reward model training."""
    dataset = _PreferenceDataset(split, tokenizer, max_length=max_length, num_samples=num_samples)

    def collate_fn(batch: list[_PreferenceRow]) -> dict[str, torch.Tensor]:
        return {
            "chosen_input_ids": torch.stack([row.chosen_input_ids for row in batch]),
            "chosen_attention_mask": torch.stack([row.chosen_attention_mask for row in batch]),
            "rejected_input_ids": torch.stack([row.rejected_input_ids for row in batch]),
            "rejected_attention_mask": torch.stack([row.rejected_attention_mask for row in batch]),
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
