#!/usr/bin/env python3
"""Fine-tune a DistilBERT LoRA multi-label safety classifier on Jigsaw."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn import BCEWithLogitsLoss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.safety.data import EncodedTextDataset, label_matrix, load_jigsaw_csv
from src.safety.taxonomy import TARGET_LABELS, TAXONOMY_VERSION


class ImbalanceAwareTrainer(Trainer):
    """BCE trainer with train-split-only positive class weights."""

    def __init__(self, *args, positive_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.positive_weights = positive_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = BCEWithLogitsLoss(pos_weight=self.positive_weights.to(outputs.logits.device))(
            outputs.logits, labels
        )
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/safety_classifier.yaml")
    parser.add_argument("--jigsaw-csv", default="data/raw/jigsaw/train.csv")
    parser.add_argument("--output-dir", default="checkpoints/safety_classifier_v1")
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-eval-examples", type=int)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subset(frame, limit: int | None, seed: int):
    if limit is None or limit >= len(frame):
        return frame
    return frame.sample(n=limit, random_state=seed).sort_values("id")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if config["taxonomy_version"] != TAXONOMY_VERSION:
        raise ValueError("Config taxonomy version does not match code")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    source_path = Path(args.jigsaw_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_jigsaw_csv(source_path, seed)
    train = subset(frame[frame["split"] == "train"], args.max_train_examples, seed)
    calibration = subset(
        frame[frame["split"] == "calibration"], args.max_eval_examples, seed
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    base = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=len(TARGET_LABELS),
        problem_type="multi_label_classification",
        id2label=dict(enumerate(TARGET_LABELS)),
        label2id={name: index for index, name in enumerate(TARGET_LABELS)},
    )
    train_cfg = config["train"]
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(train_cfg["lora_r"]),
        lora_alpha=int(train_cfg["lora_alpha"]),
        lora_dropout=float(train_cfg["lora_dropout"]),
        target_modules=list(train_cfg["target_modules"]),
        bias="none",
    )
    model = get_peft_model(base, lora)

    y_train = label_matrix(train["labels"])
    positive = y_train.sum(axis=0)
    if np.any(positive == 0):
        raise ValueError("Every target needs at least one positive in the training subset")
    positive_weights = torch.tensor((len(y_train) - positive) / positive, dtype=torch.float32)

    train_dataset = EncodedTextDataset(
        train["comment_text"].tolist(), y_train, tokenizer, int(config["max_length"])
    )
    calibration_dataset = EncodedTextDataset(
        calibration["comment_text"].tolist(),
        label_matrix(calibration["labels"]),
        tokenizer,
        int(config["max_length"]),
    )
    use_fp16 = bool(train_cfg["fp16"]) and torch.cuda.is_available() and not args.cpu
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(train_cfg["epochs"]),
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        per_device_train_batch_size=int(train_cfg["train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        fp16=use_fp16,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=seed,
        data_seed=seed,
        use_cpu=args.cpu,
    )
    trainer = ImbalanceAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=calibration_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
        positive_weights=positive_weights,
    )
    model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    manifest = {
        "experiment_name": config["experiment_name"],
        "taxonomy_version": TAXONOMY_VERSION,
        "seed": seed,
        "source": str(source_path),
        "source_sha256": sha256_file(source_path),
        "split_counts_full": frame["split"].value_counts().sort_index().to_dict(),
        "n_train_used": len(train),
        "n_calibration_used": len(calibration),
        "positive_counts_train": dict(zip(TARGET_LABELS, positive.astype(int).tolist())),
        "positive_weights": dict(zip(TARGET_LABELS, positive_weights.tolist())),
        "max_train_examples": args.max_train_examples,
        "max_eval_examples": args.max_eval_examples,
        "full_run": args.max_train_examples is None and args.max_eval_examples is None,
        "device": str(trainer.args.device),
        "config": config,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
