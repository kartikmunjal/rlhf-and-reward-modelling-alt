#!/usr/bin/env python3
"""Run exactly one preregistered safety-classifier v2 trial."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.safety.data import (  # noqa: E402
    EncodedTextDataset,
    MultiLabelDataCollator,
    label_matrix,
    load_jigsaw_csv,
)
from src.safety.losses import multilabel_loss, prevalence_positive_weights  # noqa: E402
from src.safety.taxonomy import TARGET_LABELS  # noqa: E402
from src.safety.v2_data import (  # noqa: E402
    combine_v2_training,
    v2_calibration_role,
    verify_manifested_file,
)


class V2Trainer(Trainer):
    def __init__(self, *args, loss_specification, positive_weights, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_specification = loss_specification
        self.positive_weights = positive_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = multilabel_loss(
            outputs.logits,
            labels,
            self.loss_specification,
            self.positive_weights.to(outputs.logits.device),
        )
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--config", default="configs/safety_v2.yaml")
    parser.add_argument("--ledger", default="configs/safety_v2_trial_ledger.json")
    parser.add_argument("--jigsaw-csv", default="data/raw/jigsaw/train.csv")
    parser.add_argument("--data-dir", default="data/processed/safety_v2")
    parser.add_argument("--output-root", default="checkpoints/safety_v2")
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-eval-examples", type=int)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subset(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None or limit >= len(frame):
        return frame
    return frame.sample(n=limit, random_state=seed).sort_values("id")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ledger = json.loads(Path(args.ledger).read_text(encoding="utf-8"))
    matches = [trial for trial in ledger["trials"] if trial["trial_id"] == args.trial_id]
    if len(matches) != 1:
        raise ValueError(f"Trial id is not uniquely preregistered: {args.trial_id}")
    trial = matches[0]
    loss_specs = {item["name"]: item for item in config["losses"]}
    if trial["loss"] not in loss_specs or trial["seed"] not in config["seeds"]:
        raise ValueError("Trial does not match locked v2 config")
    loss_spec = loss_specs[trial["loss"]]
    seed = int(trial["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_dir = Path(args.data_dir)
    data_manifest = json.loads((data_dir / "data_manifest.json").read_text(encoding="utf-8"))
    beaver_path = data_dir / "beavertails_train.parquet"
    verify_manifested_file(beaver_path, data_manifest, "beavertails_train")
    beaver = pd.read_parquet(beaver_path)
    beaver["labels"] = [np.asarray(value, dtype=np.float32) for value in beaver["labels"]]

    jigsaw = load_jigsaw_csv(args.jigsaw_csv, seed=2025)
    train = combine_v2_training(jigsaw[jigsaw["split"] == "train"], beaver)
    calibration_source = jigsaw[jigsaw["split"] == "calibration"].copy()
    calibration_source["v2_role"] = [
        v2_calibration_role(str(identifier)) for identifier in calibration_source["id"]
    ]
    calibration_source = calibration_source[
        calibration_source["v2_role"] == "threshold_calibration"
    ]
    calibration = pd.DataFrame(
        {
            "id": "jigsaw:" + calibration_source["id"].astype(str),
            "text": calibration_source["comment_text"],
            "labels": calibration_source["labels"],
            "source": "jigsaw",
        }
    )
    train = subset(train, args.max_train_examples, seed)
    calibration = subset(calibration, args.max_eval_examples, seed)

    train_labels = torch.tensor(label_matrix(train["labels"]), dtype=torch.float32)
    cap = loss_spec.get("positive_weight_cap")
    positive_weights = prevalence_positive_weights(
        train_labels, float(cap) if cap is not None else None
    )

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    base = AutoModelForSequenceClassification.from_pretrained(
        config["base_model"],
        num_labels=len(TARGET_LABELS),
        problem_type="multi_label_classification",
        id2label=dict(enumerate(TARGET_LABELS)),
        label2id={name: index for index, name in enumerate(TARGET_LABELS)},
    )
    train_cfg = config["train"]
    model = get_peft_model(
        base,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=int(train_cfg["lora_r"]),
            lora_alpha=int(train_cfg["lora_alpha"]),
            lora_dropout=float(train_cfg["lora_dropout"]),
            target_modules=list(train_cfg["target_modules"]),
            bias="none",
        ),
    )
    output_dir = Path(args.output_root) / args.trial_id
    output_dir.mkdir(parents=True, exist_ok=True)
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
        save_total_limit=1,
        logging_steps=100,
        fp16=bool(train_cfg["fp16"]) and torch.cuda.is_available() and not args.cpu,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=seed,
        data_seed=seed,
        use_cpu=args.cpu,
    )
    trainer = V2Trainer(
        model=model,
        args=training_args,
        train_dataset=EncodedTextDataset(
            train["text"].tolist(), train_labels.numpy(), tokenizer, int(config["max_length"])
        ),
        eval_dataset=EncodedTextDataset(
            calibration["text"].tolist(),
            label_matrix(calibration["labels"]),
            tokenizer,
            int(config["max_length"]),
        ),
        data_collator=MultiLabelDataCollator(tokenizer),
        processing_class=tokenizer,
        loss_specification=loss_spec,
        positive_weights=positive_weights,
    )
    model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    manifest = {
        "family": ledger["family"],
        "trial": trial,
        "planned_n_trials": int(ledger["planned_n_trials"]),
        "loss_specification": loss_spec,
        "taxonomy_version": config["taxonomy_version"],
        "jigsaw_sha256": file_sha256(Path(args.jigsaw_csv)),
        "data_manifest_sha256": file_sha256(data_dir / "data_manifest.json"),
        "n_train": int(len(train)),
        "n_calibration": int(len(calibration)),
        "train_source_counts": train["source"].value_counts().sort_index().to_dict(),
        "positive_counts": dict(
            zip(TARGET_LABELS, train_labels.sum(dim=0).int().tolist())
        ),
        "positive_weights": dict(zip(TARGET_LABELS, positive_weights.tolist())),
        "full_run": args.max_train_examples is None and args.max_eval_examples is None,
        "device": str(trainer.args.device),
        "config": config,
    }
    with (output_dir / "run_manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
