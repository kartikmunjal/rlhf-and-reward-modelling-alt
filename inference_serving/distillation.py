"""On-policy vocabulary-compatible GPT-2-medium to GPT-2-small distillation."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path


def seed_all(seed: int) -> None:
    random.seed(seed)
    import numpy as np
    import torch
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def canonical_config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def distillation_loss(student_logits, teacher_logits, labels, *, temperature: float, kl_weight: float):
    import torch
    import torch.nn.functional as F
    student = student_logits[:, :-1].float()
    teacher = teacher_logits[:, :-1].float()
    targets = labels[:, 1:]
    mask = targets.ne(-100)
    hard = F.cross_entropy(student.reshape(-1, student.shape[-1]), targets.reshape(-1), ignore_index=-100)
    token_kl = F.kl_div(
        F.log_softmax(student / temperature, dim=-1),
        F.softmax(teacher / temperature, dim=-1),
        reduction="none",
    ).sum(-1) * (temperature ** 2)
    soft = token_kl.masked_select(mask).mean()
    total = kl_weight * soft + (1 - kl_weight) * hard
    return total, hard, soft


class TokenDataset:
    def __init__(self, parquet_path: Path, tokenizer, *, limit: int, seed: int, namespace: str, max_tokens: int,
                 exclude_ids: set[str] | None = None):
        import pandas as pd
        frame = pd.read_parquet(parquet_path, columns=["article_id", "source", "reference"])
        frame["rank"] = frame["article_id"].astype(str).map(
            lambda value: hashlib.sha256(f"{seed}\0{namespace}\0{value}".encode()).hexdigest()
        )
        if exclude_ids:
            frame = frame[~frame["article_id"].astype(str).isin(exclude_ids)]
        self.rows = frame.sort_values("rank").head(limit).to_dict("records")
        if len(self.rows) != limit:
            raise ValueError(f"Requested {limit} rows but only selected {len(self.rows)}")
        self.tokenizer, self.max_tokens = tokenizer, max_tokens

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        text = f"Article:\n{row['source']}\n\nSummary:\n{row['reference']}"
        encoded = self.tokenizer(text, truncation=True, max_length=self.max_tokens, add_special_tokens=False)
        return {"input_ids": encoded["input_ids"], "article_id": str(row["article_id"])}


def collate(batch, pad_id: int):
    import torch
    width = max(len(row["input_ids"]) for row in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for index, row in enumerate(batch):
        values = torch.tensor(row["input_ids"], dtype=torch.long)
        ids[index, :len(values)] = values; mask[index, :len(values)] = 1
    labels = ids.clone(); labels[mask == 0] = -100
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def evaluate(student, teacher, loader, config: dict) -> dict:
    import torch
    totals = {"loss": 0.0, "hard_nll": 0.0, "token_kl": 0.0}; count = 0
    student.eval(); teacher.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            teacher_logits = teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            student_logits = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            loss, hard, soft = distillation_loss(student_logits, teacher_logits, batch["labels"],
                temperature=config["draft_training"]["temperature"], kl_weight=config["draft_training"]["kl_weight"])
            for key, value in (("loss", loss), ("hard_nll", hard), ("token_kl", soft)):
                totals[key] += float(value)
            count += 1
    return {key: value / count for key, value in totals.items()} | {"batches": count}


def train_draft(config: dict, train_parquet: Path, output_dir: Path, *, resume: bool = True) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from inference_serving.data import verify_tokenizer_identity

    seed_all(config["seed"]); settings = config["draft_training"]
    teacher_info, student_info = config["artifacts"]["base"], config["artifacts"]["draft_initialization"]
    teacher_tok = AutoTokenizer.from_pretrained(teacher_info["model"], revision=teacher_info["revision"])
    student_tok = AutoTokenizer.from_pretrained(student_info["model"], revision=student_info["revision"])
    tokenizer_check = verify_tokenizer_identity({"teacher": teacher_tok, "student": student_tok})
    teacher_tok.pad_token = teacher_tok.eos_token; student_tok.pad_token = student_tok.eos_token
    teacher = AutoModelForCausalLM.from_pretrained(teacher_info["model"], revision=teacher_info["revision"], torch_dtype=torch.float16).cuda().eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    student = AutoModelForCausalLM.from_pretrained(student_info["model"], revision=student_info["revision"], torch_dtype=torch.float16).cuda()
    student.gradient_checkpointing_enable(); student.config.use_cache = False

    valid = TokenDataset(train_parquet, student_tok, limit=settings["validation_examples"], seed=config["seed"], namespace="draft-valid", max_tokens=settings["max_tokens"])
    valid_ids = {row["article_id"] for row in valid.rows}
    train = TokenDataset(train_parquet, student_tok, limit=settings["train_examples"], seed=config["seed"], namespace="draft-train", max_tokens=settings["max_tokens"], exclude_ids=valid_ids)
    train_ids = {row["article_id"] for row in train.rows}
    if train_ids & valid_ids:
        raise ValueError("Draft train/validation overlap")
    # The cryptographic rank already randomizes selection. Fixed order makes
    # checkpoint resume byte-for-byte deterministic instead of reshuffling and
    # silently skipping different examples.
    make_loader = lambda dataset: DataLoader(dataset, batch_size=settings["per_device_batch_size"], shuffle=False,
        collate_fn=lambda rows: collate(rows, student_tok.pad_token_id), num_workers=0)
    train_loader, valid_loader = make_loader(train), make_loader(valid)
    optimizer = torch.optim.AdamW(student.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    accumulation = settings["gradient_accumulation_steps"]
    total_steps = math.ceil(len(train_loader) * settings["epochs"] / accumulation)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * settings["warmup_ratio"]), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=settings["fp16"])
    output_dir.mkdir(parents=True, exist_ok=True); state_path = output_dir / "training_state.pt"
    global_step, seen_batches, baseline = 0, 0, None
    if resume and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        student.load_state_dict(state["student"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        global_step, seen_batches, baseline = state["global_step"], state["seen_batches"], state["baseline"]
    if baseline is None:
        baseline = evaluate(student, teacher, valid_loader, config)
    started = time.time(); torch.cuda.reset_peak_memory_stats(); student.train(); optimizer.zero_grad(set_to_none=True)
    for epoch in range(settings["epochs"]):
        for batch_index, batch in enumerate(train_loader):
            if batch_index < seen_batches: continue
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            with torch.inference_mode(), torch.autocast("cuda", enabled=settings["fp16"]):
                teacher_logits = teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            with torch.autocast("cuda", enabled=settings["fp16"]):
                student_logits = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                loss, _, _ = distillation_loss(student_logits, teacher_logits, batch["labels"],
                    temperature=settings["temperature"], kl_weight=settings["kl_weight"])
                loss = loss / accumulation
            scaler.scale(loss).backward(); seen_batches = batch_index + 1
            if seen_batches % accumulation == 0 or seen_batches == len(train_loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True); global_step += 1
                if global_step % settings["checkpoint_steps"] == 0:
                    torch.save({"student": student.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "global_step": global_step, "seen_batches": seen_batches, "baseline": baseline}, state_path)
    final = evaluate(student, teacher, valid_loader, config)
    success = final["token_kl"] < baseline["token_kl"] and final["hard_nll"] < baseline["hard_nll"]
    student.config.use_cache = True; student.save_pretrained(output_dir / "model", safe_serialization=True); student_tok.save_pretrained(output_dir / "model")
    manifest = {"stage": "draft_distillation", "success": success, "seed": config["seed"], "config_sha256": canonical_config_hash(config),
        "teacher": teacher_info, "student_initialization": student_info, "train_examples": len(train), "validation_examples": len(valid),
        "global_steps": global_step, "baseline": baseline, "final": final, "tokenizer_check": tokenizer_check,
        "runtime_seconds": time.time() - started, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
