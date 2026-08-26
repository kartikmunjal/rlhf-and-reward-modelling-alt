"""RTX-3070-compatible LoRA SFT and DPO for summarization."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path

from summarization_finetune.data import SummarizationSFTDataset, causal_lm_collator, read_jsonl


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch
        np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def config_sha256(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def hardware_manifest(torch) -> dict:
    return {"torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}


def train_sft(config: dict, train_parquet: Path, output_dir: Path, *, resume: bool = True) -> dict:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    seed_everything(config["seed"])
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base"])
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["model"]["base"])
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if config["sft"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=config["model"]["lora_rank"],
                      lora_alpha=config["model"]["lora_alpha"], lora_dropout=config["model"]["lora_dropout"],
                      target_modules=config["model"]["lora_targets"], bias="none")
    model = get_peft_model(model, lora)
    dataset = SummarizationSFTDataset(train_parquet, tokenizer, config)
    args = TrainingArguments(
        output_dir=str(output_dir), num_train_epochs=config["data"]["sft_epochs"],
        per_device_train_batch_size=config["sft"]["per_device_batch_size"],
        gradient_accumulation_steps=config["sft"]["gradient_accumulation_steps"],
        learning_rate=config["sft"]["learning_rate"], weight_decay=config["sft"]["weight_decay"],
        warmup_ratio=config["sft"]["warmup_ratio"], lr_scheduler_type=config["sft"]["scheduler"],
        fp16=config["sft"]["fp16"], gradient_checkpointing=config["sft"]["gradient_checkpointing"],
        save_strategy="steps", save_steps=1000, save_total_limit=2, logging_steps=50,
        report_to="none", remove_unused_columns=False, seed=config["seed"], data_seed=config["seed"],
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=causal_lm_collator(tokenizer))
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1]))
    result = trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if resume and checkpoints else None)
    adapter_dir, merged_dir = output_dir / "adapter", output_dir / "merged"
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    model.save_pretrained(adapter_dir); tokenizer.save_pretrained(adapter_dir)
    merged = model.merge_and_unload(); merged.config.use_cache = True
    merged.save_pretrained(merged_dir, safe_serialization=True); tokenizer.save_pretrained(merged_dir)
    manifest = {"stage": "sft", "seed": config["seed"], "config_sha256": config_sha256(config),
                "base_model": config["model"]["base"],
                "train_examples": len(dataset), "epochs": config["data"]["sft_epochs"],
                "trainable_parameters": trainable, "train_loss": result.training_loss,
                "global_steps": result.global_step, "trainer_metrics": result.metrics,
                "adapter_dir": str(adapter_dir), "merged_dir": str(merged_dir), "hardware": hardware_manifest(torch)}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _encode_dpo_row(row: dict, tokenizer, config: dict) -> tuple[list[int], list[int], int]:
    from summarization_finetune.generation import prompt_ids
    prompt = prompt_ids(tokenizer, row["source"], config["model"]["source_tokens"])
    chosen = tokenizer.encode(row["chosen"], add_special_tokens=False)[:config["model"]["new_summary_tokens"]] + [tokenizer.eos_token_id]
    rejected = tokenizer.encode(row["rejected"], add_special_tokens=False)[:config["model"]["new_summary_tokens"]] + [tokenizer.eos_token_id]
    return prompt + chosen, prompt + rejected, len(prompt)


def _sequence_logprob(model, sequences, prompt_lengths, pad_id):
    import torch
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=model.device)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for index, (sequence, prompt_len) in enumerate(zip(sequences, prompt_lengths)):
        ids[index, :len(sequence)] = torch.tensor(sequence, device=model.device)
        mask[index, :len(sequence)] = 1
        labels[index, prompt_len:len(sequence)] = ids[index, prompt_len:len(sequence)]
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1]
    targets = labels[:, 1:]
    valid = targets != -100
    token_logp = torch.log_softmax(logits.float(), dim=-1).gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (token_logp * valid).sum(-1)


def train_dpo(config: dict, preference_path: Path, sft_merged_dir: Path, output_dir: Path) -> dict:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    seed_everything(config["seed"])
    started = time.time()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    rows = read_jsonl(preference_path)
    if not rows:
        raise ValueError("No unambiguous DPO preferences")
    tokenizer = AutoTokenizer.from_pretrained(sft_merged_dir); tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if config["dpo"]["fp16"] else torch.float32
    reference = AutoModelForCausalLM.from_pretrained(sft_merged_dir, torch_dtype=dtype).cuda().eval()
    for parameter in reference.parameters(): parameter.requires_grad_(False)
    base = AutoModelForCausalLM.from_pretrained(sft_merged_dir, torch_dtype=dtype).cuda()
    base.config.use_cache = False
    if config["dpo"]["gradient_checkpointing"]: base.gradient_checkpointing_enable()
    policy = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=config["model"]["lora_rank"],
                             lora_alpha=config["model"]["lora_alpha"], lora_dropout=config["model"]["lora_dropout"],
                             target_modules=config["model"]["lora_targets"], bias="none"))
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["dpo"]["learning_rate"])
    accumulation = config["dpo"]["gradient_accumulation_steps"]
    steps = math.ceil(len(rows) * config["dpo"]["epochs"] / accumulation)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(steps * 0.1), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=config["dpo"]["fp16"])
    losses, global_step = [], 0
    policy.train(); optimizer.zero_grad(set_to_none=True)
    for epoch in range(config["dpo"]["epochs"]):
        order = list(range(len(rows))); random.Random(config["seed"] + epoch).shuffle(order)
        for position, row_index in enumerate(order):
            chosen, rejected, prompt_len = _encode_dpo_row(rows[row_index], tokenizer, config)
            with torch.no_grad():
                ref = _sequence_logprob(reference, [chosen, rejected], [prompt_len, prompt_len], tokenizer.pad_token_id)
            with torch.cuda.amp.autocast(enabled=config["dpo"]["fp16"]):
                pol = _sequence_logprob(policy, [chosen, rejected], [prompt_len, prompt_len], tokenizer.pad_token_id)
                logit = config["dpo"]["beta"] * ((pol[0] - pol[1]) - (ref[0] - ref[1]))
                loss = -torch.nn.functional.logsigmoid(logit) / accumulation
            scaler.scale(loss).backward(); losses.append(float(loss.detach()) * accumulation)
            if (position + 1) % accumulation == 0 or position + 1 == len(order):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                global_step += 1
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir, merged_dir = output_dir / "adapter", output_dir / "merged"
    policy.save_pretrained(adapter_dir); tokenizer.save_pretrained(adapter_dir)
    merged = policy.merge_and_unload(); merged.config.use_cache = True
    merged.save_pretrained(merged_dir, safe_serialization=True); tokenizer.save_pretrained(merged_dir)
    manifest = {"stage": "dpo", "seed": config["seed"], "config_sha256": config_sha256(config),
                "preference_rows": len(rows),
                "epochs": config["dpo"]["epochs"], "beta": config["dpo"]["beta"],
                "global_steps": global_step, "mean_loss": sum(losses) / len(losses),
                "runtime_seconds": time.time() - started, "pairs_per_second": len(rows) / (time.time() - started),
                "adapter_dir": str(adapter_dir), "merged_dir": str(merged_dir), "hardware": hardware_manifest(torch)}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
