#!/usr/bin/env python3
"""Train the frozen SFT adapter and K=3 reward ensemble, resumably.

Every completed artifact has a manifest; an existing valid manifest is the only
condition under which a phase is skipped.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, random, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recursive_self_improvement.config import load_effective_config
from src.models.reward_model import GPT2RewardModel


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def seed_all(seed, torch):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def lora_config(config, *, reward=False):
    from peft import LoraConfig, TaskType
    p = config["peft"]
    kwargs = dict(r=p["rank"], lora_alpha=p["alpha"], lora_dropout=p["dropout"], target_modules=p["target_modules"], bias=p["bias"])
    if reward: kwargs["modules_to_save"] = ["reward_head"]
    else: kwargs["task_type"] = TaskType.CAUSAL_LM
    return LoraConfig(**kwargs)


def encode(tokenizer, texts, maximum, device):
    return tokenizer(texts, padding=True, truncation=True, max_length=maximum, return_tensors="pt").to(device)


def save_manifest(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def valid_complete(path):
    if not path.exists(): return False
    try: return json.loads(path.read_text(encoding="utf-8"))["status"] == "complete"
    except Exception: return False


def train_sft(config, tokenizer, device, torch, output_root, data_root):
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM
    target = output_root / "sft"; manifest = target / "run_manifest.json"
    if valid_complete(manifest): print("SFT already complete"); return
    rows = read_jsonl(data_root / "sft_train.jsonl"); seed = config["seed"]; seed_all(seed, torch)
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(config["model"]["base"], revision=config["model"]["base_revision"]), lora_config(config)).to(device)
    model.config.use_cache = False; model.train(); optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config["sft"]["learning_rate"], weight_decay=config["sft"]["weight_decay"])
    accumulation=config["sft"]["gradient_accumulation_steps"]; order=list(range(len(rows))); random.Random(seed).shuffle(order)
    start=time.perf_counter(); losses=[]; tokens=0; optimizer.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats()
    for number, index in enumerate(order, 1):
        row=rows[index]; batch=encode(tokenizer,[row["prompt"]+row["chosen"]],config["model"]["context_length"],device); tokens += int(batch["attention_mask"].sum())
        with torch.autocast("cuda", dtype=torch.float16): loss=model(**batch,labels=batch["input_ids"]).loss / accumulation
        loss.backward(); losses.append(float(loss.detach())*accumulation)
        if number % accumulation == 0 or number == len(order): optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if number % 250 == 0: print(json.dumps({"stage":"sft","examples":number,"loss_mean_last250":sum(losses[-250:])/len(losses[-250:])}), flush=True)
    target.mkdir(parents=True, exist_ok=True); model.save_pretrained(target); tokenizer.save_pretrained(target)
    save_manifest(manifest,{"status":"complete","stage":"sft","git_commit":git_commit(),"seed":seed,"examples":len(rows),"optimizer_steps":math.ceil(len(rows)/accumulation),"non_padding_training_tokens":tokens,"loss_mean":sum(losses)/len(losses),"runtime_seconds":time.perf_counter()-start,"peak_allocated_gpu_memory_bytes":torch.cuda.max_memory_allocated(),"source_sha256":sha256(data_root/"sft_train.jsonl")})
    del model, optimizer; torch.cuda.empty_cache()


def reward_accuracy(model, tokenizer, rows, maximum, device, torch):
    model.eval(); correct=0; margins=[]
    with torch.no_grad():
        for row in rows:
            chosen=encode(tokenizer,[row["prompt"]+row["chosen"]],maximum,device); rejected=encode(tokenizer,[row["prompt"]+row["rejected"]],maximum,device)
            margin=float((model(**chosen).rewards-model(**rejected).rewards).item()); margins.append(margin); correct += margin > 0
    return correct/len(rows), margins


def train_reward(config, tokenizer, device, torch, output_root, data_root, seed):
    from peft import PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM
    target=output_root/"reward_ensemble"/f"seed{seed}"; manifest=target/"run_manifest.json"
    if valid_complete(manifest): print(f"Reward seed {seed} already complete"); return
    train=read_jsonl(data_root/"reward_train.jsonl"); validation=read_jsonl(data_root/"reward_validation.jsonl"); seed_all(seed,torch)
    # Standard RLHF ordering: initialize each RM backbone from the frozen SFT
    # policy, then attach a new LoRA adapter and independently seeded head.
    sft_base=AutoModelForCausalLM.from_pretrained(config["model"]["base"],revision=config["model"]["base_revision"])
    sft_merged=PeftModel.from_pretrained(sft_base,output_root/"sft").merge_and_unload()
    reward_base=GPT2RewardModel(sft_merged.config)
    reward_base.transformer.load_state_dict(sft_merged.transformer.state_dict())
    del sft_base,sft_merged
    model=get_peft_model(reward_base,lora_config(config,reward=True)).to(device); model.train()
    protocol=config["reward_execution"]; accumulation=protocol["gradient_accumulation_steps"]; epochs=protocol["epochs"]
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=protocol["learning_rate"],weight_decay=protocol["weight_decay"])
    total_steps=math.ceil(len(train)*epochs/accumulation); warmup_steps=int(total_steps*protocol["warmup_ratio"])
    from transformers import get_cosine_schedule_with_warmup
    scheduler=get_cosine_schedule_with_warmup(optimizer,warmup_steps,total_steps)
    scaler=torch.cuda.amp.GradScaler(enabled=protocol["fp16_gradient_scaler"])
    start=time.perf_counter(); losses=[]; tokens=0; optimizer.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats(); seen=0; optimizer_steps=0
    for epoch in range(epochs):
        order=list(range(len(train))); random.Random(seed+epoch).shuffle(order); model.train()
        for index in order:
            seen += 1; row=train[index]; chosen=encode(tokenizer,[row["prompt"]+row["chosen"]],config["model"]["context_length"],device); rejected=encode(tokenizer,[row["prompt"]+row["rejected"]],config["model"]["context_length"],device); tokens += int(chosen["attention_mask"].sum()+rejected["attention_mask"].sum())
            with torch.autocast("cuda",dtype=torch.float16): loss=-torch.nn.functional.logsigmoid(model(**chosen).rewards-model(**rejected).rewards).mean()/accumulation
            scaler.scale(loss).backward(); losses.append(float(loss.detach())*accumulation)
            if seen%accumulation==0 or seen==len(train)*epochs:
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),protocol["gradient_clip_norm"])
                scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True); optimizer_steps += 1
            if seen%250==0: print(json.dumps({"stage":"reward","seed":seed,"epoch":epoch+1,"examples_seen":seen,"loss_mean_last250":sum(losses[-250:])/len(losses[-250:]),"learning_rate":scheduler.get_last_lr()[0]}),flush=True)
    accuracy,margins=reward_accuracy(model,tokenizer,validation,config["model"]["context_length"],device,torch); target.mkdir(parents=True,exist_ok=True); model.save_pretrained(target); tokenizer.save_pretrained(target)
    save_manifest(target/"validation_predictions.json",{"prompt_ids":[r["prompt_id"] for r in validation],"margins":margins})
    status="complete" if accuracy>=config["reward_ensemble"]["minimum_pairwise_accuracy"] else "failed_validation_gate"
    save_manifest(manifest,{"status":status,"stage":"reward","protocol":"established_reward_config_corrected_attempt","git_commit":git_commit(),"seed":seed,"epochs":epochs,"examples":len(train),"examples_seen":seen,"validation_examples":len(validation),"validation_pairwise_accuracy":accuracy,"minimum_pairwise_accuracy":config["reward_ensemble"]["minimum_pairwise_accuracy"],"gate_scope":config["reward_ensemble"]["gate_scope"],"optimizer_steps":optimizer_steps,"warmup_steps":warmup_steps,"non_padding_training_tokens":tokens,"loss_mean":sum(losses)/len(losses),"runtime_seconds":time.perf_counter()-start,"peak_allocated_gpu_memory_bytes":torch.cuda.max_memory_allocated(),"train_sha256":sha256(data_root/"reward_train.jsonl"),"validation_sha256":sha256(data_root/"reward_validation.jsonl")})
    if status != "complete": raise RuntimeError(f"Reward seed {seed} validation accuracy {accuracy:.4f} failed locked gate")
    del model,optimizer; torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--phase",choices=["all","sft","reward"],default="all"); args=parser.parse_args()
    import torch
    from transformers import AutoTokenizer
    if not torch.cuda.is_available(): raise SystemExit("CUDA is required")
    config=load_effective_config(ROOT); tokenizer=AutoTokenizer.from_pretrained(config["model"]["base"],revision=config["model"]["base_revision"]); tokenizer.pad_token=tokenizer.eos_token; tokenizer.padding_side="right"
    data_root=ROOT/"data/processed/recursive_self_improvement_v1"; output_root=ROOT/"checkpoints/recursive_self_improvement_v1"; device=torch.device("cuda")
    if args.phase in ("all","sft"): train_sft(config,tokenizer,device,torch,output_root,data_root)
    if args.phase in ("all","reward"):
        for seed in config["reward_ensemble"]["member_seeds"]: train_reward(config,tokenizer,device,torch,output_root,data_root,seed)

if __name__=="__main__": main()
