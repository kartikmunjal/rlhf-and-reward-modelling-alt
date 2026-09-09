#!/usr/bin/env python3
"""Run the frozen RTX-3070 SFT/RM/DPO/rollout feasibility gate."""

from __future__ import annotations

import json, math, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from recursive_self_improvement.config import load_effective_config
from src.models.reward_model import GPT2RewardModel


def rows(path, limit):
    with path.open(encoding="utf-8") as handle: return [json.loads(line) for _, line in zip(range(limit), handle)]


def encode(tokenizer, texts, max_length, device):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)


def lora_config(config, *, reward=False):
    from peft import LoraConfig, TaskType
    peft = config["peft"]
    kwargs = dict(r=peft["rank"], lora_alpha=peft["alpha"], lora_dropout=peft["dropout"], target_modules=peft["target_modules"], bias=peft["bias"])
    if reward: kwargs["modules_to_save"] = ["reward_head"]
    else: kwargs["task_type"] = TaskType.CAUSAL_LM
    return LoraConfig(**kwargs)


def finite_gradients(model):
    return all(torch.isfinite(parameter.grad).all().item() for parameter in model.parameters() if parameter.grad is not None)


def response_logp(model, tokenizer, prompts, responses, max_length, device):
    full = [p + r for p, r in zip(prompts, responses)]
    batch = encode(tokenizer, full, max_length, device)
    prompt_lengths = [len(tokenizer(p, truncation=True, max_length=max_length)["input_ids"]) for p in prompts]
    logits = model(**batch).logits[:, :-1]
    labels = batch["input_ids"][:, 1:]
    token_logp = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    mask = batch["attention_mask"][:, 1:].bool()
    for index, length in enumerate(prompt_lengths): mask[index, :max(0, length - 1)] = False
    return (token_logp * mask).sum(1)


def run_stage(name, function, output):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
    try:
        details = function(); status = "pass"
        if not details.pop("finite", True): status = "fail_nonfinite"
    except Exception as error:
        status = "fail_exception"; details = {"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
    peak = torch.cuda.max_memory_allocated()
    output[name] = {"status": status, "runtime_seconds": time.perf_counter() - start, "peak_allocated_gpu_memory_bytes": peak, **details}


def main():
    global torch
    import torch
    from peft import PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    config = load_effective_config(ROOT); base = config["model"]["base"]; device = torch.device("cuda")
    if not torch.cuda.is_available(): raise SystemExit("CUDA required")
    tokenizer = AutoTokenizer.from_pretrained(base, revision=config["model"]["base_revision"]); tokenizer.pad_token = tokenizer.eos_token
    data = ROOT / "data/processed/recursive_self_improvement_v1"; smoke_root = ROOT / "checkpoints/recursive_self_improvement_v1/smoke"
    smoke_root.mkdir(parents=True, exist_ok=True); output = {"study_id": config["study_id"], "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "stages": {}}

    def sft():
        model = get_peft_model(AutoModelForCausalLM.from_pretrained(base, revision=config["model"]["base_revision"]), lora_config(config)).to(device); model.train()
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config["sft"]["learning_rate"])
        sample = rows(data / "sft_train.jsonl", config["smoke_gate"]["per_stage_examples"]["sft"]); losses=[]
        for row in sample[:config["smoke_gate"]["optimizer_steps"]]:
            batch=encode(tokenizer,[row["prompt"]+row["chosen"]],config["model"]["context_length"],device); loss=model(**batch,labels=batch["input_ids"]).loss
            loss.backward(); ok=finite_gradients(model); optimizer.step(); optimizer.zero_grad(); losses.append(float(loss));
            if not ok: return {"finite": False, "losses": losses}
        path=smoke_root/"sft"; model.save_pretrained(path); fresh=AutoModelForCausalLM.from_pretrained(base,revision=config["model"]["base_revision"]); PeftModel.from_pretrained(fresh,path)
        return {"finite": all(math.isfinite(v) for v in losses), "losses": losses, "adapter_reload": True}

    def reward():
        model=get_peft_model(GPT2RewardModel.from_pretrained_backbone(base),lora_config(config,reward=True)).to(device); model.train()
        optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-5); sample=rows(data/"reward_train.jsonl",config["smoke_gate"]["per_stage_examples"]["reward"]); losses=[]
        for row in sample[:config["smoke_gate"]["optimizer_steps"]]:
            chosen=encode(tokenizer,[row["prompt"]+row["chosen"]],config["model"]["context_length"],device); rejected=encode(tokenizer,[row["prompt"]+row["rejected"]],config["model"]["context_length"],device)
            loss=-torch.nn.functional.logsigmoid(model(**chosen).rewards-model(**rejected).rewards).mean(); loss.backward(); ok=finite_gradients(model); optimizer.step(); optimizer.zero_grad(); losses.append(float(loss))
            if not ok: return {"finite":False,"losses":losses}
        path=smoke_root/"reward"; model.save_pretrained(path); fresh=GPT2RewardModel.from_pretrained_backbone(base); PeftModel.from_pretrained(fresh,path)
        return {"finite":all(math.isfinite(v) for v in losses),"losses":losses,"adapter_reload":True}

    def dpo():
        policy=get_peft_model(AutoModelForCausalLM.from_pretrained(base,revision=config["model"]["base_revision"]),lora_config(config)).to(device); reference=AutoModelForCausalLM.from_pretrained(base,revision=config["model"]["base_revision"]).to(device).eval()
        for p in reference.parameters(): p.requires_grad_(False)
        optimizer=torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad),lr=config["ordinary_dpo"]["learning_rate"]); sample=rows(data/"reward_train.jsonl",config["smoke_gate"]["per_stage_examples"]["dpo"]); losses=[]
        for row in sample[:config["smoke_gate"]["optimizer_steps"]]:
            prompts=[row["prompt"]]; chosen=[row["chosen"]]; rejected=[row["rejected"]]
            pc=response_logp(policy,tokenizer,prompts,chosen,config["model"]["context_length"],device); pr=response_logp(policy,tokenizer,prompts,rejected,config["model"]["context_length"],device)
            with torch.no_grad(): rc=response_logp(reference,tokenizer,prompts,chosen,config["model"]["context_length"],device); rr=response_logp(reference,tokenizer,prompts,rejected,config["model"]["context_length"],device)
            loss=-torch.nn.functional.logsigmoid(config["ordinary_dpo"]["beta"]*((pc-pr)-(rc-rr))).mean(); loss.backward(); ok=finite_gradients(policy); optimizer.step(); optimizer.zero_grad(); losses.append(float(loss))
            if not ok: return {"finite":False,"losses":losses}
        path=smoke_root/"dpo"; policy.save_pretrained(path); fresh=AutoModelForCausalLM.from_pretrained(base,revision=config["model"]["base_revision"]); PeftModel.from_pretrained(fresh,path)
        return {"finite":all(math.isfinite(v) for v in losses),"losses":losses,"adapter_reload":True}

    def rollout():
        model=AutoModelForCausalLM.from_pretrained(base,revision=config["model"]["base_revision"]).to(device).eval(); sample=rows(data/"improvement.jsonl",config["smoke_gate"]["per_stage_examples"]["rollout_prompts"]); tokens=0
        with torch.no_grad():
            for row in sample:
                batch=encode(tokenizer,[row["prompt"]],256,device); generated=model.generate(**batch,do_sample=True,temperature=.8,top_p=.9,max_new_tokens=16,pad_token_id=tokenizer.eos_token_id); tokens += int(generated.shape[1]-batch["input_ids"].shape[1])
        return {"finite":True,"prompts":len(sample),"generated_tokens":tokens}

    for name, fn in (("sft",sft),("reward",reward),("dpo",dpo),("rollout",rollout)): run_stage(name,fn,output["stages"])
    limit=int(7.6*1024**3)
    for row in output["stages"].values():
        row["memory_gate_pass"] = row["peak_allocated_gpu_memory_bytes"] < limit
        if not row["memory_gate_pass"] and row["status"] == "pass": row["status"]="fail_memory_gate"
    output["overall_pass"] = all(row["status"] == "pass" for row in output["stages"].values())
    destination=ROOT/"results/recursive_self_improvement_v1/smoke_manifest.json"; destination.parent.mkdir(parents=True,exist_ok=True); destination.write_text(json.dumps(output,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(destination); print(json.dumps(output,indent=2))

if __name__ == "__main__": main()
