#!/usr/bin/env python3
"""Run ordinary DPO and the preregistered eight-round rolling-2 branch.

The command is phase-resumable at completed checkpoint manifests. Generated
candidates, all three reward scores, preference pairs, and compute telemetry
are retained before each policy update.
"""
from __future__ import annotations

import hashlib, json, math, os, random, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from recursive_self_improvement.config import load_effective_config
from src.models.reward_model import GPT2RewardModel


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as h: return [json.loads(x) for x in h]
def write_jsonl(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w",encoding="utf-8",newline="\n") as h:
        for row in rows: h.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n")
    os.replace(tmp,path)
def save_json(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(".tmp"); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); os.replace(tmp,path)
def complete(path):
    try:return json.loads(Path(path).read_text(encoding="utf-8"))["status"]=="complete"
    except Exception:return False
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def commit():return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
def seed_all(seed,torch):random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def encode(tok,texts,maximum,device):return tok(texts,padding=True,truncation=True,max_length=maximum,return_tensors="pt").to(device)
def response_logp(model,tok,prompts,responses,maximum,device):
    batch=encode(tok,[p+r for p,r in zip(prompts,responses)],maximum,device)
    prompt_lengths=[len(tok(p,truncation=True,max_length=maximum)["input_ids"]) for p in prompts]
    logits=model(**batch).logits[:,:-1]; labels=batch["input_ids"][:,1:]; values=logits.log_softmax(-1).gather(-1,labels.unsqueeze(-1)).squeeze(-1); mask=batch["attention_mask"][:,1:].bool()
    for i,n in enumerate(prompt_lengths):mask[i,:max(0,n-1)]=False
    return (values*mask).sum(1),mask.sum(1)


def load_policy(config,adapter,trainable,device):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    base=AutoModelForCausalLM.from_pretrained(config["model"]["base"],revision=config["model"]["base_revision"])
    model=PeftModel.from_pretrained(base,adapter,is_trainable=trainable).to(device); model.config.use_cache=not trainable
    return model


def load_reward(config,sft_adapter,reward_adapter,device):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    base=AutoModelForCausalLM.from_pretrained(config["model"]["base"],revision=config["model"]["base_revision"])
    merged=PeftModel.from_pretrained(base,sft_adapter).merge_and_unload(); reward=GPT2RewardModel(merged.config); reward.transformer.load_state_dict(merged.transformer.state_dict()); del base,merged
    return PeftModel.from_pretrained(reward,reward_adapter).to(device).eval()


def dpo_update(config,tok,source_adapter,pairs,target,steps,seed,torch,device):
    target.mkdir(parents=True,exist_ok=True); training_ledger=target/"training_preferences.jsonl"; write_jsonl(training_ledger,pairs)
    policy=load_policy(config,source_adapter,True,device); reference=load_policy(config,ROOT/"checkpoints/recursive_self_improvement_v1/sft",False,device).eval()
    for p in reference.parameters():p.requires_grad_(False)
    optimizer=torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad),lr=config["ordinary_dpo"]["learning_rate"])
    accumulation=16; beta=config["ordinary_dpo"]["beta"]; rng=random.Random(seed); order=[]
    while len(order)<steps*accumulation:
        block=list(range(len(pairs)));rng.shuffle(block);order.extend(block)
    order=order[:steps*accumulation]; losses=[]; tokens=0; optimizer.zero_grad(set_to_none=True); start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for seen,index in enumerate(order,1):
        row=pairs[index]; prompts=[row["prompt"]]; chosen=[row["chosen"]]; rejected=[row["rejected"]]
        with torch.autocast("cuda",dtype=torch.float16):
            pc,tc=response_logp(policy,tok,prompts,chosen,config["model"]["context_length"],device);pr,tr=response_logp(policy,tok,prompts,rejected,config["model"]["context_length"],device)
            with torch.no_grad():rc,_=response_logp(reference,tok,prompts,chosen,config["model"]["context_length"],device);rr,_=response_logp(reference,tok,prompts,rejected,config["model"]["context_length"],device)
            loss=-torch.nn.functional.logsigmoid(beta*((pc-pr)-(rc-rr))).mean()/accumulation
        loss.backward();losses.append(float(loss.detach())*accumulation);tokens+=int(tc.item()+tr.item())
        if seen%accumulation==0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(),1.0);optimizer.step();optimizer.zero_grad(set_to_none=True)
            if seen%(accumulation*20)==0:print(json.dumps({"stage":target.name,"optimizer_steps":seen//accumulation,"loss_mean_last20":sum(losses[-accumulation*20:])/len(losses[-accumulation*20:])}),flush=True)
    policy.save_pretrained(target);tok.save_pretrained(target)
    save_json(target/"run_manifest.json",{"status":"complete","git_commit":commit(),"source_adapter":str(Path(source_adapter).relative_to(ROOT)),"preference_rows":len(pairs),"preference_sha256":sha(training_ledger),"optimizer_steps":steps,"microbatch_pairs":1,"gradient_accumulation_steps":accumulation,"effective_batch_pairs":accumulation,"non_padding_training_tokens":tokens,"loss_mean":sum(losses)/len(losses),"runtime_seconds":time.perf_counter()-start,"peak_allocated_gpu_memory_bytes":torch.cuda.max_memory_allocated(),"seed":seed})
    del policy,reference,optimizer;torch.cuda.empty_cache()


def rollout(config,tok,adapter,prompt_rows,round_dir,seed,torch,device):
    candidates_path=round_dir/"candidates.jsonl"
    if candidates_path.exists():return read_jsonl(candidates_path)
    policy=load_policy(config,adapter,False,device).eval(); seed_all(seed,torch); rows=[]; start=time.perf_counter();tokens=0
    with torch.no_grad():
        for i,row in enumerate(prompt_rows,1):
            enc=encode(tok,[row["prompt"]],256,device); outputs=[]
            for candidate in range(2):
                generator=torch.Generator(device=device).manual_seed(seed+i*17+candidate)
                out=policy.generate(**enc,do_sample=True,temperature=.8 if candidate==0 else 1.0,top_p=.9 if candidate==0 else .95,max_new_tokens=config["evaluation"]["max_new_tokens"],pad_token_id=tok.eos_token_id,generator=generator)
                response=tok.decode(out[0,enc["input_ids"].shape[1]:],skip_special_tokens=True);outputs.append(response);tokens+=int(out.shape[1]-enc["input_ids"].shape[1])
            rows.append({"prompt_id":row["prompt_id"],"prompt":row["prompt"],"candidate_a":outputs[0],"candidate_b":outputs[1]})
            if i%32==0:print(json.dumps({"stage":round_dir.name,"rollout_prompts":i}),flush=True)
    write_jsonl(candidates_path,rows);save_json(round_dir/"rollout_manifest.json",{"status":"complete","adapter":str(Path(adapter).relative_to(ROOT)),"prompts":len(rows),"generated_tokens":tokens,"runtime_seconds":time.perf_counter()-start,"sha256":sha(candidates_path),"seed":seed})
    del policy;torch.cuda.empty_cache();return rows


def label(config,tok,candidates,round_dir,torch,device):
    scored_path=round_dir/"scored_candidates.jsonl";preferences_path=round_dir/"preferences.jsonl"
    if preferences_path.exists():return read_jsonl(preferences_path)
    for row in candidates:row["member_scores_a"]=[];row["member_scores_b"]=[]
    sft=ROOT/"checkpoints/recursive_self_improvement_v1/sft"
    for seed in config["reward_ensemble"]["member_seeds"]:
        model=load_reward(config,sft,ROOT/f"checkpoints/recursive_self_improvement_v1/reward_ensemble/seed{seed}",device)
        with torch.no_grad():
            for i,row in enumerate(candidates,1):
                a=encode(tok,[row["prompt"]+row["candidate_a"]],config["model"]["context_length"],device);b=encode(tok,[row["prompt"]+row["candidate_b"]],config["model"]["context_length"],device)
                row["member_scores_a"].append(float(model(**a).rewards.item()));row["member_scores_b"].append(float(model(**b).rewards.item()))
        del model;torch.cuda.empty_cache()
    pairs=[]
    for row in candidates:
        ma=sum(row["member_scores_a"])/len(row["member_scores_a"]);mb=sum(row["member_scores_b"])/len(row["member_scores_b"]); a_wins=ma>=mb
        row["ensemble_mean_a"]=ma;row["ensemble_mean_b"]=mb
        pairs.append({"prompt_id":row["prompt_id"],"prompt":row["prompt"],"chosen":row["candidate_a"] if a_wins else row["candidate_b"],"rejected":row["candidate_b"] if a_wins else row["candidate_a"],"chosen_candidate":"a" if a_wins else "b","ensemble_margin":abs(ma-mb),"member_scores_a":row["member_scores_a"],"member_scores_b":row["member_scores_b"]})
    write_jsonl(scored_path,candidates);write_jsonl(preferences_path,pairs);return pairs


def main():
    import torch
    from transformers import AutoTokenizer
    if not torch.cuda.is_available():raise SystemExit("CUDA required")
    config=load_effective_config(ROOT);device=torch.device("cuda");seed=config["seed"];tok=AutoTokenizer.from_pretrained(config["model"]["base"],revision=config["model"]["base_revision"]);tok.pad_token=tok.eos_token
    root=ROOT/"checkpoints/recursive_self_improvement_v1";results=ROOT/"results/recursive_self_improvement_v1/stage1";data=ROOT/"data/processed/recursive_self_improvement_v1"
    ordinary=root/"ordinary_dpo";ordinary_pairs=read_jsonl(data/"reward_train.jsonl")[:config["ordinary_dpo"]["human_preference_pairs"]]
    ordinary_parent=ordinary.parent/"ordinary_dpo_inputs";ordinary_parent.mkdir(parents=True,exist_ok=True);write_jsonl(ordinary_parent/"preferences.jsonl",ordinary_pairs)
    if not complete(ordinary/"run_manifest.json"):dpo_update(config,tok,root/"sft",ordinary_pairs,ordinary,config["ordinary_dpo"]["steps"],seed+100,torch,device)
    prompts=read_jsonl(data/"improvement.jsonl");block=config["stage1"]["prompt_schedule"]["prompts_per_round"]
    for round_index in range(1,config["stage1"]["rounds"]+1):
        round_dir=results/f"round_{round_index}";target=root/f"iterative_dpo_round_{round_index}";source=root/"sft" if round_index==1 else root/f"iterative_dpo_round_{round_index-1}"
        if complete(target/"run_manifest.json"):continue
        current=rollout(config,tok,source,prompts[(round_index-1)*block:round_index*block],round_dir,seed+round_index,torch,device);label(config,tok,current,round_dir,torch,device)
        buffer=[]
        for prior in range(max(1,round_index-1),round_index+1):buffer.extend(read_jsonl(results/f"round_{prior}/preferences.jsonl"))
        write_jsonl(target.parent/f"iterative_dpo_round_{round_index}_inputs/preferences.jsonl",buffer)
        # Copy the exact training ledger alongside the result before updating.
        target.parent.joinpath(f"iterative_dpo_round_{round_index}_inputs").mkdir(parents=True,exist_ok=True)
        dpo_update(config,tok,source,buffer,target,config["stage1"]["dpo_steps_per_round"],seed+1000+round_index,torch,device)

if __name__=="__main__":main()
