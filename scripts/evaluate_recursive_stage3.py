#!/usr/bin/env python3
"""Generate and independently evaluate every Stage-3 checkpoint with frozen K=3 RMs."""
import hashlib,json,math,os,statistics,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from recursive_self_improvement.config import load_effective_config
from scripts.train_recursive_stage1 import encode,load_policy,load_reward,read_jsonl,write_jsonl

def save(path,payload):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8");os.replace(tmp,path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def complete(path):
 try:return json.loads(Path(path).read_text(encoding="utf-8")).get("status")=="complete"
 except:return False
def response_kl(policy,reference,tok,prompt,response,maximum,device,torch):
 batch=encode(tok,[prompt+response],maximum,device);prompt_len=len(tok(prompt,truncation=True,max_length=maximum)["input_ids"])
 with torch.no_grad(),torch.autocast("cuda",dtype=torch.float16):p=policy(**batch).logits[:,:-1].float().log_softmax(-1);q=reference(**batch).logits[:,:-1].float().log_softmax(-1)
 mask=batch["attention_mask"][:,1:].bool();mask[:,:max(0,prompt_len-1)]=False;token_kl=(p.exp()*(p-q)).sum(-1)
 return float(token_kl[mask].mean().item()) if mask.any() else 0.0
def generate_all(c,tok,torch,device,outroot):
 prompts=read_jsonl(ROOT/"data/processed/recursive_self_improvement_v1/independent_eval.jsonl");start_adapter=ROOT/"checkpoints/recursive_self_improvement_v1/iterative_dpo_round_3"
 for percent in c["stage3"]["label_mixture_percent_self"]:
  for round_index in range(1,c["stage3"]["rounds_per_condition"]+1):
   name=f"self_{percent}_round_{round_index}";path=outroot/"generations"/f"{name}.jsonl";manifest=path.with_suffix(".manifest.json")
   if complete(manifest):continue
   adapter=ROOT/f"checkpoints/recursive_self_improvement_v1/stage3_self_{percent}_round_{round_index}";policy=load_policy(c,adapter,False,device).eval();reference=load_policy(c,start_adapter,False,device).eval();rows=[];started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
   with torch.no_grad():
    for i,row in enumerate(prompts,1):
     batch=tok(row["prompt"],return_tensors="pt",truncation=True,max_length=256).to(device);out=policy.generate(**batch,do_sample=False,max_new_tokens=c["evaluation"]["max_new_tokens"],pad_token_id=tok.eos_token_id);response=tok.decode(out[0,batch["input_ids"].shape[1]:],skip_special_tokens=True)
     rows.append({"checkpoint":name,"percent_self":percent,"round":round_index,"prompt_id":row["prompt_id"],"prompt":row["prompt"],"response":response,"response_tokens":len(tok(response)["input_ids"]),"response_words":len(response.split()),"kl_from_stage3_start":response_kl(policy,reference,tok,row["prompt"],response,c["model"]["context_length"],device,torch)})
     if i%25==0:print(json.dumps({"stage3_eval_generation":name,"prompts":i}),flush=True)
   write_jsonl(path,rows);save(manifest,{"status":"complete","rows":len(rows),"sha256":sha(path),"runtime_seconds":time.perf_counter()-started,"peak_allocated_gpu_memory_bytes":torch.cuda.max_memory_allocated()});del policy,reference;torch.cuda.empty_cache()
def score_all(c,tok,torch,device,outroot):
 reference={r["prompt_id"]:r for r in read_jsonl(ROOT/"results/recursive_self_improvement_v1/generations/iterative_dpo_round_3.jsonl")};items=[]
 for path in sorted((outroot/"generations").glob("self_*_round_*.jsonl")):items.extend(read_jsonl(path))
 for seed in c["reward_ensemble"]["member_seeds"]:
  path=outroot/"reward_scores"/f"seed{seed}.jsonl";manifest=path.with_suffix(".manifest.json")
  if complete(manifest):continue
  model=load_reward(c,ROOT/"checkpoints/recursive_self_improvement_v1/sft",ROOT/f"checkpoints/recursive_self_improvement_v1/reward_ensemble/seed{seed}",device);scores=[];started=time.perf_counter()
  with torch.no_grad():
   for i,row in enumerate(items,1):
    a=encode(tok,[row["prompt"]+row["response"]],c["model"]["context_length"],device);ref=reference[row["prompt_id"]];b=encode(tok,[ref["prompt"]+ref["response"]],c["model"]["context_length"],device)
    scores.append({"checkpoint":row["checkpoint"],"prompt_id":row["prompt_id"],"policy_reward":float(model(**a).rewards.item()),"reference_reward":float(model(**b).rewards.item()),"seed":seed})
    if i%250==0:print(json.dumps({"reward_seed":seed,"scored":i,"total":len(items)}),flush=True)
  write_jsonl(path,scores);save(manifest,{"status":"complete","rows":len(scores),"sha256":sha(path),"runtime_seconds":time.perf_counter()-started});del model;torch.cuda.empty_cache()
def assemble(c,outroot):
 generation={}
 for path in (outroot/"generations").glob("*.jsonl"):
  for r in read_jsonl(path):generation[(r["checkpoint"],r["prompt_id"])]=r
 scores={k:[] for k in generation}
 for path in (outroot/"reward_scores").glob("seed*.jsonl"):
  for r in read_jsonl(path):scores[(r["checkpoint"],r["prompt_id"])].append(r)
 output=[]
 for key,row in sorted(generation.items()):
  members=scores[key]
  if len(members)!=3:raise ValueError(f"Expected K=3 for {key}")
  margins=[r["policy_reward"]-r["reference_reward"] for r in members];mean=sum(margins)/3
  output.append({**row,"valid":True,"external_evaluator_win":1.0 if mean>0 else 0.0 if mean<0 else .5,"external_margin":mean,"ensemble_disagreement":statistics.pstdev(margins),"member_margins":margins})
 write_jsonl(ROOT/"results/recursive_self_improvement_v1/stage3_evaluations.jsonl",output);save(outroot/"evaluation_manifest.json",{"status":"complete","rows":len(output),"sha256":sha(ROOT/"results/recursive_self_improvement_v1/stage3_evaluations.jsonl")})
def main():
 import torch
 from transformers import AutoTokenizer
 if not torch.cuda.is_available():raise SystemExit("CUDA required")
 c=load_effective_config(ROOT);tok=AutoTokenizer.from_pretrained(c["model"]["base"],revision=c["model"]["base_revision"]);tok.pad_token=tok.eos_token;device=torch.device("cuda");out=ROOT/"results/recursive_self_improvement_v1/stage3_evaluation"
 generate_all(c,tok,torch,device,out);score_all(c,tok,torch,device,out);assemble(c,out)
if __name__=="__main__":main()
