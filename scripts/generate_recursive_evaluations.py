#!/usr/bin/env python3
"""Generate deterministic held-out outputs for every primary-family checkpoint."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from recursive_self_improvement.config import load_effective_config

def rows(path):return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]
def write(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(".tmp")
 with tmp.open("w",encoding="utf-8",newline="\n") as h:
  for row in data:h.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n")
 os.replace(tmp,path)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(config,name,device):
 from transformers import AutoModelForCausalLM
 base=AutoModelForCausalLM.from_pretrained(config["model"]["base"],revision=config["model"]["base_revision"])
 if name=="base":return base.to(device).eval()
 from peft import PeftModel
 mapping={"sft":"sft","dpo":"ordinary_dpo","ordinary_dpo":"ordinary_dpo",**{f"iterative_dpo_round_{i}":f"iterative_dpo_round_{i}" for i in range(1,9)}}
 return PeftModel.from_pretrained(base,ROOT/"checkpoints/recursive_self_improvement_v1"/mapping[name]).to(device).eval()
def main():
 import torch
 from transformers import AutoTokenizer
 if not torch.cuda.is_available():raise SystemExit("CUDA required")
 config=load_effective_config(ROOT);device=torch.device("cuda");tok=AutoTokenizer.from_pretrained(config["model"]["base"],revision=config["model"]["base_revision"]);tok.pad_token=tok.eos_token
 prompts=rows(ROOT/"data/processed/recursive_self_improvement_v1/independent_eval.jsonl");outdir=ROOT/"results/recursive_self_improvement_v1/generations"
 names=config["scope"]["primary_family"]
 for name in names:
  path=outdir/f"{name}.jsonl";manifest=outdir/f"{name}.manifest.json"
  if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")).get("status")=="complete":continue
  model=load(config,name,device);generated=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
  with torch.no_grad():
   for i,row in enumerate(prompts,1):
    enc=tok(row["prompt"],return_tensors="pt",truncation=True,max_length=256).to(device);out=model.generate(**enc,do_sample=False,max_new_tokens=config["evaluation"]["max_new_tokens"],pad_token_id=tok.eos_token_id);text=tok.decode(out[0,enc["input_ids"].shape[1]:],skip_special_tokens=True)
    generated.append({"checkpoint":name,"prompt_id":row["prompt_id"],"prompt":row["prompt"],"response":text})
    if i%25==0:print(json.dumps({"checkpoint":name,"prompts":i}),flush=True)
  write(path,generated);manifest.write_text(json.dumps({"status":"complete","checkpoint":name,"rows":len(generated),"sha256":digest(path),"runtime_seconds":time.perf_counter()-start,"peak_allocated_gpu_memory_bytes":torch.cuda.max_memory_allocated(),"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()},indent=2,sort_keys=True)+"\n",encoding="utf-8")
  del model;torch.cuda.empty_cache()
if __name__=="__main__":main()
