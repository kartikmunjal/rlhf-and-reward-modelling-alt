#!/usr/bin/env python3
"""Windows GPU worker for one frozen Stage-3 condition/round boundary."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from recursive_self_improvement.config import load_effective_config
from scripts.train_recursive_stage1 import complete,dpo_update,encode,load_policy,read_jsonl,response_logp,rollout,write_jsonl

def paths(percent,round_index):
 result=ROOT/f"results/recursive_self_improvement_v1/stage3/self_{percent}/round_{round_index}"
 target=ROOT/f"checkpoints/recursive_self_improvement_v1/stage3_self_{percent}_round_{round_index}"
 source=ROOT/"checkpoints/recursive_self_improvement_v1/iterative_dpo_round_3" if round_index==1 else ROOT/f"checkpoints/recursive_self_improvement_v1/stage3_self_{percent}_round_{round_index-1}"
 return result,target,source

def prepare(config,percent,round_index,tok,torch,device):
 result,target,source=paths(percent,round_index); scored=result/"candidates_with_self_scores.jsonl"
 if scored.exists():print(scored);return
 allocation=config["stage3"]["prompt_allocation"]; start,stop=allocation["round_blocks"][round_index-1]; prompt_rows=read_jsonl(ROOT/allocation["source"])[start:stop]
 candidates=rollout(config,tok,source,prompt_rows,result,config["seed"]+30000+round_index,torch,device)
 model=load_policy(config,source,False,device).eval()
 with torch.no_grad():
  for i,row in enumerate(candidates,1):
   a,na=response_logp(model,tok,[row["prompt"]],[row["candidate_a"]],config["model"]["context_length"],device);b,nb=response_logp(model,tok,[row["prompt"]],[row["candidate_b"]],config["model"]["context_length"],device)
   row["self_mean_logp_a"]=float((a/na.clamp_min(1)).item());row["self_mean_logp_b"]=float((b/nb.clamp_min(1)).item());row["percent_self"]=percent;row["round"]=round_index
   if i%32==0:print(json.dumps({"self_scored":i,"percent_self":percent,"round":round_index}),flush=True)
 write_jsonl(scored,candidates);print(scored)

def train(config,percent,round_index,tok,torch,device):
 result,target,source=paths(percent,round_index)
 if complete(target/"run_manifest.json"):print("already complete");return
 current=result/"preferences.jsonl"
 if not current.exists():raise SystemExit(f"Missing {current}")
 current_rows=read_jsonl(current)
 if len(current_rows)!=config["stage1"]["rollout_prompts_per_round"]:raise ValueError("Stage-3 preference ledger must contain exactly 256 rows")
 buffer=[]
 for prior in range(max(1,round_index-1),round_index+1):buffer.extend(read_jsonl(ROOT/f"results/recursive_self_improvement_v1/stage3/self_{percent}/round_{prior}/preferences.jsonl"))
 dpo_update(config,tok,source,buffer,target,config["stage1"]["dpo_steps_per_round"],config["seed"]+40000+percent*100+round_index,torch,device)

def main():
 p=argparse.ArgumentParser();p.add_argument("phase",choices=["prepare","train"]);p.add_argument("--percent-self",type=int,required=True);p.add_argument("--round",type=int,required=True);a=p.parse_args()
 import torch
 from transformers import AutoTokenizer
 c=load_effective_config(ROOT)
 if a.percent_self not in c["stage3"]["label_mixture_percent_self"] or not 1<=a.round<=c["stage3"]["rounds_per_condition"]:raise SystemExit("outside frozen matrix")
 if not torch.cuda.is_available():raise SystemExit("CUDA required")
 tok=AutoTokenizer.from_pretrained(c["model"]["base"],revision=c["model"]["base_revision"]);tok.pad_token=tok.eos_token;device=torch.device("cuda")
 (prepare if a.phase=="prepare" else train)(c,a.percent_self,a.round,tok,torch,device)
if __name__=="__main__":main()
