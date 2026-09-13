#!/usr/bin/env python3
"""Validate position-controlled judge records and build analysis inputs."""
import json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from recursive_self_improvement.config import load_effective_config
from recursive_self_improvement.statistics import wilson

def read(path):
 with Path(path).open(encoding="utf-8") as handle:return [json.loads(line) for line in handle if line.strip()]
def write(path,rows):
 with Path(path).open("w",encoding="utf-8",newline="\n") as h:
  for r in rows:h.write(json.dumps(r,sort_keys=True)+"\n")
def score(row):
 w=row["parsed"]["winner"]
 if w=="tie":return .5
 return float((row["order"]=="candidate_first" and w=="A") or (row["order"]=="reference_first" and w=="B"))
def main():
 c=load_effective_config(ROOT);root=ROOT/"results/recursive_self_improvement_v1";records=[r for r in read(root/"claude_pairwise.jsonl") if r["status"]=="success"]
 grouped=defaultdict(dict)
 for r in records:
  key=(r["checkpoint"],r["prompt_id"])
  if r["order"] in grouped[key]:raise ValueError(f"duplicate successful order {key}")
  grouped[key][r["order"]]=r
 output=[]
 for (checkpoint,pid),orders in sorted(grouped.items()):
  valid=set(orders)=={"candidate_first","reference_first"}
  values=[score(orders[x]) for x in sorted(orders)] if valid else []
  output.append({"checkpoint":checkpoint,"prompt_id":pid,"valid":valid,"win_vs_sft":sum(values)/2 if valid else None,"position_consistent":valid and values[0]==values[1]})
 for row in read(root/"generations/sft.jsonl"):output.append({"checkpoint":"sft","prompt_id":row["prompt_id"],"valid":True,"win_vs_sft":.5,"position_consistent":True})
 expected=len(c["scope"]["primary_family"])*c["data"]["independent_eval_prompts"]
 if len(output)!=expected:raise ValueError(f"expected {expected}, got {len(output)}")
 write(root/"evaluations.jsonl",output)
 judged=[r for r in output if r["checkpoint"]!="sft" and r["valid"]];consistent=sum(r["position_consistent"] for r in judged);position=wilson(consistent,len(judged))
 usage={"successful_calls":len(records),"input_tokens":sum(r["input_tokens"] for r in records),"output_tokens":sum(r["output_tokens"] for r in records),"estimated_usd":(sum(r["input_tokens"] for r in records)+5*sum(r["output_tokens"] for r in records))/1e6,"judged_prompt_checkpoint_pairs":len(judged),"evaluation_rows_including_synthetic_sft_reference":sum(r["valid"] for r in output),"position_consistent_pairs":consistent,"position_consistency_rate":position["rate"],"position_consistency_wilson_ci95":position["wilson_ci95"]}
 (root/"judge_audit.json").write_text(json.dumps(usage,indent=2,sort_keys=True)+"\n",encoding="utf-8");print(json.dumps(usage,indent=2))
if __name__=="__main__":main()
