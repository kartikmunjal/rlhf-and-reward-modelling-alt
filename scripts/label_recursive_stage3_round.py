#!/usr/bin/env python3
"""Apply the frozen self/external mixture to one Stage-3 candidate ledger."""
import argparse,concurrent.futures,hashlib,json,random,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_judge_summeval.providers import AnthropicProvider
from recursive_self_improvement.config import load_effective_config
from recursive_self_improvement.mixture import choose_by_normalized_log_likelihood,use_self_label

def read(p):return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x]
def score(row):
 w=row["parsed"]["winner"]
 if w=="tie":return .5
 return float((row["order"]=="a_first" and w=="A") or (row["order"]=="b_first" and w=="B"))
def main():
 p=argparse.ArgumentParser();p.add_argument("--candidates",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--percent-self",type=int,required=True);p.add_argument("--round",type=int,required=True);p.add_argument("--workers",type=int,default=16);a=p.parse_args()
 c=load_effective_config(ROOT);rows=read(a.candidates);prompt=json.loads((ROOT/c["independent_judge"]["prompt_path"]).read_text(encoding="utf-8"));ledger=a.output.parent/"claude_training_labels.jsonl";ledger.parent.mkdir(parents=True,exist_ok=True);done={}
 if ledger.exists():
  for r in read(ledger):
   if r.get("status")=="success":done[(r["prompt_id"],r["order"])]=r
 external=[r for r in rows if not use_self_label(c["seed"],a.percent_self,a.round,r["prompt_id"])]
 tasks=[(r,o) for r in external for o in ("a_first","b_first") if (r["prompt_id"],o) not in done];provider=AnthropicProvider(prompt["model"]);lock=threading.Lock()
 def call(task):
  row,order=task;ra,rb=(row["candidate_a"],row["candidate_b"]) if order=="a_first" else (row["candidate_b"],row["candidate_a"]);user=prompt["user_template"].replace("{prompt}",row["prompt"]).replace("{response_a}",ra).replace("{response_b}",rb)
  for attempt in range(6):
   try:
    z=provider.request(prompt["system"],user,prompt["schema"],max_tokens=350);out={"status":"success","prompt_id":row["prompt_id"],"order":order,"parsed":z.parsed,"input_tokens":z.input_tokens,"output_tokens":z.output_tokens,"response_id":z.response_id};break
   except Exception as e:
    if attempt==5:raise
    time.sleep(min(30,2**attempt+random.random()))
  with lock:
   with ledger.open("a",encoding="utf-8",newline="\n") as h:h.write(json.dumps(out,sort_keys=True)+"\n")
  return out
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
  for out in ex.map(call,tasks):done[(out["prompt_id"],out["order"])]=out
 preferences=[]
 for row in rows:
  if use_self_label(c["seed"],a.percent_self,a.round,row["prompt_id"]):chosen,rejected,rule=choose_by_normalized_log_likelihood(row["candidate_a"],row["candidate_b"],row["self_mean_logp_a"],row["self_mean_logp_b"]);source="self"
  else:
   vals=[score(done[(row["prompt_id"],o)]) for o in ("a_first","b_first")];mean=sum(vals)/2
   if mean>.5:chosen,rejected=row["candidate_a"],row["candidate_b"]
   elif mean<.5:chosen,rejected=row["candidate_b"],row["candidate_a"]
   else:chosen,rejected,rule=choose_by_normalized_log_likelihood(row["candidate_a"],row["candidate_b"],0,0)
   rule="claude_both_orders" if mean!=.5 else "claude_tie_hash";source="external"
  preferences.append({"prompt_id":row["prompt_id"],"prompt":row["prompt"],"chosen":chosen,"rejected":rejected,"label_source":source,"label_rule":rule,"percent_self":a.percent_self,"round":a.round,"self_mean_logp_a":row["self_mean_logp_a"],"self_mean_logp_b":row["self_mean_logp_b"]})
 with a.output.open("w",encoding="utf-8",newline="\n") as h:
  for r in preferences:h.write(json.dumps(r,sort_keys=True)+"\n")
 print(json.dumps({"rows":len(preferences),"self":sum(r["label_source"]=="self" for r in preferences),"external":sum(r["label_source"]=="external" for r in preferences),"new_api_calls":len(tasks)}))
if __name__=="__main__":main()
