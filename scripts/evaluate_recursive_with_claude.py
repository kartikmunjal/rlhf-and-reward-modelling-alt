#!/usr/bin/env python3
"""Resumable, position-controlled Claude evaluation of the primary family."""
import argparse,concurrent.futures,hashlib,json,random,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_judge_summeval.providers import AnthropicProvider,ProviderError
from recursive_self_improvement.config import load_effective_config

def load(path):return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]
def main():
 p=argparse.ArgumentParser();p.add_argument("--workers",type=int,default=16);p.add_argument("--limit",type=int);a=p.parse_args()
 config=load_effective_config(ROOT);prompt=json.loads((ROOT/config["independent_judge"]["prompt_path"]).read_text(encoding="utf-8"));root=ROOT/"results/recursive_self_improvement_v1";ledger=root/"claude_pairwise.jsonl";ledger.parent.mkdir(parents=True,exist_ok=True)
 sft={r["prompt_id"]:r for r in load(root/"generations/sft.jsonl")};done=set()
 if ledger.exists():
  for r in load(ledger):
   if r.get("status")=="success":done.add(r["request_id"])
 tasks=[]
 for name in config["scope"]["primary_family"]:
  if name=="sft":continue
  for row in load(root/f"generations/{name}.jsonl")[:a.limit]:
   for order in ("candidate_first","reference_first"):
    rid=hashlib.sha256(f"{name}:{row['prompt_id']}:{order}:{prompt['version']}".encode()).hexdigest()
    if rid not in done:tasks.append((rid,name,row,sft[row["prompt_id"]],order))
 provider=AnthropicProvider(prompt["model"]);lock=threading.Lock()
 def run(task):
  rid,name,row,ref,order=task;a_text,b_text=(row["response"],ref["response"]) if order=="candidate_first" else (ref["response"],row["response"])
  user=prompt["user_template"].format(prompt=row["prompt"],response_a=a_text,response_b=b_text)
  for attempt in range(6):
   try:
    result=provider.request(prompt["system"],user,prompt["schema"],max_tokens=350);out={"status":"success","request_id":rid,"checkpoint":name,"prompt_id":row["prompt_id"],"order":order,"parsed":result.parsed,"input_tokens":result.input_tokens,"output_tokens":result.output_tokens,"response_id":result.response_id,"served_model":result.model,"attempt":attempt,"completed_unix":time.time()};break
   except Exception as e:
    if attempt==5:out={"status":"failed","request_id":rid,"checkpoint":name,"prompt_id":row["prompt_id"],"order":order,"error_type":type(e).__name__,"error":str(e),"completed_unix":time.time()};break
    time.sleep(min(30,2**attempt+random.random()))
  with lock:
   with ledger.open("a",encoding="utf-8",newline="\n") as h:h.write(json.dumps(out,sort_keys=True)+"\n");h.flush()
  return out["status"]
 print(json.dumps({"pending":len(tasks),"workers":a.workers}),flush=True);counts={"success":0,"failed":0}
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
  for i,status in enumerate(ex.map(run,tasks),1):
   counts[status]+=1
   if i%100==0:print(json.dumps({"completed":i,"total":len(tasks),**counts}),flush=True)
 print(json.dumps(counts),flush=True)
if __name__=="__main__":main()
