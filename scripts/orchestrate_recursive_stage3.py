#!/usr/bin/env python3
"""Resume the frozen Stage-3 Windows/Claude hybrid matrix end to end."""
import base64,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from recursive_self_improvement.config import load_effective_config

REMOTE=r"C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-recursive-run"
def run(cmd,**kw):
 print("RUN",cmd[0],*("<redacted>" if "KEY" in x else x for x in cmd[1:3]),flush=True);return subprocess.run(cmd,check=True,text=True,**kw)
def ssh_ps(code,capture=False):
 encoded=base64.b64encode(code.encode("utf-16le")).decode("ascii")
 return run(["ssh","norgate","powershell","-NoProfile","-EncodedCommand",encoded],capture_output=capture).stdout if capture else ""
def task_info(task):
 code=f"$t=Get-ScheduledTask -TaskName '{task}' -ErrorAction SilentlyContinue; if ($null -eq $t) {{ 'Missing|NA' }} else {{ $i=Get-ScheduledTaskInfo -TaskName '{task}'; [string]$t.State+'|'+[string]$i.LastTaskResult }}"
 out=ssh_ps(code,True).strip().splitlines();return out[-1].strip() if out else "Missing|NA"
def launch(phase,percent,round_index):
 task=f"RecursiveStage3_{phase}_{percent}_{round_index}";state=task_info(task).split("|")[0]
 if state=="Running":return task
 code=f"Set-Location '{REMOTE}'; & '.\\scripts\\register_recursive_stage3_worker.ps1' -Phase {phase} -PercentSelf {percent} -Round {round_index}"
 ssh_ps(code);return task
def wait_task(task,artifact):
 while True:
  info=task_info(task);print(json.dumps({"task":task,"state":info}),flush=True)
  state,result=info.split("|",1)
  if state=="Ready":
   exists=ssh_ps(f"if (Test-Path '{artifact}') {{ 'yes' }} else {{ 'no' }}",True).strip().endswith("yes")
   if result=="0" and exists:return
   raise RuntimeError(f"{task} failed: {info}, artifact={exists}")
  if state=="Missing":raise RuntimeError(f"Missing task {task}")
  time.sleep(60)
def main():
 c=load_effective_config(ROOT);local=ROOT/"results/recursive_self_improvement_v1/stage3"
 for percent in c["stage3"]["label_mixture_percent_self"]:
  for round_index in range(1,c["stage3"]["rounds_per_condition"]+1):
   relative=f"results\\recursive_self_improvement_v1\\stage3\\self_{percent}\\round_{round_index}"
   remote_candidates=f"{REMOTE}\\{relative}\\candidates_with_self_scores.jsonl";task=launch("prepare",percent,round_index);wait_task(task,remote_candidates)
   here=local/f"self_{percent}/round_{round_index}";here.mkdir(parents=True,exist_ok=True);candidates=here/"candidates_with_self_scores.jsonl";preferences=here/"preferences.jsonl"
   if not candidates.exists():run(["scp",f"norgate:{remote_candidates.replace(chr(92),'/')}",str(candidates)])
   if not preferences.exists():run([sys.executable,"scripts/label_recursive_stage3_round.py","--candidates",str(candidates),"--output",str(preferences),"--percent-self",str(percent),"--round",str(round_index),"--workers","16"],cwd=ROOT)
   run(["scp",str(preferences),f"norgate:{REMOTE.replace(chr(92),'/')}/{relative.replace(chr(92),'/')}/preferences.jsonl"])
   remote_manifest=f"{REMOTE}\\checkpoints\\recursive_self_improvement_v1\\stage3_self_{percent}_round_{round_index}\\run_manifest.json";task=launch("train",percent,round_index);wait_task(task,remote_manifest)
 print("STAGE3_MATRIX_COMPLETE",flush=True)
if __name__=="__main__":main()
