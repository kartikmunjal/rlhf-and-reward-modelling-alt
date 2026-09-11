#!/usr/bin/env python3
"""Build the preregistered policy-lineage compute ledger from run manifests."""
import argparse,json
from pathlib import Path

def manifest(path):
 row=json.loads(Path(path).read_text(encoding="utf-8"))
 if row.get("status")!="complete":raise ValueError(f"incomplete manifest: {path}")
 return row
def main():
 p=argparse.ArgumentParser();p.add_argument("--manifest-root",type=Path,required=True);p.add_argument("--output",type=Path,default=Path("results/recursive_self_improvement_v1/compute.jsonl"));a=p.parse_args()
 sft=manifest(a.manifest_root/"sft/run_manifest.json");ordinary=manifest(a.manifest_root/"ordinary_dpo/run_manifest.json")
 rows=[{"checkpoint":"base","cumulative_non_padding_training_tokens":0,"cumulative_optimizer_steps":0,"cumulative_runtime_seconds":0.0,"incremental_peak_allocated_gpu_memory_bytes":0,"lineage":"base"},{"checkpoint":"sft","cumulative_non_padding_training_tokens":sft["non_padding_training_tokens"],"cumulative_optimizer_steps":sft["optimizer_steps"],"cumulative_runtime_seconds":sft["runtime_seconds"],"incremental_peak_allocated_gpu_memory_bytes":sft["peak_allocated_gpu_memory_bytes"],"lineage":"sft"},{"checkpoint":"dpo","cumulative_non_padding_training_tokens":sft["non_padding_training_tokens"]+ordinary["non_padding_training_tokens"],"cumulative_optimizer_steps":sft["optimizer_steps"]+ordinary["optimizer_steps"],"cumulative_runtime_seconds":sft["runtime_seconds"]+ordinary["runtime_seconds"],"incremental_peak_allocated_gpu_memory_bytes":ordinary["peak_allocated_gpu_memory_bytes"],"lineage":"sft_to_ordinary_dpo"}]
 tokens=sft["non_padding_training_tokens"];steps=sft["optimizer_steps"];runtime=sft["runtime_seconds"]
 for i in range(1,9):
  row=manifest(a.manifest_root/f"iterative_dpo_round_{i}/run_manifest.json");tokens+=row["non_padding_training_tokens"];steps+=row["optimizer_steps"];runtime+=row["runtime_seconds"]
  rows.append({"checkpoint":f"iterative_dpo_round_{i}","cumulative_non_padding_training_tokens":tokens,"cumulative_optimizer_steps":steps,"cumulative_runtime_seconds":runtime,"lineage":"sft_to_iterative_dpo","incremental_non_padding_training_tokens":row["non_padding_training_tokens"],"incremental_peak_allocated_gpu_memory_bytes":row["peak_allocated_gpu_memory_bytes"]})
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open("w",encoding="utf-8",newline="\n") as h:
  for row in rows:h.write(json.dumps(row,sort_keys=True)+"\n")
 print(a.output)
if __name__=="__main__":main()
