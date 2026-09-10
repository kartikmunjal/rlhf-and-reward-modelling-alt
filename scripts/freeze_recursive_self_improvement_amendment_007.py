#!/usr/bin/env python3
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from recursive_self_improvement.config import canonical_text_sha256
paths=[ROOT/"recursive_self_improvement/protocol_amendment_007_task_appropriate_judge.json",ROOT/"recursive_self_improvement/claude_pairwise_prompt_v1.json"]
out={"amendment_id":"recursive_self_improvement_v1_task_appropriate_judge_007","files":[{"path":str(p.relative_to(ROOT)),"sha256":canonical_text_sha256(p)} for p in paths]}
(ROOT/"recursive_self_improvement/protocol_amendment_007_manifest.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
