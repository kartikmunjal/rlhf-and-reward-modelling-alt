#!/usr/bin/env python3
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from recursive_self_improvement.config import canonical_text_sha256
p=ROOT/"recursive_self_improvement/protocol_amendment_008_stage3_prompt_allocation.json"
(ROOT/"recursive_self_improvement/protocol_amendment_008_manifest.json").write_text(json.dumps({"amendment_id":"recursive_self_improvement_v1_stage3_prompt_allocation_008","path":str(p.relative_to(ROOT)),"sha256":canonical_text_sha256(p)},indent=2)+"\n",encoding="utf-8")
