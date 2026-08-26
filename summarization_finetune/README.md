# Summarization Fine-Tuning + Judge-in-the-Loop

This module closes the loop from training through frozen-instrument evaluation:

1. generate a zero-shot `gpt2-medium` baseline;
2. LoRA-SFT on contamination-filtered CNN/DailyMail train;
3. evaluate base and SFT with the unchanged validated SummEval judge;
4. sample SFT candidates and construct conservative judge preferences;
5. LoRA-DPO against those preferences;
6. compare DPO with SFT on the identical untouched articles;
7. test whether any DPO gain survives length control and agrees with ROUGE-L
   and the amended GPT-5-mini cross-provider audit.

The locked protocols are [`sft_preregistration.md`](sft_preregistration.md) and
[`dpo_preregistration.md`](dpo_preregistration.md). Their hashes and the frozen
judge hashes are in [`preregistration_manifest.json`](preregistration_manifest.json).
The judge prompt, rubric, schemas, and Claude snapshot are never tuned here.

## Fixed experiment

The full CNN/DailyMail train split is used for one rank-16 LoRA SFT epoch. The
RTX 3070 configuration uses batch size one, gradient accumulation, fp16, and
gradient checkpointing. DPO uses the merged SFT checkpoint as both initialization
and frozen reference, training only a fresh rank-16 adapter. One training seed
is in scope; article-bootstrap intervals therefore do not measure seed variance.

SummEval is excluded by canonical ID and normalized source hash. Preference,
loss-evaluation, and final-evaluation partitions are deterministically selected
and disjoint. The data preparer refuses unexpected standard split sizes,
remaining SummEval overlap, or any cross-partition overlap.

## Windows GPU execution

From the repository root, create a gitignored `.env` containing
`ANTHROPIC_API_KEY` and `OPENAI_API_KEY`, then run:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q tests\test_summarization_finetune.py
.\scripts\register_summarization_finetune_task.ps1
```

The registered task runs [`run_summarization_finetune_windows.cmd`](../scripts/run_summarization_finetune_windows.cmd),
writes `logs\summarization_finetune_v1.log`, and resumes completed generation
and API requests. Training checkpoints are saved periodically. Status commands:

```powershell
Get-ScheduledTask -TaskName "SummarizationFineTuneV1" | Select-Object TaskName, State
Get-Content ".\logs\summarization_finetune_v1.log" -Tail 50
.\.venv-gpu\Scripts\python.exe scripts\status_summarization_finetune.py
```

The end-to-end runner downloads CNN/DailyMail and GPT-2-medium through the
standard Hugging Face cache. Checkpoints and bulk processed data remain
gitignored; locked manifests, API audit ledgers, generated summaries, metrics,
and reports are publishable after the run and integrity audit.

## Claims

SFT and DPO each require a positive paired article-bootstrap lower confidence
bound on both relevance and consistency. DPO preference judging and held-out
evaluation must also meet the locked validity floor. A raw DPO gain that does
not survive within-article length adjustment and lacks independent proxy support
is reported as length-exploitation risk. No secondary metric can rescue a failed
primary claim.
