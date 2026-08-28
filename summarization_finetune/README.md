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
standard Hugging Face cache. Checkpoints, source-bearing API ledgers, generated
summaries, and bulk processed data remain gitignored. Aggregate metrics,
reports, the locked data manifest, and lightweight run manifests are published
after the integrity audit.

## Claims

SFT and DPO each require a positive paired article-bootstrap lower confidence
bound on both relevance and consistency. DPO preference judging and held-out
evaluation must also meet the locked validity floor. A raw DPO gain that does
not survive within-article length adjustment and lacks independent proxy support
is reported as length-exploitation risk. No secondary metric can rescue a failed
primary claim.

<!-- GENERATED-FINDINGS:START -->
## Generated findings

SFT **passed** its locked criterion: relevance improved by 0.995 (95% CI 0.884–1.111; N_trials=198; 2,000 bootstraps) and consistency by 0.596 (95% CI 0.460–0.722; N_trials=198; 2,000 bootstraps).

DPO **failed** its locked two-axis criterion. Relevance changed by 0.101 (95% CI 0.010–0.197; N_trials=198; 2,000 bootstraps); consistency changed by 0.061 (95% CI -0.040–0.162; N_trials=198; 2,000 bootstraps). DPO length changed by -0.555 (95% CI -2.645–1.565; N_trials=200; 2,000 bootstraps) words, providing no evidence of verbosity exploitation.

The run used N_training_seeds=1, 287,113 SFT examples, and 1,341 conservative judge-preference pairs. Claude confirmatory coverage met the locked floor; GPT-5-mini SFT/DPO coverage did not and remains diagnostic.

Complete estimates, 95% intervals, validity rates, and interpretation boundaries are generated in [`../results/summarization_finetune_v1/report.md`](../results/summarization_finetune_v1/report.md). Lightweight training provenance is in [`../results/summarization_finetune_v1/manifests/`](../results/summarization_finetune_v1/manifests/).
<!-- GENERATED-FINDINGS:END -->
