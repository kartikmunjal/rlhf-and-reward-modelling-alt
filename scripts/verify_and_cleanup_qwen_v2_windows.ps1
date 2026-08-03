$ErrorActionPreference = "Stop"
$repo = "C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-and-reward-modelling-alt"
$log = Join-Path $repo "logs\ppo_grpo_v2_cleanup.log"
while ((Get-ScheduledTask -TaskName "PPOGRPOV2").State -eq "Running") { Start-Sleep -Seconds 30 }
$info = Get-ScheduledTaskInfo -TaskName "PPOGRPOV2"
if ($info.LastTaskResult -ne 0) { throw "V2 task failed; retaining checkpoints and cache." }
$baseline = Join-Path $repo "results\ppo_grpo_v2\baseline_sft\predictions.json"
if (-not (Test-Path -LiteralPath $baseline)) { throw "Missing baseline predictions." }
if ((Get-Content $baseline -Raw | ConvertFrom-Json).Count -ne 400) { throw "Baseline row-count failure." }
foreach ($method in @("ppo", "grpo")) {
    foreach ($seed in @(2025, 2026, 2027)) {
        $run = Join-Path $repo "results\ppo_grpo_v2\${method}_seed${seed}"
        $predictions = Get-Content (Join-Path $run "predictions.json") -Raw | ConvertFrom-Json
        $manifest = Get-Content (Join-Path $run "run_manifest.json") -Raw | ConvertFrom-Json
        if ($predictions.Count -ne 400) { throw "Prediction row-count failure: $method $seed" }
        if ($manifest.observed_budget.optimizer_steps -ne 200 -or
            $manifest.observed_budget.rollout_groups -ne 100 -or
            $manifest.observed_budget.generated_completions -ne 400) {
            throw "Compute-budget failure: $method $seed"
        }
    }
}
$targets = @(
    (Join-Path $env:USERPROFILE ".cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct"),
    (Join-Path $repo "checkpoints\ppo_grpo_v2_sft")
)
$removed = @()
foreach ($target in $targets) {
    if ((Split-Path $target -Leaf) -notin @("models--Qwen--Qwen2.5-0.5B-Instruct", "ppo_grpo_v2_sft")) {
        throw "Refusing unexpected cleanup target: $target"
    }
    if (Test-Path -LiteralPath $target) {
        $bytes = (Get-ChildItem -LiteralPath $target -Recurse -File | Measure-Object Length -Sum).Sum
        Remove-Item -LiteralPath $target -Recurse -Force
        $removed += "$target ($bytes bytes)"
    }
}
"[$(Get-Date -Format o)] Verified baseline, six runs, and budgets; removed: $($removed -join '; ')." |
    Set-Content -LiteralPath $log -Encoding UTF8
