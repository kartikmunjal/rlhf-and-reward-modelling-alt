$ErrorActionPreference = "Stop"
$repo = "C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-and-reward-modelling-alt"
$log = Join-Path $repo "logs\ppo_grpo_v1_cleanup.log"

while ((Get-ScheduledTask -TaskName "PPOGRPOV1").State -eq "Running") {
    Start-Sleep -Seconds 30
}

$info = Get-ScheduledTaskInfo -TaskName "PPOGRPOV1"
if ($info.LastTaskResult -ne 0) {
    throw "Training task failed with result $($info.LastTaskResult); retaining model cache for diagnosis."
}

foreach ($method in @("ppo", "grpo")) {
    foreach ($seed in @(2025, 2026, 2027)) {
        $run = Join-Path $repo "results\ppo_grpo_v1\${method}_seed${seed}"
        $manifest = Join-Path $run "run_manifest.json"
        $predictions = Join-Path $run "predictions.json"
        if (-not (Test-Path -LiteralPath $manifest)) {
            throw "Missing manifest: $manifest"
        }
        if (-not (Test-Path -LiteralPath $predictions)) {
            throw "Missing predictions: $predictions"
        }
        $rows = Get-Content -LiteralPath $predictions -Raw | ConvertFrom-Json
        if ($rows.Count -ne 400) {
            throw "Expected 400 predictions in $predictions; found $($rows.Count)."
        }
    }
}

$cacheRoot = Join-Path $env:USERPROFILE ".cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct"
if ((Split-Path $cacheRoot -Leaf) -ne "models--Qwen--Qwen2.5-0.5B-Instruct") {
    throw "Refusing unexpected cache target: $cacheRoot"
}
if (Test-Path -LiteralPath $cacheRoot) {
    $bytes = (Get-ChildItem -LiteralPath $cacheRoot -Recurse -File | Measure-Object Length -Sum).Sum
    Remove-Item -LiteralPath $cacheRoot -Recurse -Force
    "[$(Get-Date -Format o)] Verified six runs and 2,400 predictions; removed Qwen cache $cacheRoot ($bytes bytes)." |
        Set-Content -LiteralPath $log -Encoding UTF8
} else {
    "[$(Get-Date -Format o)] Verified six runs and 2,400 predictions; Qwen cache already absent." |
        Set-Content -LiteralPath $log -Encoding UTF8
}
