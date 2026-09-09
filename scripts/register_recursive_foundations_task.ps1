$ErrorActionPreference = "Stop"
$repo = "C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-recursive-run"
$python = "C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-and-reward-modelling-alt\.venv-gpu\Scripts\python.exe"

Set-Location $repo
& $python -m py_compile "scripts\train_recursive_foundations.py"

$action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument '/c ""C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-recursive-run\scripts\run_recursive_foundations_windows.cmd""'
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5)
$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Days 3) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 2)

Register-ScheduledTask `
    -TaskName "RecursiveFoundationsV1" `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Force | Out-Null
Start-ScheduledTask -TaskName "RecursiveFoundationsV1"
Start-Sleep -Seconds 5
Get-ScheduledTask -TaskName "RecursiveFoundationsV1" |
    Select-Object TaskName, State
Get-ScheduledTaskInfo -TaskName "RecursiveFoundationsV1" |
    Select-Object LastRunTime, LastTaskResult, NextRunTime
