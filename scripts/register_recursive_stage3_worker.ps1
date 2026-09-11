param(
    [ValidateSet("prepare", "train")][string]$Phase,
    [ValidateSet(0, 25, 50, 75, 100)][int]$PercentSelf,
    [ValidateRange(1, 4)][int]$Round
)
$ErrorActionPreference="Stop"
$repo="C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-recursive-run"
$python="C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-and-reward-modelling-alt\.venv-gpu\Scripts\python.exe"
Set-Location $repo
$task="RecursiveStage3_${Phase}_${PercentSelf}_${Round}"
$arguments="scripts\run_recursive_stage3_worker.py $Phase --percent-self $PercentSelf --round $Round"
$action=New-ScheduledTaskAction -Execute $python -Argument $arguments -WorkingDirectory $repo
$trigger=New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5)
$principal=New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Days 2) -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 2)
Register-ScheduledTask -TaskName $task -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force|Out-Null
Start-ScheduledTask -TaskName $task;Start-Sleep -Seconds 3
Get-ScheduledTask -TaskName $task|Select-Object TaskName,State
