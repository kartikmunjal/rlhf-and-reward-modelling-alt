$ErrorActionPreference="Stop"
$repo="C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-recursive-run";Set-Location $repo
$python="C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-and-reward-modelling-alt\.venv-gpu\Scripts\python.exe"
& $python -m py_compile "scripts\generate_recursive_evaluations.py"
$action=New-ScheduledTaskAction -Execute "cmd.exe" -Argument '/c ""C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-recursive-run\scripts\run_recursive_evaluations_windows.cmd""'
$trigger=New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5)
$principal=New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Days 2) -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 2)
Register-ScheduledTask -TaskName "RecursiveEvaluationsV1" -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force|Out-Null
Start-ScheduledTask -TaskName "RecursiveEvaluationsV1";Start-Sleep -Seconds 3
Get-ScheduledTask -TaskName "RecursiveEvaluationsV1"|Select-Object TaskName,State
