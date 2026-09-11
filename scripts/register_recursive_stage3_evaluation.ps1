$ErrorActionPreference="Stop"
$repo="C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-recursive-run";$python="C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-and-reward-modelling-alt\.venv-gpu\Scripts\python.exe";Set-Location $repo
$action=New-ScheduledTaskAction -Execute $python -Argument "scripts\evaluate_recursive_stage3.py" -WorkingDirectory $repo
$trigger=New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5);$principal=New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest;$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Days 4) -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 2)
Register-ScheduledTask -TaskName "RecursiveStage3EvaluationV1" -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force|Out-Null;Start-ScheduledTask -TaskName "RecursiveStage3EvaluationV1";Start-Sleep -Seconds 3;Get-ScheduledTask -TaskName "RecursiveStage3EvaluationV1"|Select-Object TaskName,State
