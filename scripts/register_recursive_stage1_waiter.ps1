$ErrorActionPreference = "Stop"
$repo = "C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-recursive-run"
Set-Location $repo
$python = "C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-and-reward-modelling-alt\.venv-gpu\Scripts\python.exe"
& $python -m py_compile "scripts\train_recursive_stage1.py" "scripts\check_recursive_foundations.py"
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument '/c ""C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-recursive-run\scripts\wait_then_run_recursive_stage1_windows.cmd""'
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Days 5) -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 2)
Register-ScheduledTask -TaskName "RecursiveStage1V1" -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName "RecursiveStage1V1"
Start-Sleep -Seconds 3
Get-ScheduledTask -TaskName "RecursiveStage1V1" | Select-Object TaskName,State
