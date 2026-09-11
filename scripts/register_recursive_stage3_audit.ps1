param(
  [string]$Repo = "C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-recursive-run",
  [string]$Python = "C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-and-reward-modelling-alt\.venv-gpu\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$taskName = "RecursiveStage3Audit"
$script = Join-Path $Repo "scripts\audit_recursive_stage3_training_labels.py"
$logDir = Join-Path $Repo "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdout = Join-Path $logDir "recursive_stage3_audit.stdout.log"
$stderr = Join-Path $logDir "recursive_stage3_audit.stderr.log"
$action = New-ScheduledTaskAction -Execute $Python -Argument ('"{0}"' -f $script) -WorkingDirectory $Repo
$trigger = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(2))
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Days 2) -StartWhenAvailable
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Write-Output "Started $taskName. Python output is visible through the task process; artifacts are written atomically."
