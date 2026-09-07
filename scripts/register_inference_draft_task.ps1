$ErrorActionPreference = "Stop"
$root = "C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-and-reward-modelling-alt"
$taskName = "InferenceDraftV1"
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing -and $existing.State -eq "Running") { throw "$taskName is already running" }
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/d /c `"$root\scripts\run_inference_draft_windows.cmd`"" -WorkingDirectory $root
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Days 3) -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Start-Sleep -Seconds 3
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName, State
Get-ScheduledTaskInfo -TaskName $taskName | Select-Object LastRunTime, LastTaskResult
