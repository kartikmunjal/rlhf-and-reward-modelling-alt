param(
    [string]$TaskName = "SummarizationFineTuneV1",
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$cmdPath = Join-Path $RepoRoot "scripts\run_summarization_finetune_windows.cmd"
if (-not (Test-Path $cmdPath)) { throw "Runner not found: $cmdPath" }
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$cmdPath`"" -WorkingDirectory $RepoRoot
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Write-Host "Started $TaskName in $RepoRoot"
