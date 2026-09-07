$root = "C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-and-reward-modelling-alt"
Set-Location $root
Get-ScheduledTask -TaskName "InferenceDraftV1" | Select-Object TaskName, State
Get-ScheduledTaskInfo -TaskName "InferenceDraftV1" | Select-Object LastRunTime, LastTaskResult
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object CommandLine -Like "*train_gpt2_small_draft.py*" |
  Select-Object ProcessId, CreationDate, CommandLine
if (Test-Path ".\checkpoints\inference_serving_v1\gpt2_small_draft\progress.json") {
  Get-Content ".\checkpoints\inference_serving_v1\gpt2_small_draft\progress.json"
}
Get-Content ".\logs\inference_draft_v1.log" -Tail 20
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader
