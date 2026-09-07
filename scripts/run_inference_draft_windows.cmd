@echo off
cd /d "C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-and-reward-modelling-alt"
if not exist logs mkdir logs
echo [%date% %time%] Starting or resuming inference draft training>>logs\inference_draft_v1.log
".venv-gpu\Scripts\python.exe" scripts\train_gpt2_small_draft.py>>logs\inference_draft_v1.log 2>&1
set EXIT_CODE=%ERRORLEVEL%
echo [%date% %time%] Draft training exited with code %EXIT_CODE%>>logs\inference_draft_v1.log
exit /b %EXIT_CODE%
