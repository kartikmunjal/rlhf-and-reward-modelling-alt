@echo off
setlocal
cd /d "C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-and-reward-modelling-alt"
if not exist logs mkdir logs
echo [%date% %time%] Starting v2 arithmetic SFT >> logs\ppo_grpo_v2_pilot.log
".venv-gpu\Scripts\python.exe" scripts\train_ppo_grpo_v2_sft.py >> logs\ppo_grpo_v2_pilot.log 2>&1
if errorlevel 1 (
  echo [%date% %time%] FAILED v2 arithmetic SFT >> logs\ppo_grpo_v2_pilot.log
  exit /b 1
)
echo [%date% %time%] Starting outcome-blind v2 feasibility pilot >> logs\ppo_grpo_v2_pilot.log
".venv-gpu\Scripts\python.exe" scripts\run_ppo_grpo_v2_pilot.py >> logs\ppo_grpo_v2_pilot.log 2>&1
set EXIT_CODE=%ERRORLEVEL%
echo [%date% %time%] V2 pilot exited with code %EXIT_CODE% >> logs\ppo_grpo_v2_pilot.log
exit /b %EXIT_CODE%
