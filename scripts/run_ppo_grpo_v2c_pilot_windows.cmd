@echo off
setlocal
cd /d "C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-and-reward-modelling-alt"
if not exist logs mkdir logs
echo [%date% %time%] Starting outcome-blind v2c feasibility pilot >> logs\ppo_grpo_v2c_pilot.log
".venv-gpu\Scripts\python.exe" scripts\run_ppo_grpo_v2c_pilot.py >> logs\ppo_grpo_v2c_pilot.log 2>&1
set EXIT_CODE=%ERRORLEVEL%
echo [%date% %time%] V2c pilot exited with code %EXIT_CODE% >> logs\ppo_grpo_v2c_pilot.log
exit /b %EXIT_CODE%
