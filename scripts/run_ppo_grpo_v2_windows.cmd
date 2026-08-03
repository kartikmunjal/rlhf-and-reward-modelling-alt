@echo off
setlocal
cd /d "C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-and-reward-modelling-alt"
if not exist logs mkdir logs
echo [%date% %time%] Starting PPO-GRPO v2 baseline >> logs\ppo_grpo_v2.log
".venv-gpu\Scripts\python.exe" scripts\run_ppo_grpo_v2.py --method baseline >> logs\ppo_grpo_v2.log 2>&1
if errorlevel 1 exit /b 1
for %%M in (ppo grpo) do (
  for %%S in (2025 2026 2027) do (
    echo Starting %%M seed %%S >> logs\ppo_grpo_v2.log
    ".venv-gpu\Scripts\python.exe" scripts\run_ppo_grpo_v2.py --method %%M --seed %%S --evaluate >> logs\ppo_grpo_v2.log 2>&1
    if errorlevel 1 (
      echo FAILED %%M seed %%S >> logs\ppo_grpo_v2.log
      exit /b 1
    )
    echo Finished %%M seed %%S >> logs\ppo_grpo_v2.log
  )
)
echo [%date% %time%] PPO-GRPO v2 complete >> logs\ppo_grpo_v2.log
exit /b 0
