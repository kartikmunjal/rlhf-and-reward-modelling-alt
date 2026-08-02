@echo off
setlocal
cd /d "C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-and-reward-modelling-alt"
if not exist logs mkdir logs
echo [%date% %time%] Starting PPO-GRPO v1 >> logs\ppo_grpo_v1.log
for %%M in (ppo grpo) do (
  for %%S in (2025 2026 2027) do (
    echo [%date% %time%] Starting %%M seed %%S >> logs\ppo_grpo_v1.log
    ".venv-gpu\Scripts\python.exe" scripts\run_ppo_grpo_v1.py --method %%M --seed %%S --evaluate >> logs\ppo_grpo_v1.log 2>&1
    if errorlevel 1 (
      echo [%date% %time%] FAILED %%M seed %%S >> logs\ppo_grpo_v1.log
      exit /b 1
    )
    echo [%date% %time%] Finished %%M seed %%S >> logs\ppo_grpo_v1.log
  )
)
echo [%date% %time%] PPO-GRPO v1 complete >> logs\ppo_grpo_v1.log
exit /b 0
