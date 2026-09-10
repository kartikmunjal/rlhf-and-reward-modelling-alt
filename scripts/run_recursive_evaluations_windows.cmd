@echo off
cd /d "C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-recursive-run"
if not exist logs mkdir logs
echo [%date% %time%] Starting/resuming held-out generation >> logs\recursive_evaluations.log
"C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-and-reward-modelling-alt\.venv-gpu\Scripts\python.exe" scripts\generate_recursive_evaluations.py >> logs\recursive_evaluations.log 2>&1
set EXIT_CODE=%ERRORLEVEL%
echo [%date% %time%] Held-out generation exited with code %EXIT_CODE% >> logs\recursive_evaluations.log
exit /b %EXIT_CODE%
