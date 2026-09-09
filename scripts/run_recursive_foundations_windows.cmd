@echo off
cd /d "C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-recursive-run"
if not exist logs mkdir logs
echo [%date% %time%] Starting/resuming recursive foundations >> logs\recursive_foundations.log
"C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-and-reward-modelling-alt\.venv-gpu\Scripts\python.exe" scripts\train_recursive_foundations.py --phase all >> logs\recursive_foundations.log 2>&1
set EXIT_CODE=%ERRORLEVEL%
echo [%date% %time%] Foundations exited with code %EXIT_CODE% >> logs\recursive_foundations.log
exit /b %EXIT_CODE%
