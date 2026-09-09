@echo off
cd /d "C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-recursive-run"
if not exist logs mkdir logs
echo [%date% %time%] Waiting for recursive foundations >> logs\recursive_stage1.log
:WAIT_FOUNDATIONS
"C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-and-reward-modelling-alt\.venv-gpu\Scripts\python.exe" scripts\check_recursive_foundations.py
set GATE_CODE=%ERRORLEVEL%
if "%GATE_CODE%"=="0" goto RUN_STAGE1
if "%GATE_CODE%"=="1" goto FAILED_GATE
timeout /t 60 /nobreak >nul
goto WAIT_FOUNDATIONS
:RUN_STAGE1
echo [%date% %time%] Foundations passed; starting/resuming Stage 1 >> logs\recursive_stage1.log
"C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-and-reward-modelling-alt\.venv-gpu\Scripts\python.exe" scripts\train_recursive_stage1.py >> logs\recursive_stage1.log 2>&1
set EXIT_CODE=%ERRORLEVEL%
echo [%date% %time%] Stage 1 exited with code %EXIT_CODE% >> logs\recursive_stage1.log
exit /b %EXIT_CODE%
:FAILED_GATE
echo [%date% %time%] Foundations gate failed; Stage 1 not started >> logs\recursive_stage1.log
exit /b 1
