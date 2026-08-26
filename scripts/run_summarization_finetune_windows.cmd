@echo off
setlocal
cd /d "%~dp0\.."
if not exist logs mkdir logs
echo [%date% %time%] Starting/resuming summarization SFT+DPO pipeline >> logs\summarization_finetune_v1.log
".venv-gpu\Scripts\python.exe" -u scripts\run_summarization_finetune_pipeline.py >> logs\summarization_finetune_v1.log 2>&1
set EXIT_CODE=%ERRORLEVEL%
echo [%date% %time%] Pipeline exited with code %EXIT_CODE% >> logs\summarization_finetune_v1.log
exit /b %EXIT_CODE%
