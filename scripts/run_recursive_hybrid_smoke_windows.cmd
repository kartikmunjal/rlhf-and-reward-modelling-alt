@echo off
cd /d "C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-recursive-run"
if not exist logs mkdir logs
echo [%date% %time%] Starting hybrid smoke>>logs\hybrid_smoke.out.log
"C:\Users\Kunal Munjal\Desktop\Kartik\rlhf-and-reward-modelling-alt\.venv-gpu\Scripts\python.exe" scripts\run_recursive_hybrid_smoke.py >>logs\hybrid_smoke.out.log 2>>logs\hybrid_smoke.err.log
set EXIT_CODE=%ERRORLEVEL%
echo [%date% %time%] Exited %EXIT_CODE%>>logs\hybrid_smoke.out.log
echo %EXIT_CODE%>logs\hybrid_smoke.exit_code
exit /b %EXIT_CODE%
