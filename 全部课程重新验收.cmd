@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" "%~dp0tools\verify_course.py" --mode all --report personal_validation.json
set "COURSE_EXIT=%ERRORLEVEL%"
pause
exit /b %COURSE_EXIT%
