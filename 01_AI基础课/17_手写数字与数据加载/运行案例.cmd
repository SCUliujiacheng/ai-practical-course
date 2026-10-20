@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
"..\..\.venv\Scripts\python.exe" "..\..\tools\run_case.py" "%~dp0solution.py"
set "COURSE_EXIT=%ERRORLEVEL%"
if not "%COURSE_EXIT%"=="0" echo 运行失败，请查看错误信息和根目录故障排查说明。
pause
exit /b %COURSE_EXIT%
