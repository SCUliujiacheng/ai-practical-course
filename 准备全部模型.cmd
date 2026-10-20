@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo 请先运行“首次安装.cmd”。
  pause
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0tools\prepare_models.py" --all %*
set "COURSE_EXIT=%ERRORLEVEL%"
pause
exit /b %COURSE_EXIT%
