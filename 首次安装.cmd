@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
if exist "%~dp0runtime\python\python.exe" (
  "%~dp0runtime\python\python.exe" "%~dp0tools\setup_environment.py" %*
) else (
  py -3.12 "%~dp0tools\setup_environment.py" %*
)
set "COURSE_EXIT=%ERRORLEVEL%"
if not "%COURSE_EXIT%"=="0" echo 如未找到 Python，请安装官方 Python 3.12 x64：https://www.python.org/downloads/windows/
pause
exit /b %COURSE_EXIT%
