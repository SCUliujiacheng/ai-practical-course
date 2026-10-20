@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
if not exist "%~dp0runtime\python\python.exe" (
  echo 缺少离线运行时，请先在线完成“准备离线恢复包.cmd”。
  pause
  exit /b 1
)
echo 请先关闭 Jupyter、Notebook 内核及案例进程。本操作会保留原环境备份。
"%~dp0runtime\python\python.exe" "%~dp0tools\repair_environment.py" --rebuild %*
set "COURSE_EXIT=%ERRORLEVEL%"
pause
exit /b %COURSE_EXIT%
