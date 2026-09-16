@echo off
chcp 65001 >nul
REM ============================================================
REM  Math Tutor Agent - one-click start (daily use)
REM  Usage: double-click, or run from cmd: start.bat
REM  Starts Redis (short-term memory cache, optional) then the API server.
REM ============================================================
cd /d "%~dp0"

set PY=C:\Users\lenovo\.workbuddy\binaries\python\envs\default\Scripts\python.exe
if not exist "%PY%" set PY=python

set PYTHONPATH=%~dp0src
set HOST=127.0.0.1
set PORT=8787

REM ---- Redis（短期记忆缓存）：未在跑则拉起，已在跑则跳过 ----
set REDIS_EXE=%~dp0tools\redis\redis-server.exe
if exist "%REDIS_EXE%" (
  netstat -ano | findstr /C:":6379" | findstr /C:"LISTENING" >nul 2>&1
  if errorlevel 1 (
    echo [start.bat] Starting Redis on 127.0.0.1:6379 ...
    start "math-agent-redis" /min "%REDIS_EXE%" --port 6379 --bind 127.0.0.1 --save "" --appendonly no
  ) else (
    echo [start.bat] Redis already running on 6379, skip.
  )
) else (
  echo [start.bat] Redis not found under tools\redis\, skip. App falls back to in-memory cache.
)

echo [start.bat] PYTHONPATH=%PYTHONPATH%
echo [start.bat] Python: %PY%
echo [start.bat] Starting server at http://%HOST%:%PORT% ...
echo [start.bat] Press Ctrl+C to stop.
echo.

"%PY%" -m math_agent.api.server
pause
