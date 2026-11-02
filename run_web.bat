@echo off
cd /d "%~dp0"

echo Starting Mia API in a new window...
start "Mia API" "%~dp0run_api.bat"

echo Waiting for API at http://127.0.0.1:8000 ...
set /a _tries=0
:wait_api
set /a _tries+=1
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/health' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if %ERRORLEVEL%==0 goto api_ready
if %_tries% GEQ 60 (
  echo API did not become ready in time. Check the "Mia API" window for errors.
  pause
  exit /b 1
)
timeout /t 1 /nobreak >nul
goto wait_api

:api_ready
echo API is up. Starting Vite — browser opens at http://127.0.0.1:5173
cd /d "%~dp0web"
call npm run dev -- --host 127.0.0.1 --port 5173 --open
if errorlevel 1 pause
