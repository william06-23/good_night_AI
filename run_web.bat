@echo off
cd /d "%~dp0"

echo Starting AI Text Game API in a new window...
echo (Close any old "AI Text Game API" window first so the new live-feed code loads.)
start "AI Text Game API" "%~dp0run_api.bat"

echo Waiting for API at http://127.0.0.1:8000 ...
set /a _tries=0
:wait_api
set /a _tries+=1
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/health' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if %ERRORLEVEL%==0 goto api_ready
if %_tries% GEQ 60 (
  echo API did not become ready in time. Check the "AI Text Game API" window for errors.
  pause
  exit /b 1
)
timeout /t 1 /nobreak >nul
goto wait_api

:api_ready
echo API is up.
powershell -NoProfile -Command "try { $j = (Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/health' -UseBasicParsing -TimeoutSec 2).Content | ConvertFrom-Json; if (-not $j.live_feed) { Write-Host 'WARNING: API has no live_feed flag — you may still be on an old process. Close the API window and re-run.' -ForegroundColor Yellow } else { Write-Host 'API live_feed OK.' -ForegroundColor Green } } catch { }"

echo Starting Vite on http://127.0.0.1:5173 ...
cd /d "%~dp0web"

if not exist "node_modules\.bin\vite.cmd" (
  echo Local Vite missing. Running npm install in web\ ...
  call npm install
  if errorlevel 1 (
    echo npm install failed.
    pause
    exit /b 1
  )
)

if not exist "node_modules\.bin\vite.cmd" (
  echo Still no web\node_modules\.bin\vite.cmd after npm install.
  pause
  exit /b 1
)

REM Open the UI explicitly (vite --open is unreliable with some npm shims).
start "" "http://127.0.0.1:5173/"

REM Call the local binary directly — some npm shims omit node_modules\.bin from PATH.
call "node_modules\.bin\vite.cmd" --host 127.0.0.1 --port 5173
if errorlevel 1 pause
