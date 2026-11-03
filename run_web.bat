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

REM Start Vite first (strictPort in vite.config — fails if 5173 is already taken).
REM Open the browser only after the port answers, so we don't get a dead 5173 tab
REM plus a second Vite on 5174.
start "AI Text Game Vite" /D "%~dp0web" cmd /c "node_modules\.bin\vite.cmd --host 127.0.0.1 --port 5173"

echo Waiting for Vite at http://127.0.0.1:5173 ...
set /a _vtries=0
:wait_vite
set /a _vtries+=1
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:5173/' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -lt 500) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if %ERRORLEVEL%==0 goto vite_ready
if %_vtries% GEQ 30 (
  echo Vite did not become ready on 5173. Close any old Vite window using that port and re-run.
  pause
  exit /b 1
)
timeout /t 1 /nobreak >nul
goto wait_vite

:vite_ready
start "" "http://127.0.0.1:5173/"
echo UI opened. Leave the "AI Text Game Vite" and "AI Text Game API" windows open.
pause
