@echo off
cd /d "%~dp0"

echo Starting Mia API in a new window (models load there)...
start "Mia Rizz API" "%~dp0run_api.bat"

cd /d "%~dp0web"
echo Starting Vite — browser opens at http://127.0.0.1:5173
echo If the page says API offline, wait for the Mia API window to finish loading.
call npm run dev -- --host 127.0.0.1 --port 5173 --open
if errorlevel 1 pause
