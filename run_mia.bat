@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0
"C:\Users\William\William's_AI_project\torch_env\Scripts\python.exe" -m mia_chat.chat_cli
if errorlevel 1 pause
