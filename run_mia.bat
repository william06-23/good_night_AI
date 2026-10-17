@echo off
cd /d "%~dp0"
"C:\Users\William\William's_AI_project\torch_env\Scripts\python.exe" mia_chat\chat_cli.py
if errorlevel 1 pause
