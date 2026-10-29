@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0
rem CUDA 12 runtime DLLs for llama-cpp-python (14B GGUF)
set "PATH=%~dp0models\llama_cpp_cudart12;%PATH%"
"%~dp0.venv\Scripts\python.exe" -m mia_chat.chat_cli
if errorlevel 1 pause
