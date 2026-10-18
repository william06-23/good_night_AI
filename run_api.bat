@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0
rem CUDA 12 runtime DLLs for llama-cpp-python (14B GGUF)
set "PATH=C:\Users\William\William's_AI_project\llama_cpp_cudart12;%PATH%"
echo Loading Mia API on http://127.0.0.1:8000 (models load on first start)...
"C:\Users\William\William's_AI_project\torch_env\Scripts\python.exe" -m uvicorn mia_chat.api:app --host 127.0.0.1 --port 8000
if errorlevel 1 pause
