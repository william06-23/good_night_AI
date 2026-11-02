@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0
rem CUDA 12 runtime DLLs for llama-cpp-python (14B GGUF)
set "PATH=%~dp0models\llama_cpp_cudart12;%PATH%"
if defined MIA_MODELS_DIR if exist "%MIA_MODELS_DIR%\llama_cpp_cudart12" (
  set "PATH=%MIA_MODELS_DIR%\llama_cpp_cudart12;%PATH%"
)

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo No .venv found. Create it per README, or set PY to your Python with torch.
  pause
  exit /b 1
)

echo Loading Mia API on http://127.0.0.1:8000 (models load on first start)...
"%PY%" -m uvicorn mia_chat.api:app --host 127.0.0.1 --port 8000
if errorlevel 1 pause
