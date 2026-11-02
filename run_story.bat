@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0
rem CUDA 12 runtime DLLs for llama-cpp-python (14B GGUF)
set "PATH=%~dp0models\llama_cpp_cudart12;%PATH%"
if defined MIA_MODELS_DIR if exist "%MIA_MODELS_DIR%\llama_cpp_cudart12" (
  set "PATH=%MIA_MODELS_DIR%\llama_cpp_cudart12;%PATH%"
)

echo AI Text Game CLI — no browser / Vite needed.

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo No .venv found. Create it per README, or set PY to your Python with torch.
  pause
  exit /b 1
)

rem Models: default models\ next to this bat, or set MIA_MODELS_DIR
"%PY%" -m mia_chat.story_cli
if errorlevel 1 pause
