@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0
if exist "%~dp0local_env.bat" call "%~dp0local_env.bat"
if exist "%~dp0mia_local.bat" call "%~dp0mia_local.bat"

rem CUDA 12 runtime DLLs for llama-cpp-python (14B GGUF)
set "PATH=%~dp0models\llama_cpp_cudart12;%PATH%"
if defined AI_TEXT_GAME_MODELS_DIR if exist "%AI_TEXT_GAME_MODELS_DIR%\llama_cpp_cudart12" (
  set "PATH=%AI_TEXT_GAME_MODELS_DIR%\llama_cpp_cudart12;%PATH%"
)
if defined MIA_MODELS_DIR if exist "%MIA_MODELS_DIR%\llama_cpp_cudart12" (
  set "PATH=%MIA_MODELS_DIR%\llama_cpp_cudart12;%PATH%"
)

if not defined PY set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo No Python found.
  echo Create .venv per README, or copy local_env.bat.example to local_env.bat and set PY.
  pause
  exit /b 1
)

echo Loading AI Text Game API on http://127.0.0.1:8000 ...
echo Using: %PY%
if defined AI_TEXT_GAME_MODELS_DIR (
  echo Models: %AI_TEXT_GAME_MODELS_DIR%
) else if defined MIA_MODELS_DIR (
  echo Models: %MIA_MODELS_DIR%
)
"%PY%" -m uvicorn ai_text_game.api:app --host 127.0.0.1 --port 8000
if errorlevel 1 pause
