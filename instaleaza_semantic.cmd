@echo off
setlocal
set "SSFR_ROOT=%~dp0"
set "SSFR_PYTHON=%SSFR_ROOT%.venv\Scripts\python.exe"

if not exist "%SSFR_PYTHON%" (
  echo [EROARE] Nu exista mediul Python: "%SSFR_PYTHON%"
  echo Creeaza-l mai intai cu: python -m venv .venv
  exit /b 1
)

echo [SSFR] Instalez backendul semantic local in mediul proiectului...
pushd "%SSFR_ROOT%"
"%SSFR_PYTHON%" -m pip install -e ".[embeddings]"
set "SSFR_INSTALL_STATUS=%ERRORLEVEL%"
popd
if not "%SSFR_INSTALL_STATUS%"=="0" exit /b %SSFR_INSTALL_STATUS%

nvidia-smi >nul 2>&1
if not errorlevel 1 (
  "%SSFR_PYTHON%" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"
  if errorlevel 1 (
    echo [SSFR] Exista GPU NVIDIA; instalez build-ul PyTorch CUDA 13.0...
    "%SSFR_PYTHON%" -m pip install --force-reinstall --no-deps ^
      torch==2.13.0+cu130 ^
      --index-url https://download.pytorch.org/whl/cu130
    if errorlevel 1 exit /b 1
  )
)

"%SSFR_PYTHON%" -c "import torch; print('[SSFR] PyTorch:', torch.__version__); print('[SSFR] CUDA disponibila:', torch.cuda.is_available()); print('[SSFR] Dispozitiv:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
exit /b %ERRORLEVEL%
