@echo off
setlocal
set "SSFR_ROOT=%~dp0"
set "SSFR_PYTHON=%SSFR_ROOT%.venv\Scripts\python.exe"
set "SSFR_INDEX=%SSFR_ROOT%artifacts\products"

if not exist "%SSFR_PYTHON%" (
  echo [EROARE] Nu exista mediul Python: "%SSFR_PYTHON%"
  echo Ruleaza mai intai instalarea proiectului.
  exit /b 1
)

if not exist "%SSFR_INDEX%\catalog_manifest.json" (
  echo [EROARE] Indexul nu exista: "%SSFR_INDEX%"
  echo Construieste mai intai indexul cu comanda ssfr build.
  exit /b 1
)

"%SSFR_PYTHON%" -m ssfr.cli interactive --index "%SSFR_INDEX%" %*
exit /b %ERRORLEVEL%
