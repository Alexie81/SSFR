@echo off
setlocal
set "SSFR_ROOT=%~dp0"
set "SSFR_PYTHON=%SSFR_ROOT%.venv\Scripts\python.exe"
set "SSFR_INDEX=%SSFR_ROOT%artifacts\products_1m_semantic"

if not exist "%SSFR_INDEX%\catalog_manifest.json" (
  echo [EROARE] Indexul semantic pentru 1 milion de produse nu exista.
  echo Ruleaza mai intai: .\construieste_index_1m.cmd
  exit /b 1
)

"%SSFR_PYTHON%" -m ssfr.cli interactive ^
  --index "%SSFR_INDEX%" ^
  --top-only ^
  --top-k 20 ^
  --probe-shards 32 ^
  --page-size 10 ^
  --native-threads 1

exit /b %ERRORLEVEL%
