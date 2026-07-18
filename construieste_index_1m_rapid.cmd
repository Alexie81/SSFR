@echo off
setlocal
set "SSFR_ROOT=%~dp0"
set "SSFR_PYTHON=%SSFR_ROOT%.venv\Scripts\python.exe"
set "SSFR_CSV=%SSFR_ROOT%data\generated\products_1m.csv"
set "SSFR_INDEX=%SSFR_ROOT%artifacts\products_1m_fast"

if not exist "%SSFR_PYTHON%" (
  echo [EROARE] Nu exista mediul Python: "%SSFR_PYTHON%"
  exit /b 1
)

if not exist "%SSFR_CSV%" (
  echo [EROARE] Nu exista catalogul: "%SSFR_CSV%"
  exit /b 1
)

echo [SSFR] Build rapid lexical. Este pentru teste de viteza, nu pentru relevanta semantica.

"%SSFR_PYTHON%" -m ssfr.cli build ^
  --csv "%SSFR_CSV%" ^
  --output "%SSFR_INDEX%" ^
  --shards 256 ^
  --bands 8,16,32,64,128 ^
  --probe-shards 32 ^
  --embedding-provider fast-hash ^
  --embedding-dimension 64 ^
  --local-index auto ^
  --max-spectral-attempts 0 ^
  --streaming ^
  --batch-size 10000 ^
  --kmeans-epochs 1 ^
  --progress-every 50000 ^
  --native-threads 8

exit /b %ERRORLEVEL%
