@echo off
setlocal
set "SSFR_ROOT=%~dp0"
set "SSFR_PYTHON=%SSFR_ROOT%.venv\Scripts\python.exe"
set "SSFR_CSV=%SSFR_ROOT%data\generated\products_1m.csv"
set "SSFR_INDEX=%SSFR_ROOT%artifacts\products_1m"

if not exist "%SSFR_PYTHON%" (
  echo [EROARE] Nu exista mediul Python: "%SSFR_PYTHON%"
  exit /b 1
)

if not exist "%SSFR_CSV%" (
  echo [EROARE] Nu exista catalogul: "%SSFR_CSV%"
  echo Genereaza-l cu tools\generate_million_products.py.
  exit /b 1
)

echo [SSFR] Se construieste indexul fizic pentru 1.000.000 de produse.
echo [SSFR] Operatia poate dura mai multe minute si poate folosi cativa GB RAM.

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
