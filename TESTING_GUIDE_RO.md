# Ghid explicit de înțelegere și testare SSFR

Acest ghid presupune Windows PowerShell și repository-ul deschis în:

```powershell
cd "D:\Alexie\algoritmi\SSFR - SpectraShard Fourier Router"
```

## 1. Ce face aplicația, pe scurt

Aplicația are două faze complet separate.

### Faza offline: construirea indexului

```text
products.csv
  -> validare și import
  -> text semantic pentru fiecare produs
  -> embedding normalizat
  -> grupare în sharduri
  -> centroid și rază pentru fiecare shard
  -> ordonarea centroizilor
  -> compresie Fourier și reziduuri
  -> index local persistent pentru fiecare shard
```

Această fază poate fi mai lentă, dar se execută o singură dată pentru o versiune a
catalogului. Rezultatul este salvat în `artifacts/products`.

### Faza online: o căutare

```text
query text
  -> query embedding
  -> SSFR estimează scorul tuturor centroizilor
  -> verifică certificatul top-B
  -> mărește banda Fourier dacă este necesar
  -> folosește fallback exact dacă certificatul nu apare
  -> caută numai în shardurile selectate
  -> unește rezultatele
  -> compară cu oracle-ul exact global
```

Indexul nu este reconstruit la fiecare query. Comanda `search` încarcă artefactele
deja create.

## 2. Ce înseamnă câmpurile importante

- `spectral_bands`: benzile Fourier încercate, în ordine crescătoare. O bandă mai
  mare păstrează mai multă informație, dar costă mai mult.
- `probe_shards`: numărul de sharduri în care se face căutarea locală.
- `used_band`: banda la care rezultatul a fost certificat sau ultima bandă încercată.
- `centroid_ranking_certified`: selecția top-B este demonstrată corectă pentru
  scorurile centroizilor.
- `vector_pruning_certified`: shardurile neaccesate sunt demonstrate irelevante și
  pentru pragul top-k de vectori. Acesta este un certificat separat și mai puternic.
- `exact_fallback`: routerul a calculat exact toate scorurile centroizilor deoarece
  benzile aproximative nu au certificat selecția.
- `Recall@k`: proporția rezultatelor oracle-ului exact global găsite de fluxul SSFR.

Un fallback exact nu este o eroare. Este mecanismul conservator prin care routerul
refuză să prezinte o aproximare necertificată drept certitudine.

## 3. Instalare

Prima dată:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,api,parquet]"
```

Dacă PowerShell blochează activarea mediului virtual, nu este obligatoriu să schimbi
politica sistemului. Poți folosi direct executabilul:

```powershell
.\.venv\Scripts\python.exe --version
```

Toate comenzile de mai jos folosesc această variantă explicită.

Backendul HNSW este opțional:

```powershell
.\.venv\Scripts\python.exe -m pip install hnswlib
```

Aplicația funcționează și fără HNSW, prin backendul exact NumPy.

## 4. Testele automate

Rulează toate testele:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Acestea verifică, între altele:

1. FFT + IFFT cu banda completă reconstruiește centroizii;
2. intervalele Fourier acoperă scorurile exacte;
3. certificatul nu este acceptat când intervalele se suprapun;
4. fallback-ul exact produce același top-B ca produsul matricial;
5. ordonările sunt permutări valide;
6. save/load păstrează rezultatele;
7. query-ul zero și dimensiunea greșită produc erori explicite;
8. benzile duplicate sau prea mari sunt ajustate;
9. rutarea batch este echivalentă cu rutarea individuală;
10. actualizarea DFT incrementală coincide cu rebuild-ul;
11. local index save/load funcționează;
12. căutarea în toate shardurile coincide cu oracle-ul global;
13. importul CSV, filtrele și API-ul sunt valide.

Un singur test:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_router.py::test_full_band_route_equals_exact -v
```

Teste plus coverage:

```powershell
.\.venv\Scripts\python.exe -m pytest --cov=ssfr --cov-report=term-missing
```

## 5. Construirea indexului din CSV

Catalogul demonstrativ este `data/products.csv`.

```powershell
.\.venv\Scripts\python.exe -m ssfr.cli build `
  --csv data/products.csv `
  --output artifacts/products `
  --shards 8 `
  --bands 1,2,4 `
  --probe-shards 2 `
  --embedding-provider hash `
  --embedding-dimension 384 `
  --local-index exact
```

Ce trebuie să observi:

- numărul de produse valide;
- dimensiunea embeddingurilor;
- numărul de sharduri;
- benzile efectiv acceptate;
- backendul indexurilor locale;
- calea artefactelor;
- timpul real de build.

Fișierele importante rezultate:

```text
artifacts/products/
  import_report.json
  products.parquet
  semantic_texts.jsonl
  embeddings.npy
  product_ids.npy
  shard_assignments.npy
  shard_centroids.npy
  shard_radii.npy
  catalog_manifest.json
  ssfr_router/
  local_indexes/
```

Verifică raportul de import:

```powershell
Get-Content artifacts\products\import_report.json
```

Verifică manifestul:

```powershell
Get-Content artifacts\products\catalog_manifest.json
```

Rulează încă o dată aceeași comandă. Câmpul `embedding_cache_hit` trebuie să fie
adevărat dacă CSV-ul și providerul nu s-au schimbat.

## 6. Căutare simplă

```powershell
.\.venv\Scripts\python.exe -m ssfr.cli search `
  --index artifacts/products `
  --query "adidași negri impermeabili pentru alergare pe munte" `
  --top-k 3 `
  --probe-shards 2
```

Raportul afișează:

- shardurile selectate;
- banda Fourier;
- certificatul de centroizi;
- certificatul de pruning la nivel de vectori;
- fallback-ul exact;
- latența routerului;
- latența indexurilor locale;
- timpul total;
- Recall@k față de oracle;
- produsele și scorurile lor.

Pentru ieșire JSON:

```powershell
.\.venv\Scripts\python.exe -m ssfr.cli search `
  --index artifacts/products `
  --query "telefon cu baterie mare și cameră bună" `
  --top-k 3 `
  --probe-shards 4 `
  --json
```

## 7. Testarea filtrelor

Pre-filter: produsele incompatibile sunt eliminate înainte de căutarea locală.

```powershell
.\.venv\Scripts\python.exe -m ssfr.cli search `
  --index artifacts/products `
  --query "adidași pentru alergare" `
  --color negru `
  --price-max 500 `
  --in-stock-only `
  --filter-strategy pre `
  --top-k 5 `
  --probe-shards 4
```

Post-filter: se caută semantic, apoi candidații sunt filtrați și bugetul este mărit
dacă este necesar.

```powershell
.\.venv\Scripts\python.exe -m ssfr.cli search `
  --index artifacts/products `
  --query "adidași pentru alergare" `
  --color negru `
  --price-max 500 `
  --in-stock-only `
  --filter-strategy post `
  --top-k 5 `
  --probe-shards 4
```

Compară timpul total, numărul de candidați și Recall@k. Pe catalogul de 20 de
produse, timpii sunt prea mici pentru concluzii de performanță, dar fluxul poate fi
verificat funcțional.

## 8. Demo-ul CSV dintr-o comandă

Construire și căutare:

```powershell
.\.venv\Scripts\python.exe demos\csv_ecommerce_search.py --build --search
```

Doar căutare pe indexul existent:

```powershell
.\.venv\Scripts\python.exe demos\csv_ecommerce_search.py `
  --search `
  --query "rochie verde elegantă pentru nuntă"
```

## 9. Demo sintetic cu 100.000 de produse

```powershell
.\.venv\Scripts\python.exe demos\ecommerce_demo.py `
  --items 100000 `
  --shards 256 `
  --dimensions 96 `
  --probe-shards 16 `
  --top-k 10
```

Acest demo creează fizic 100.000 de vectori sintetici în memorie. El validează
fluxul end-to-end, dar nu reprezintă dovadă pentru un catalog de un miliard.

Pentru un smoke test rapid:

```powershell
.\.venv\Scripts\python.exe demos\ecommerce_demo.py `
  --items 5000 `
  --shards 32 `
  --probe-shards 8
```

## 10. Benchmarkuri

Benchmarkul sintetic al routerelor:

```powershell
.\.venv\Scripts\python.exe benchmarks\compare_baselines.py `
  --items 1000000 `
  --shards 128 `
  --dimensions 64 `
  --queries 50 `
  --probe-shards 8 `
  --bands 4,8,16,32,64 `
  --output reports
```

`--items` este doar o estimare de dimensiune a catalogului în acest benchmark.
Vectorii creați fizic sunt centroizii și query-urile declarate.

Benchmarkul pe CSV:

```powershell
.\.venv\Scripts\python.exe benchmarks\benchmark_csv_catalog.py `
  --csv data/products.csv `
  --queries data/search_queries.csv `
  --shards 8 `
  --probe-values 2,4,8 `
  --bands 1,2,4 `
  --top-k 5 `
  --output reports
```

Rezultatele reale sunt salvate în:

```text
reports/benchmark_report.json
reports/benchmark_report.md
reports/plots/
reports/csv_search_evaluation.csv
```

Pentru HNSW, memoria raportată include array-urile Python persistente, dar nu poate
măsura portabil memoria internă a grafului alocată de biblioteca nativă.

Verifică în JSON secțiunea `kill_criteria`. Ea raportează explicit situații în care
SSFR nu este avantajos: bandă prea mare, fallback frecvent, latență mai slabă decât
matrix multiplication sau memorie spectrală prea mare.

## 11. API FastAPI

Pornește serverul:

```powershell
$env:SSFR_INDEX_PATH = "artifacts/products"
.\.venv\Scripts\python.exe -m uvicorn demos.api_demo:app `
  --host 127.0.0.1 `
  --port 8000
```

În alt PowerShell, verifică health:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Statistici:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/router/stats
```

Căutare text:

```powershell
$body = @{
  query = "laptop pentru programare și editare video"
  top_k = 3
  probe_shards = 4
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/search `
  -ContentType "application/json" `
  -Body $body
```

Documentația interactivă Swagger este la:

```text
http://127.0.0.1:8000/docs
```

Oprești serverul cu `Ctrl+C`.

## 12. Cum interpretezi corect rezultatele

Exemplu:

```text
used Fourier band: 4
centroid ranking certified: true
vector pruning certified: false
exact fallback: false
Recall@3: 0.6667
```

Interpretare:

- top-B centroizi este certificat corect;
- acest lucru nu garantează că toate produsele top-k globale se află în acele
  sharduri;
- certificatul de vectori nu a putut demonstra pruning-ul complet;
- routerul nu a folosit fallback exact;
- două dintre cele trei rezultate oracle au fost găsite.

Alt exemplu:

```text
centroid ranking certified: true
exact fallback: true
```

Înseamnă că rezultatul centroizilor este exact, dar aproximarea Fourier nu a produs
un certificat suficient la benzile configurate. Acest caz trebuie raportat, nu
ascuns.

## 13. Teste de eroare recomandate

CSV fără `description`:

```powershell
Set-Content bad.csv "product_id,title`nP1,Produs"
.\.venv\Scripts\python.exe -m ssfr.cli build `
  --csv bad.csv `
  --output artifacts/bad `
  --shards 1
```

Trebuie să primești o eroare explicită despre coloana obligatorie lipsă.

Număr imposibil de sharduri:

```powershell
.\.venv\Scripts\python.exe -m ssfr.cli build `
  --csv data/products.csv `
  --output artifacts/too-many `
  --shards 100
```

Trebuie să primești o eroare deoarece există numai 20 de produse.

Query gol poate fi testat direct:

```powershell
.\.venv\Scripts\python.exe -c `
  "from ssfr.catalog import CatalogIndex; c=CatalogIndex.load('artifacts/products'); c.search_text('')"
```

Trebuie să primești eroarea `query text cannot be empty`.

## 14. Ordinea recomandată pentru validarea ta

1. `pytest -v`;
2. build pe `data/products.csv`;
3. search fără filtre;
4. search cu pre-filter și post-filter;
5. inspectarea manifestelor și a raportului CSV;
6. benchmarkul CSV;
7. benchmarkul sintetic;
8. pornirea API-ului și o cerere `/search`;
9. opțional, demo-ul cu 100.000 de produse și HNSW.

Această ordine separă corectitudinea matematică de performanță și face mai ușor de
localizat orice problemă.
