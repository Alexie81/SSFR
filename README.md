# SSFR — SpectraShard Fourier Router

SSFR is an experimental, research-oriented router for distributed vector search. It
orders shard centroids, compresses them along the shard axis with a Fourier
transform, reconstructs approximate centroid scores at query time, and uses
deterministic residual bounds to either certify the selected top shards, expand the
spectral band, or fall back to exact centroid scoring.

The project also contains a complete CSV e-commerce workflow: validated import,
offline embeddings, shard construction, persistent local indexes, text search,
structured filters, exact-oracle evaluation, an API, and reproducible benchmarks.

> SSFR has not been demonstrated at one-billion-vector scale. Any billion-scale
> number produced by this repository is explicitly an estimate, not a physical
> benchmark.

## What is certified

- **Centroid-ranking certification** proves that the selected top-B centroid scores
  cannot be displaced by an unselected centroid under the stored Fourier residual
  bounds.
- **Vector-level shard-pruning certification** is separate. It additionally needs
  per-shard vector radii and a real top-k candidate threshold after local search.
- **Approximate routing** is returned only when exact fallback is disabled and no
  band certifies the result.
- **Exact fallback** computes every centroid score and is always reported.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .[dev,api,ann]
```

FAISS, HNSWlib, Sentence Transformers, and Parquet support are optional. The
e-commerce demo runs without a paid API or model download by using the deterministic
hash embedding provider and the exact NumPy local index.

## Quick start with the included CSV

```bash
python -m ssfr.cli build \
  --csv data/products.csv \
  --output artifacts/products \
  --shards 8 \
  --bands 1,2,4 \
  --embedding-provider hash \
  --local-index exact

python -m ssfr.cli search \
  --index artifacts/products \
  --query "adidași negri impermeabili pentru alergare pe munte" \
  --top-k 3 \
  --probe-shards 2
```

Structured filtering:

```bash
python -m ssfr.cli search \
  --index artifacts/products \
  --query "adidași pentru alergare" \
  --color negru \
  --price-max 500 \
  --in-stock-only \
  --filter-strategy pre \
  --top-k 5
```

## Tests, demo, benchmark, and API

```bash
pytest -v
python demos/csv_ecommerce_search.py --build --search
python benchmarks/compare_baselines.py --shards 128 --dimensions 64 --queries 50
python benchmarks/benchmark_csv_catalog.py --csv data/products.csv --queries data/search_queries.csv
uvicorn demos.api_demo:app
```

Benchmark scripts write only measured results to `reports/benchmark_report.json`,
`reports/benchmark_report.md`, and `reports/plots/`. CSV searches append their
measured evaluation to `reports/csv_search_evaluation.csv`.

## Main API

```python
from ssfr import SSFRConfig, SSFRRouter

router = SSFRRouter(
    SSFRConfig(
        spectral_bands=(8, 16, 32, 64),
        probe_shards=16,
        exact_fallback=True,
        ordering_method="recursive_pca",
        distance_metric="cosine",
        normalize_vectors=True,
    )
).fit(centroids)

route = router.route(query)
print(route.shard_ids, route.used_band, route.used_exact_fallback)
```

## Current limitations

- Spectral compressibility is data- and ordering-dependent.
- Recursive PCA ordering is an offline cost and dynamic updates can require a full
  reorder and rebuild.
- A centroid certificate is not a top-k vector certificate.
- The deterministic hash embedding provider is reproducible and useful for an
  offline demo, but it is not a substitute for a trained semantic model.
- HNSW and FAISS behavior depends on optional third-party packages.
- Large CSV support is designed around batched interfaces, but the reference CLI
  currently materializes metadata and embeddings for a single-machine prototype.
- Synthetic benchmarks do not establish production performance.

See [ARCHITECTURE.md](ARCHITECTURE.md), [RESEARCH_NOTE.md](RESEARCH_NOTE.md), and
[BENCHMARKS.md](BENCHMARKS.md) for the design, mathematical scope, and evaluation
guidance. For a Romanian, step-by-step walkthrough, use
[TESTING_GUIDE_RO.md](TESTING_GUIDE_RO.md).
