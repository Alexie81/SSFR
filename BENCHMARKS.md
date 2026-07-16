# Benchmarks

The benchmark scripts never ship hardcoded performance claims. They measure the
current machine and write JSON, Markdown, and plots under `reports/`.

## Synthetic comparison

```bash
python benchmarks/compare_baselines.py \
  --items 1000000 \
  --shards 1024 \
  --dimensions 128 \
  --queries 100 \
  --probe-shards 16
```

Compared methods:

1. exhaustive centroid matrix product;
2. random shard selection;
3. PCA low-rank approximation;
4. truncated SVD;
5. hierarchical centroid routing;
6. IVF-style exhaustive centroid probing;
7. SSFR with recursive PCA ordering;
8. SSFR without ordering;
9. SSFR without certification/adaptive fallback.

The `--items` value is recorded only as a catalog-size estimate; the synthetic
router benchmark physically creates shard centroids and queries, not that many item
vectors.

## Real CSV benchmark

```bash
python benchmarks/benchmark_csv_catalog.py \
  --csv data/products.csv \
  --queries data/search_queries.csv \
  --shards 8 \
  --probe-values 2,4,8 \
  --bands 1,2,4
```

It compares global exact search, global HNSW when installed, exhaustive centroid
routing plus local search, hierarchical routing plus local search, and SSFR plus
local search.

## Required interpretation

Inspect `kill_criteria` in the generated report. A benchmark where SSFR loses is not
rewritten or omitted. The sample catalog is too small for meaningful latency
conclusions; it exists to validate correctness and reporting.
