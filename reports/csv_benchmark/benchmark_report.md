# SSFR benchmark report

Generated from measured runs. Seed: `42`.

## Configuration

```json
{
  "csv": "data/products.csv",
  "queries": "data/search_queries.csv",
  "products": 20,
  "shards": 8,
  "probe_values": [
    2,
    4,
    8
  ],
  "bands": [
    1,
    2,
    4
  ],
  "top_k": 5,
  "physical_vectors_loaded": true,
  "local_index_backend": "hnsw",
  "max_spectral_attempts": 0
}
```

## Results

| Method | Mean ms | P95 ms | Centroid recall | Vector recall | Certified | Fallback | Memory bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| global_exact | 0.0163 | 0.0317 | n/a | 1.0000 | 0.0000 | 0.0000 | 31120 |
| global_hnsw | 0.0252 | 0.0464 | n/a | 1.0000 | 0.0000 | 0.0000 | 31120 |
| exhaustive_centroid_plus_local_hnsw | 0.0805 | 0.1196 | n/a | 0.8267 | 1.0000 | 0.0000 | 55696 |
| hierarchical_plus_local_hnsw | 0.0830 | 0.1158 | n/a | 0.8400 | 0.0000 | 0.0000 | 64944 |
| ssfr_plus_local_hnsw | 0.0787 | 0.1315 | n/a | 0.8267 | 1.0000 | 0.0000 | 56016 |

## Triggered kill criteria

- Measured SSFR vector Recall@k was below 0.95.
