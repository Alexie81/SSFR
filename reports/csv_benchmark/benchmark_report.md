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
  "local_index_backend": "hnsw"
}
```

## Results

| Method | Mean ms | P95 ms | Centroid recall | Vector recall | Certified | Fallback | Memory bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| global_exact | 0.0415 | 0.0601 | n/a | 1.0000 | 0.0000 | 0.0000 | 31120 |
| global_hnsw | 0.0676 | 0.0937 | n/a | 1.0000 | 0.0000 | 0.0000 | 31120 |
| exhaustive_centroid_plus_local_hnsw | 0.1986 | 0.3464 | n/a | 0.8267 | 1.0000 | 0.0000 | 55696 |
| hierarchical_plus_local_hnsw | 0.2436 | 0.5587 | n/a | 0.8400 | 0.0000 | 0.0000 | 64944 |
| ssfr_plus_local_hnsw | 1.7820 | 2.6814 | n/a | 0.8267 | 1.0000 | 0.0000 | 154320 |

## Triggered kill criteria

- Measured SSFR vector Recall@k was below 0.95.
- SSFR end-to-end latency did not improve on exhaustive centroid routing.
