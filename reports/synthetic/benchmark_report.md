# SSFR benchmark report

Generated from measured runs. Seed: `42`.

## Configuration

```json
{
  "shards": 128,
  "dimensions": 64,
  "queries": 50,
  "probe_shards": 8,
  "bands": [
    4,
    8,
    16,
    32,
    64
  ],
  "low_rank": 16,
  "physical_vectors_loaded": false,
  "billion_scale_estimate": false,
  "estimated_catalog_items": 1000000,
  "estimated_only": true
}
```

## Results

| Method | Mean ms | P95 ms | Centroid recall | Vector recall | Certified | Fallback | Memory bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| exhaustive | 0.0097 | 0.0122 | 1.0000 | n/a | 0.0000 | 0.0000 | 65536 |
| random | 0.0079 | 0.0075 | 0.0650 | n/a | 0.0000 | 0.0000 | 0 |
| pca_low_rank | 0.0117 | 0.0131 | 0.8525 | n/a | 0.0000 | 0.0000 | 25088 |
| truncated_svd | 0.0119 | 0.0122 | 0.8650 | n/a | 0.0000 | 0.0000 | 25088 |
| hierarchical | 0.0173 | 0.0210 | 1.0000 | n/a | 0.0000 | 0.0000 | 72192 |
| ivf_centroid_probing | 0.0097 | 0.0107 | 1.0000 | n/a | 0.0000 | 0.0000 | 65536 |
| ssfr | 0.4923 | 0.6126 | 1.0000 | n/a | 1.0000 | 0.0000 | 330752 |
| ssfr_without_ordering | 0.5001 | 0.6006 | 1.0000 | n/a | 1.0000 | 0.0000 | 330752 |
| ssfr_without_certificate | 0.0218 | 0.0288 | 0.5700 | n/a | 0.0000 | 0.0000 | 12288 |

## Triggered kill criteria

- The mean Fourier band is close to the shard count.
- Measured SSFR routing was not faster than matrix multiplication.
- Spectral payload plus residuals exceeded centroid matrix memory.
