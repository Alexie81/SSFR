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
| exhaustive | 0.0064 | 0.0070 | 1.0000 | n/a | 0.0000 | 0.0000 | 65536 |
| random | 0.0060 | 0.0078 | 0.0650 | n/a | 0.0000 | 0.0000 | 0 |
| pca_low_rank | 0.0099 | 0.0112 | 0.8525 | n/a | 0.0000 | 0.0000 | 25088 |
| truncated_svd | 0.0087 | 0.0090 | 0.8650 | n/a | 0.0000 | 0.0000 | 25088 |
| hierarchical | 0.0222 | 0.0311 | 1.0000 | n/a | 0.0000 | 0.0000 | 72192 |
| ivf_centroid_probing | 0.0114 | 0.0144 | 1.0000 | n/a | 0.0000 | 0.0000 | 65536 |
| ssfr | 0.0816 | 0.1767 | 1.0000 | n/a | 1.0000 | 0.0000 | 81920 |
| ssfr_without_ordering | 0.0635 | 0.0852 | 1.0000 | n/a | 1.0000 | 0.0000 | 81920 |
| ssfr_without_certificate | 0.0127 | 0.0211 | 0.5700 | n/a | 0.0000 | 0.0000 | 8192 |

## Triggered kill criteria

- The mean Fourier band is close to the shard count.
- Measured SSFR routing was not faster than matrix multiplication.
