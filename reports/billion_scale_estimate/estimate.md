# Billion-scale capacity estimate

> Estimate only. No one-billion-vector corpus was physically indexed.

- Items: 1,000,000,000
- Shards: 16,384
- Mean items/shard: 61,035.2
- Probed shards: 32
- Routed item universe: 1,953,125.0
- Shard fan-out reduction: 512.0×
- Exact centroid multiply-adds: 12,582,912
- Spectral projection complex terms: 197,376
- Approximate IFFT operations: 1,146,880
- Float32 centroid matrix: 48.00 MiB
- Float64 centroid matrix used by the strict Python prototype: 96.00 MiB
- Complex64 spectral payload: 1.51 MiB
- Complex128 spectral payload used by the strict Python prototype: 3.01 MiB
- Float32 residuals: 0.44 MiB

## Latency model from supplied measurements

- Router: 2.971 ms
- Local + network waves: 9.500 ms
- Merge: 0.500 ms
- Estimated end-to-end: 12.971 ms
