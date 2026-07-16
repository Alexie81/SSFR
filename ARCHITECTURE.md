# Architecture

SSFR separates offline index construction from query-time routing.

```text
CSV -> validation -> semantic text -> embeddings -> MiniBatchKMeans
    -> shard centroids/radii -> locality-preserving order -> FFT payloads/residuals
    -> persistent router + persistent local indexes

query text -> embedding -> adaptive SSFR route -> parallel local search
    -> global merge -> structured filtering/rerun -> exact-oracle evaluation
```

## Core invariants

`order[position]` is the original shard ID at that ordered position.
`inverse_order[shard_id]` is its Fourier-axis position. The FFT is taken over the
ordered centroid matrix, never over the embedding dimension.

For every configured band, the router stores:

- the unique frequency indices containing DC, low positive frequencies, and their
  negative-frequency counterparts;
- the complex Fourier rows at those indices;
- one L2 reconstruction residual per original shard.

At query time it computes `F_k @ q`, inserts those values into a zeroed score
spectrum, applies IFFT, and restores original shard order.

## Certification boundaries

The centroid score interval follows directly from Cauchy–Schwarz:

```text
|q·c_i - q·ĉ_i| <= ||q||₂ ||c_i-ĉ_i||₂
```

Top-B is certified only when the smallest selected lower bound is no smaller than
the largest unselected upper bound.

Vector-level pruning uses a different bound. For normalized cosine search, any
vector within Euclidean radius `rho_i` of a centroid has score at most
`centroid_upper_i + ||q|| rho_i`. An unselected shard is certified prunable only
after local search has produced a real kth-candidate score above every such upper
bound.

## Persistence

Router artifacts include exact centroids because exact fallback is part of the
reference implementation. A production deployment can place that fallback
representation on a different tier. Every router directory has a SHA-256 manifest.

Catalog artifacts store metadata in Parquet when available, embeddings in NPY,
assignments and radii as arrays, one router directory, and one local-index directory
per shard.

## Dynamic updates

Adding, deleting, splitting, or merging shards performs a full rebuild. A centroid
replacement can use an experimental incremental DFT update when shard count and
order remain fixed:

```text
F'_k = F_k + Δc exp(-2πikn/S)
```

Residuals are still recomputed after that spectrum update.

## C++ port boundary

The first C++ target should contain:

1. contiguous complex payloads grouped by band;
2. batched payload/query matrix multiplication;
3. an FFT backend abstraction (pocketfft, FFTW, or MKL);
4. allocation-free interval and top-B selection;
5. a stable binary manifest shared with Python;
6. pybind11 bindings around fit-artifact loading and route/route-batch.

Ordering, CSV import, and experiment orchestration can remain in Python initially.
