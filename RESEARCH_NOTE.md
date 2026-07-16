# Research note

## Working hypothesis

Locality-preserving ordering may make the shard-centroid sequence spectrally
compressible. If a small Fourier band reconstructs every query's centroid-score
sequence accurately enough, SSFR can replace an `S × d` centroid matrix product
with a smaller spectral projection plus an IFFT.

The working novelty statement is deliberately qualified:

> To the best of our knowledge, SSFR is the first vector-shard routing approach
> that combines locality-preserving shard-centroid ordering, Fourier reconstruction
> of all shard scores, deterministic residual certificates, and adaptive
> spectral-band expansion.

This statement requires continuous prior-art review and is not a claim of absolute
novelty.

## Falsifiable questions

- Does recursive PCA ordering materially concentrate energy compared with identity
  and random order?
- Is spectral projection plus IFFT faster than the optimized centroid matrix
  product at realistic `S`, `d`, and batch sizes?
- Do deterministic intervals certify often enough to avoid exact fallback?
- Does probing fewer shards preserve vector Recall@k on real embeddings?
- Does spectral payload plus residual memory remain below the exact centroid
  representation?
- How expensive are reorder and rebuild under production update rates?

## Interpretation of negative results

The benchmark emits kill criteria when the required band is close to `S`, fallback
dominates, SSFR is slower than matrix multiplication, or spectral metadata exceeds
centroid memory. These are useful outcomes: they identify regimes where Fourier
routing should be reformulated or abandoned.

## Experimental scope

The included sample CSV validates the data path, not scale. Synthetic centroid
benchmarks isolate routing behavior but do not prove end-to-end production gains.
Any billion-item configuration is a capacity estimate unless one billion vectors
were physically indexed and queried.
