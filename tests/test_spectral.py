from __future__ import annotations

import numpy as np

from ssfr.spectral import (
    frequency_indices,
    maximum_band,
    reconstruct_centroids,
    sanitize_bands,
)


def test_full_band_fft_ifft_reconstructs_centroids(
    centroids: np.ndarray,
) -> None:
    spectrum = np.fft.fft(centroids, axis=0)
    reconstructed = reconstruct_centroids(spectrum, maximum_band(centroids.shape[0]))
    assert np.max(np.abs(reconstructed - centroids)) < 1e-9


def test_full_band_indices_cover_even_and_odd_lengths() -> None:
    for shard_count in (7, 8, 9, 10):
        indices = frequency_indices(shard_count, maximum_band(shard_count))
        assert np.array_equal(indices, np.arange(shard_count))


def test_duplicate_and_oversized_bands_are_adjusted() -> None:
    assert sanitize_bands((2, 2, 99, 4), 8) == (2, 4)


def test_dc_band_contains_only_dc() -> None:
    assert np.array_equal(frequency_indices(8, 0), np.array([0]))
