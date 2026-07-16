"""Fourier utilities for centroid compression and score reconstruction."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def maximum_band(shard_count: int) -> int:
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    return shard_count // 2


def sanitize_bands(bands: Iterable[int], shard_count: int) -> tuple[int, ...]:
    limit = maximum_band(shard_count)
    adjusted = {min(max(int(band), 0), limit) for band in bands}
    if not adjusted:
        adjusted.add(limit)
    return tuple(sorted(adjusted))


def frequency_indices(shard_count: int, band: int) -> np.ndarray:
    limit = maximum_band(shard_count)
    current = min(max(int(band), 0), limit)
    low = np.arange(0, current + 1, dtype=np.int64)
    high_start = max(shard_count - current, 1)
    high = np.arange(high_start, shard_count, dtype=np.int64)
    return np.unique(np.concatenate((low, high))).astype(np.int64)


def truncate_spectrum(spectrum: np.ndarray, band: int) -> np.ndarray:
    values = np.asarray(spectrum)
    if values.ndim not in {1, 2}:
        raise ValueError("spectrum must be one- or two-dimensional")
    truncated = np.zeros_like(values)
    indices = frequency_indices(values.shape[0], band)
    truncated[indices] = values[indices]
    return truncated


def reconstruct_centroids(spectrum: np.ndarray, band: int) -> np.ndarray:
    return np.fft.ifft(truncate_spectrum(spectrum, band), axis=0).real


def residual_norms(centroids: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    original = np.asarray(centroids, dtype=np.float64)
    approximation = np.asarray(reconstructed, dtype=np.float64)
    if original.shape != approximation.shape:
        raise ValueError("centroids and reconstructed centroids must have identical shapes")
    return np.linalg.norm(original - approximation, axis=1)


def reconstruct_scores(
    payload: np.ndarray,
    indices: np.ndarray,
    query: np.ndarray,
    shard_count: int,
) -> np.ndarray:
    spectrum = np.zeros(shard_count, dtype=np.complex128)
    spectrum[np.asarray(indices, dtype=np.int64)] = np.asarray(payload) @ np.asarray(query)
    return np.fft.ifft(spectrum).real


def spectral_energy_fraction(spectrum: np.ndarray, band: int) -> float:
    values = np.asarray(spectrum)
    total = float(np.sum(np.abs(values) ** 2))
    if total <= np.finfo(np.float64).eps:
        return 1.0
    indices = frequency_indices(values.shape[0], band)
    return float(np.sum(np.abs(values[indices]) ** 2) / total)
