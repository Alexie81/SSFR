from __future__ import annotations

import numpy as np

from ssfr.certificates import (
    certify_top_b,
    intervals_cover_exact,
    score_intervals,
)


def test_residual_intervals_cover_exact_scores(
    centroids: np.ndarray,
    rng: np.random.Generator,
) -> None:
    query = rng.normal(size=centroids.shape[1])
    query /= np.linalg.norm(query)
    spectrum = np.fft.fft(centroids, axis=0)
    truncated = np.zeros_like(spectrum)
    truncated[[0, 1, -1]] = spectrum[[0, 1, -1]]
    approximation = np.fft.ifft(truncated, axis=0).real
    residuals = np.linalg.norm(centroids - approximation, axis=1)
    approximate_scores = approximation @ query
    lower, upper = score_intervals(approximate_scores, residuals)
    assert intervals_cover_exact(centroids @ query, lower, upper)


def test_certificate_false_when_intervals_overlap() -> None:
    scores = np.array([0.9, 0.8, 0.7])
    lower = np.array([0.4, 0.3, 0.2])
    upper = np.array([1.0, 0.95, 0.9])
    _, certified, margin = certify_top_b(scores, lower, upper, 1)
    assert not certified
    assert margin < 0


def test_certificate_true_for_separated_intervals() -> None:
    scores = np.array([0.9, 0.4, 0.1])
    lower = np.array([0.8, 0.3, 0.0])
    upper = np.array([1.0, 0.5, 0.2])
    selected, certified, margin = certify_top_b(scores, lower, upper, 1)
    assert certified
    assert margin >= 0
    assert selected.tolist() == [0]
