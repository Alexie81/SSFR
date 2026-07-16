"""Deterministic score intervals and top-B certification."""

from __future__ import annotations

import numpy as np


def top_indices(scores: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if not 1 <= count <= values.size:
        raise ValueError(f"count must be between 1 and {values.size}")
    if count == values.size:
        candidates = np.arange(values.size, dtype=np.int64)
    else:
        candidates = np.argpartition(values, -count)[-count:]
    order = np.lexsort((candidates, -values[candidates]))
    return candidates[order].astype(np.int64)


def score_intervals(
    approximate_scores: np.ndarray,
    residuals: np.ndarray,
    query_norm: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(approximate_scores, dtype=np.float64)
    errors = np.asarray(residuals, dtype=np.float64)
    if scores.shape != errors.shape:
        raise ValueError("approximate_scores and residuals must have identical shapes")
    if query_norm < 0 or not np.isfinite(query_norm):
        raise ValueError("query_norm must be finite and non-negative")
    radius = query_norm * errors
    return scores - radius, scores + radius


def certify_top_b(
    approximate_scores: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    count: int,
) -> tuple[np.ndarray, bool, float]:
    scores = np.asarray(approximate_scores, dtype=np.float64)
    lower = np.asarray(lower_bounds, dtype=np.float64)
    upper = np.asarray(upper_bounds, dtype=np.float64)
    if scores.shape != lower.shape or scores.shape != upper.shape:
        raise ValueError("scores and bounds must have identical shapes")
    selected = top_indices(scores, count)
    if count == scores.size:
        return selected, True, float("inf")
    mask = np.ones(scores.size, dtype=bool)
    mask[selected] = False
    selected_floor = float(np.min(lower[selected]))
    outside_ceiling = float(np.max(upper[mask]))
    margin = selected_floor - outside_ceiling
    return selected, bool(margin >= 0.0), margin


def intervals_cover_exact(
    exact_scores: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    tolerance: float = 1e-12,
) -> bool:
    exact = np.asarray(exact_scores, dtype=np.float64)
    return bool(
        np.all(exact >= np.asarray(lower_bounds) - tolerance)
        and np.all(exact <= np.asarray(upper_bounds) + tolerance)
    )
