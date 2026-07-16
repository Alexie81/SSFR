"""Numerical and evaluation metrics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def normalize_vector(vector: np.ndarray, *, name: str = "vector") -> np.ndarray:
    result = np.asarray(vector, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    norm = float(np.linalg.norm(result))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError(f"{name} must have non-zero L2 norm")
    return result / norm


def normalize_rows(matrix: np.ndarray, *, name: str = "matrix") -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError(f"{name} contains a zero-norm row")
    return result / norms


def recall_at_k(found: Iterable[object], oracle: Iterable[object], k: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    found_values = list(found)[:k]
    oracle_values = list(oracle)[:k]
    denominator = min(k, len(oracle_values))
    if denominator == 0:
        return 1.0 if not found_values else 0.0
    return len(set(found_values) & set(oracle_values)) / denominator


def precision_at_k(found: Iterable[object], oracle: Iterable[object], k: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    found_values = list(found)[:k]
    if not found_values:
        return 1.0 if not list(oracle) else 0.0
    return len(set(found_values) & set(list(oracle)[:k])) / len(found_values)


def latency_summary(values_ms: Iterable[float]) -> dict[str, float]:
    values = np.asarray(list(values_ms), dtype=np.float64)
    if values.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def ordering_quality(ordered_centroids: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(ordered_centroids, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("ordered_centroids must be two-dimensional")
    if matrix.shape[0] < 2:
        return {
            "mean_consecutive_distance": 0.0,
            "median_consecutive_distance": 0.0,
            "total_variation": 0.0,
        }
    distances = np.linalg.norm(np.diff(matrix, axis=0), axis=1)
    return {
        "mean_consecutive_distance": float(distances.mean()),
        "median_consecutive_distance": float(np.median(distances)),
        "total_variation": float(distances.sum()),
    }
