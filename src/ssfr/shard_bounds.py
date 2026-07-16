"""Conservative vector-level shard bounds."""

from __future__ import annotations

import numpy as np


def euclidean_min_distance_bound(
    query: np.ndarray,
    centroids: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    q = np.asarray(query, dtype=np.float64)
    centers = np.asarray(centroids, dtype=np.float64)
    radius = np.asarray(radii, dtype=np.float64)
    if centers.ndim != 2 or q.shape != (centers.shape[1],):
        raise ValueError("query dimension does not match centroids")
    if radius.shape != (centers.shape[0],):
        raise ValueError("radii dimension does not match shard count")
    return np.maximum(0.0, np.linalg.norm(centers - q, axis=1) - radius)


def cosine_vector_score_bounds(
    centroid_lower: np.ndarray,
    centroid_upper: np.ndarray,
    euclidean_radii: np.ndarray,
    query_norm: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(centroid_lower, dtype=np.float64)
    upper = np.asarray(centroid_upper, dtype=np.float64)
    radii = np.asarray(euclidean_radii, dtype=np.float64)
    if lower.shape != upper.shape or lower.shape != radii.shape:
        raise ValueError("centroid bounds and radii must have identical shapes")
    error = query_norm * radii
    return np.maximum(-1.0, lower - error), np.minimum(1.0, upper + error)


def angular_cosine_score_bounds(
    centroid_scores: np.ndarray,
    angular_radii: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.clip(np.asarray(centroid_scores, dtype=np.float64), -1.0, 1.0)
    radii = np.asarray(angular_radii, dtype=np.float64)
    if scores.shape != radii.shape:
        raise ValueError("centroid_scores and angular_radii must have identical shapes")
    angles = np.arccos(scores)
    best_angles = np.maximum(0.0, angles - radii)
    worst_angles = np.minimum(np.pi, angles + radii)
    return np.cos(worst_angles), np.cos(best_angles)


def certify_cosine_shard_pruning(
    selected_shards: np.ndarray,
    kth_candidate_score: float,
    centroid_upper_bounds: np.ndarray,
    euclidean_radii: np.ndarray,
    query_norm: float = 1.0,
) -> bool:
    upper = np.asarray(centroid_upper_bounds, dtype=np.float64)
    radii = np.asarray(euclidean_radii, dtype=np.float64)
    if upper.shape != radii.shape:
        raise ValueError("centroid_upper_bounds and euclidean_radii must have identical shapes")
    mask = np.ones(upper.size, dtype=bool)
    mask[np.asarray(selected_shards, dtype=np.int64)] = False
    if not np.any(mask):
        return True
    vector_upper = np.minimum(1.0, upper + query_norm * radii)
    return bool(float(np.max(vector_upper[mask])) <= float(kth_candidate_score))
