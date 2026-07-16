"""Locality-preserving centroid orderings."""

from __future__ import annotations

import numpy as np


def validate_order(order: np.ndarray, shard_count: int) -> np.ndarray:
    result = np.asarray(order, dtype=np.int64)
    if result.shape != (shard_count,):
        raise ValueError(f"order must have shape ({shard_count},)")
    if not np.array_equal(np.sort(result), np.arange(shard_count, dtype=np.int64)):
        raise ValueError("order must be a permutation of all shard indices")
    return result


def identity_order(centroids: np.ndarray) -> np.ndarray:
    return np.arange(np.asarray(centroids).shape[0], dtype=np.int64)


def random_order(centroids: np.ndarray, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).permutation(np.asarray(centroids).shape[0]).astype(np.int64)


def _principal_direction(points: np.ndarray) -> np.ndarray | None:
    centered = points - points.mean(axis=0, keepdims=True)
    if not np.any(centered):
        return None
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= np.finfo(np.float64).eps:
        return None
    direction = right[0]
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0:
        direction = -direction
    return direction


def recursive_pca_order(centroids: np.ndarray, leaf_size: int = 8) -> np.ndarray:
    matrix = np.asarray(centroids, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("centroids must be two-dimensional")
    if leaf_size < 1:
        raise ValueError("leaf_size must be at least 1")

    def recurse(indices: np.ndarray) -> list[int]:
        if indices.size <= 1:
            return indices.tolist()
        direction = _principal_direction(matrix[indices])
        if direction is None:
            return sorted(indices.tolist())
        projections = matrix[indices] @ direction
        local_order = np.lexsort((indices, projections))
        sorted_indices = indices[local_order]
        if indices.size <= leaf_size:
            return sorted_indices.tolist()
        midpoint = indices.size // 2
        return recurse(sorted_indices[:midpoint]) + recurse(sorted_indices[midpoint:])

    return np.asarray(recurse(np.arange(matrix.shape[0], dtype=np.int64)), dtype=np.int64)


def nearest_neighbor_chain_order(centroids: np.ndarray) -> np.ndarray:
    matrix = np.asarray(centroids, dtype=np.float64)
    count = matrix.shape[0]
    if count <= 1:
        return np.arange(count, dtype=np.int64)
    remaining = np.ones(count, dtype=bool)
    current = int(np.argmin(matrix[:, 0]))
    route = [current]
    remaining[current] = False
    while len(route) < count:
        candidates = np.flatnonzero(remaining)
        distances = np.linalg.norm(matrix[candidates] - matrix[current], axis=1)
        current = int(candidates[np.argmin(distances)])
        route.append(current)
        remaining[current] = False
    return np.asarray(route, dtype=np.int64)


def spectral_seriation_order(centroids: np.ndarray) -> np.ndarray:
    matrix = np.asarray(centroids, dtype=np.float64)
    count = matrix.shape[0]
    if count <= 2:
        return np.arange(count, dtype=np.int64)
    if count > 4096:
        raise ValueError("spectral seriation is limited to 4096 shards in the Python prototype")
    squared = np.sum((matrix[:, None, :] - matrix[None, :, :]) ** 2, axis=2)
    nonzero = squared[squared > 0]
    scale = float(np.median(nonzero)) if nonzero.size else 1.0
    affinity = np.exp(-squared / max(2.0 * scale, np.finfo(np.float64).eps))
    np.fill_diagonal(affinity, 0.0)
    laplacian = np.diag(affinity.sum(axis=1)) - affinity
    _, eigenvectors = np.linalg.eigh(laplacian)
    fiedler = eigenvectors[:, 1]
    return np.lexsort((np.arange(count), fiedler)).astype(np.int64)


def hierarchical_cluster_order(centroids: np.ndarray) -> np.ndarray:
    matrix = np.asarray(centroids, dtype=np.float64)
    if matrix.shape[0] <= 2:
        return np.arange(matrix.shape[0], dtype=np.int64)
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import pdist
    except ImportError as exc:  # pragma: no cover - scipy is a core dependency
        raise RuntimeError("hierarchical ordering requires scipy") from exc
    tree = linkage(pdist(matrix, metric="euclidean"), method="average")
    return leaves_list(tree).astype(np.int64)


def build_order(centroids: np.ndarray, method: str, seed: int = 42) -> np.ndarray:
    methods = {
        "identity": identity_order,
        "recursive_pca": recursive_pca_order,
        "nearest_neighbor_chain": nearest_neighbor_chain_order,
        "spectral_seriation": spectral_seriation_order,
        "hierarchical": hierarchical_cluster_order,
        "hierarchical_cluster": hierarchical_cluster_order,
    }
    if method == "random":
        return random_order(centroids, seed=seed)
    if method not in methods:
        choices = ", ".join(sorted((*methods, "random")))
        raise ValueError(f"unknown ordering method '{method}'; choose one of: {choices}")
    return methods[method](centroids)
