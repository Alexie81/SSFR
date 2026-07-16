"""Shard construction from product embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .metrics import normalize_rows
from .types import ShardBuildResult, ShardMetadata


def _repair_empty_clusters(
    assignments: np.ndarray,
    embeddings: np.ndarray,
    provisional_centers: np.ndarray,
    shard_count: int,
) -> np.ndarray:
    labels = assignments.copy()
    counts = np.bincount(labels, minlength=shard_count)
    for empty in np.flatnonzero(counts == 0):
        donor_candidates = np.flatnonzero(counts[labels] > 1)
        if donor_candidates.size == 0:
            raise RuntimeError("unable to repair an empty shard")
        donor_distances = np.linalg.norm(
            embeddings[donor_candidates] - provisional_centers[labels[donor_candidates]],
            axis=1,
        )
        point = int(donor_candidates[np.argmax(donor_distances)])
        old = int(labels[point])
        labels[point] = int(empty)
        counts[old] -= 1
        counts[empty] += 1
    return labels


def build_shards(
    embeddings: np.ndarray,
    shard_count: int,
    *,
    random_seed: int = 42,
    batch_size: int = 4096,
) -> ShardBuildResult:
    vectors = normalize_rows(np.asarray(embeddings), name="embeddings").astype(np.float32)
    if not 1 <= shard_count <= vectors.shape[0]:
        raise ValueError("shard_count must be between 1 and the number of embeddings")
    if shard_count == 1:
        assignments = np.zeros(vectors.shape[0], dtype=np.int64)
        provisional = vectors.mean(axis=0, keepdims=True)
    else:
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError as exc:
            raise RuntimeError("build_shards requires scikit-learn") from exc
        model = MiniBatchKMeans(
            n_clusters=shard_count,
            random_state=random_seed,
            batch_size=min(max(batch_size, shard_count * 4), max(vectors.shape[0], shard_count)),
            n_init=3,
            max_no_improvement=20,
            reassignment_ratio=0.01,
        )
        assignments = model.fit_predict(vectors).astype(np.int64)
        provisional = np.asarray(model.cluster_centers_, dtype=np.float32)
        assignments = _repair_empty_clusters(
            assignments, vectors, provisional, shard_count
        )

    centroids = np.zeros((shard_count, vectors.shape[1]), dtype=np.float64)
    counts = np.bincount(assignments, minlength=shard_count).astype(np.int64)
    for shard_id in range(shard_count):
        centroids[shard_id] = vectors[assignments == shard_id].mean(axis=0)
    centroids = normalize_rows(centroids, name="shard centroids")
    radii = np.zeros(shard_count, dtype=np.float64)
    for shard_id in range(shard_count):
        local = vectors[assignments == shard_id]
        radii[shard_id] = float(
            np.max(np.linalg.norm(local - centroids[shard_id], axis=1))
        )
    return ShardBuildResult(
        assignments=assignments,
        centroids=centroids,
        euclidean_radii=radii,
        item_counts=counts,
    )


def angular_radii(
    embeddings: np.ndarray,
    shard_result: ShardBuildResult,
) -> np.ndarray:
    vectors = normalize_rows(embeddings, name="embeddings")
    result = np.zeros(shard_result.centroids.shape[0], dtype=np.float64)
    for shard_id, centroid in enumerate(shard_result.centroids):
        local = vectors[shard_result.assignments == shard_id]
        cosine = np.clip(local @ centroid, -1.0, 1.0)
        result[shard_id] = float(np.max(np.arccos(cosine)))
    return result


def build_shard_metadata(
    shard_result: ShardBuildResult,
    *,
    index_root: str | Path = "local_indexes",
    angular: np.ndarray | None = None,
) -> list[ShardMetadata]:
    root = Path(index_root)
    metadata = []
    for shard_id in range(shard_result.centroids.shape[0]):
        metadata.append(
            ShardMetadata(
                shard_id=shard_id,
                item_count=int(shard_result.item_counts[shard_id]),
                centroid=shard_result.centroids[shard_id].copy(),
                euclidean_radius=float(shard_result.euclidean_radii[shard_id]),
                angular_radius=(
                    None if angular is None else float(np.asarray(angular)[shard_id])
                ),
                index_path=str(root / f"shard_{shard_id:05d}"),
            )
        )
    return metadata
