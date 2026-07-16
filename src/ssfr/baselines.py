"""Reference routing baselines used by the benchmark suite."""

from __future__ import annotations

import numpy as np

from .certificates import top_indices
from .metrics import normalize_rows


class LowRankCentroidRouter:
    def __init__(self, centroids: np.ndarray, rank: int, *, centered: bool) -> None:
        matrix = np.asarray(centroids, dtype=np.float64)
        self.mean = matrix.mean(axis=0) if centered else np.zeros(matrix.shape[1])
        shifted = matrix - self.mean
        left, singular, right = np.linalg.svd(shifted, full_matrices=False)
        self.rank = min(max(1, int(rank)), singular.size)
        self.left_scaled = left[:, : self.rank] * singular[: self.rank]
        self.right = right[: self.rank]

    def route(self, query: np.ndarray, probe_shards: int) -> np.ndarray:
        vector = np.asarray(query, dtype=np.float64)
        scores = self.left_scaled @ (self.right @ vector) + self.mean @ vector
        return top_indices(scores, probe_shards)

    @property
    def memory_bytes(self) -> int:
        return int(self.left_scaled.nbytes + self.right.nbytes + self.mean.nbytes)


class HierarchicalCentroidRouter:
    def __init__(
        self,
        centroids: np.ndarray,
        *,
        branching_factor: int = 16,
        random_seed: int = 42,
    ) -> None:
        from sklearn.cluster import MiniBatchKMeans

        matrix = normalize_rows(centroids, name="centroids")
        group_count = min(max(1, int(np.ceil(np.sqrt(matrix.shape[0])))), branching_factor)
        group_count = min(group_count, matrix.shape[0])
        model = MiniBatchKMeans(
            n_clusters=group_count,
            random_state=random_seed,
            n_init=3,
            batch_size=max(64, group_count * 4),
        )
        self.assignments = model.fit_predict(matrix)
        self.group_centroids = normalize_rows(model.cluster_centers_, name="group centroids")
        self.centroids = matrix
        self.members = {
            group: np.flatnonzero(self.assignments == group) for group in range(group_count)
        }

    def route(self, query: np.ndarray, probe_shards: int) -> np.ndarray:
        vector = np.asarray(query, dtype=np.float64)
        group_order = top_indices(
            self.group_centroids @ vector, self.group_centroids.shape[0]
        )
        candidates: list[int] = []
        for group in group_order:
            candidates.extend(self.members[int(group)].tolist())
            if len(candidates) >= max(probe_shards * 2, probe_shards):
                break
        candidate_ids = np.asarray(candidates, dtype=np.int64)
        scores = self.centroids[candidate_ids] @ vector
        local = top_indices(scores, min(probe_shards, scores.size))
        return candidate_ids[local]

    @property
    def memory_bytes(self) -> int:
        return int(
            self.centroids.nbytes + self.group_centroids.nbytes + self.assignments.nbytes
        )


def exhaustive_centroid_route(
    centroids: np.ndarray, query: np.ndarray, probe_shards: int
) -> np.ndarray:
    return top_indices(np.asarray(centroids) @ np.asarray(query), probe_shards)


def random_shard_route(
    shard_count: int,
    probe_shards: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.choice(shard_count, size=probe_shards, replace=False).astype(np.int64)
