"""Offline calibration of the shard probe budget."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .certificates import top_indices
from .metrics import normalize_rows, recall_at_k


@dataclass(frozen=True)
class ProbeCalibrationResult:
    selected_probe_shards: int
    target_recall: float
    mean_recall_by_probe: dict[int, float]
    minimum_recall_by_probe: dict[int, float]


def calibrate_probe_count(
    *,
    embeddings: np.ndarray,
    assignments: np.ndarray,
    centroids: np.ndarray,
    validation_queries: np.ndarray,
    top_k: int,
    probe_values: tuple[int, ...],
    target_recall: float = 0.95,
) -> ProbeCalibrationResult:
    vectors = normalize_rows(embeddings, name="embeddings").astype(np.float32)
    queries = normalize_rows(validation_queries, name="validation_queries").astype(
        np.float32
    )
    shard_ids = np.asarray(assignments, dtype=np.int64)
    centers = normalize_rows(centroids, name="centroids").astype(np.float32)
    if shard_ids.shape != (vectors.shape[0],):
        raise ValueError("assignments must match embeddings")
    if centers.shape[1] != vectors.shape[1] or queries.shape[1] != vectors.shape[1]:
        raise ValueError("embedding, centroid, and query dimensions must match")
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")
    probes = tuple(
        sorted({int(value) for value in probe_values if 1 <= int(value) <= centers.shape[0]})
    )
    if not probes:
        raise ValueError("probe_values contains no valid shard count")

    values: dict[int, list[float]] = {probe: [] for probe in probes}
    for query in queries:
        exact_scores = vectors @ query
        oracle = top_indices(exact_scores, min(top_k, exact_scores.size))
        centroid_order = top_indices(centers @ query, centers.shape[0])
        for probe in probes:
            selected_shards = centroid_order[:probe]
            candidates = np.flatnonzero(np.isin(shard_ids, selected_shards))
            candidate_scores = exact_scores[candidates]
            selected = top_indices(candidate_scores, min(top_k, candidate_scores.size))
            found = candidates[selected]
            values[probe].append(recall_at_k(found, oracle, top_k))

    mean_recall = {
        probe: float(np.mean(recall_values))
        for probe, recall_values in values.items()
    }
    minimum_recall = {
        probe: float(np.min(recall_values))
        for probe, recall_values in values.items()
    }
    selected_probe = probes[-1]
    for probe in probes:
        if mean_recall[probe] >= target_recall:
            selected_probe = probe
            break
    return ProbeCalibrationResult(
        selected_probe_shards=selected_probe,
        target_recall=target_recall,
        mean_recall_by_probe=mean_recall,
        minimum_recall_by_probe=minimum_recall,
    )
