"""Parallel local-shard search and global candidate merging."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from time import perf_counter

import numpy as np

from .certificates import top_indices
from .local_index import LocalShardIndex
from .router import SSFRRouter
from .shard_bounds import certify_cosine_shard_pruning
from .types import SearchResult


class DistributedSSFRSearch:
    def __init__(
        self,
        router: SSFRRouter,
        shard_indexes: dict[int, LocalShardIndex],
        *,
        max_workers: int | None = None,
        execution_mode: str = "auto",
    ) -> None:
        if not router.fitted:
            raise RuntimeError("router must be fitted")
        missing = set(range(router.shard_count)) - set(shard_indexes)
        if missing:
            raise ValueError(f"missing local indexes for shards: {sorted(missing)}")
        dimensions = {index.dimension for index in shard_indexes.values()}
        if dimensions != {router.dimension}:
            raise ValueError("all local indexes must match the router dimension")
        if execution_mode not in {"auto", "sequential", "threaded"}:
            raise ValueError("execution_mode must be auto, sequential, or threaded")
        self.router = router
        self.shard_indexes = shard_indexes
        self.max_workers = max_workers
        self.execution_mode = execution_mode
        self._executor: ThreadPoolExecutor | None = None

    def _should_use_threads(self, selected_shards: list[int]) -> bool:
        if self.execution_mode == "sequential":
            return False
        if self.execution_mode == "threaded":
            return len(selected_shards) > 1
        # NumPy and local HNSW calls are already native and very short. A Python
        # thread-pool dispatch costs more than the work for typical local shards.
        # Remote/I/O adapters should opt into execution_mode="threaded".
        return False

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        probe_shards: int | None = None,
        *,
        allowed_ids_by_shard: dict[int, np.ndarray] | None = None,
        local_top_k: int | None = None,
    ) -> SearchResult:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        started = perf_counter()
        route = self.router.route(query, probe_shards=probe_shards)
        routing_done = perf_counter()
        candidate_k = max(top_k, local_top_k or top_k)

        def run(shard_id: int) -> tuple[np.ndarray, np.ndarray]:
            allowed = None
            if allowed_ids_by_shard is not None:
                allowed = allowed_ids_by_shard.get(shard_id, np.empty(0, dtype=str))
            return self.shard_indexes[shard_id].search(
                query, candidate_k, allowed_ids=allowed
            )

        selected_shards = [int(value) for value in route.shard_ids]
        if self._should_use_threads(selected_shards):
            if self._executor is None:
                worker_count = self.max_workers or min(
                    32, max(1, len(selected_shards))
                )
                self._executor = ThreadPoolExecutor(max_workers=worker_count)
            local_results = list(self._executor.map(run, selected_shards))
        else:
            local_results = [run(shard_id) for shard_id in selected_shards]
        local_done = perf_counter()

        non_empty = [(ids, scores) for ids, scores in local_results if ids.size]
        if non_empty:
            all_ids = np.concatenate([item[0] for item in non_empty])
            all_scores = np.concatenate([item[1] for item in non_empty])
            selected = top_indices(all_scores, min(top_k, all_scores.size))
            result_ids = all_ids[selected]
            result_scores = all_scores[selected]
        else:
            exemplar = next(iter(self.shard_indexes.values())).ids
            dtype = str if exemplar is None else exemplar.dtype
            result_ids = np.empty(0, dtype=dtype)
            result_scores = np.empty(0, dtype=np.float64)
        merge_done = perf_counter()

        vector_certified = False
        if (
            result_scores.size >= top_k
            and self.router.config.distance_metric == "cosine"
            and self.router.shard_metadata is not None
        ):
            radii = np.asarray(
                [item.euclidean_radius for item in self.router.shard_metadata],
                dtype=np.float64,
            )
            vector_certified = certify_cosine_shard_pruning(
                route.shard_ids,
                float(result_scores[min(top_k, result_scores.size) - 1]),
                route.upper_bounds,
                radii,
            )
            route = replace(route, vector_pruning_certified=vector_certified)

        candidates = 0
        for shard_id in selected_shards:
            allowed = (
                None
                if allowed_ids_by_shard is None
                else allowed_ids_by_shard.get(shard_id, np.empty(0, dtype=str))
            )
            candidates += self.shard_indexes[shard_id].estimated_candidates(allowed)

        return SearchResult(
            item_ids=result_ids,
            scores=np.asarray(result_scores, dtype=np.float64),
            route=route,
            routing_latency_ms=(routing_done - started) * 1000.0,
            local_search_latency_ms=(local_done - routing_done) * 1000.0,
            merge_latency_ms=(merge_done - local_done) * 1000.0,
            total_latency_ms=(merge_done - started) * 1000.0,
            shards_accessed=len(selected_shards),
            candidate_vectors_evaluated=int(candidates),
        )


def exact_global_search(
    query: np.ndarray,
    embeddings: np.ndarray,
    ids: np.ndarray,
    top_k: int,
    *,
    allowed_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(embeddings, dtype=np.float64)
    vector = np.asarray(query, dtype=np.float64)
    identifiers = np.asarray(ids)
    if matrix.ndim != 2 or vector.shape != (matrix.shape[1],):
        raise ValueError("query dimension does not match embeddings")
    if identifiers.shape != (matrix.shape[0],):
        raise ValueError("ids must match embeddings")
    if allowed_mask is not None:
        mask = np.asarray(allowed_mask, dtype=bool)
        matrix = matrix[mask]
        identifiers = identifiers[mask]
    if matrix.shape[0] == 0:
        return identifiers[:0], np.empty(0, dtype=np.float64)
    scores = matrix @ vector
    selected = top_indices(scores, min(top_k, scores.size))
    return identifiers[selected], scores[selected]
