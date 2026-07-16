"""Core SSFR router implementation."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Any

import numpy as np

from .certificates import certify_top_b, score_intervals, top_indices
from .metrics import normalize_rows, normalize_vector, ordering_quality
from .ordering import build_order, validate_order
from .spectral import (
    frequency_indices,
    reconstruct_centroids,
    reconstruct_scores,
    residual_norms,
    sanitize_bands,
    spectral_energy_fraction,
)
from .types import RouteResult, ShardMetadata, SSFRConfig


class SSFRRouter:
    """Adaptive Fourier router with deterministic centroid-score certificates."""

    def __init__(self, config: SSFRConfig | None = None) -> None:
        self.config = config or SSFRConfig()
        self._fitted = False
        self.centroids: np.ndarray
        self.order: np.ndarray
        self.inverse_order: np.ndarray
        self.ordered_centroids: np.ndarray
        self.shard_metadata: list[ShardMetadata] | None = None
        self.bands: tuple[int, ...] = ()
        self.frequency_map: dict[int, np.ndarray] = {}
        self.spectral_payloads: dict[int, np.ndarray] = {}
        self.residuals: dict[int, np.ndarray] = {}
        self._full_spectrum: np.ndarray | None = None
        self._rfft_payload: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def shard_count(self) -> int:
        self._require_fitted()
        return int(self.centroids.shape[0])

    @property
    def dimension(self) -> int:
        self._require_fitted()
        return int(self.centroids.shape[1])

    def fit(
        self,
        centroids: np.ndarray,
        shard_metadata: list[ShardMetadata] | None = None,
        order: np.ndarray | None = None,
    ) -> "SSFRRouter":
        matrix = np.asarray(centroids, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
            raise ValueError("centroids must have shape (shard_count, dimension)")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("centroids contain non-finite values")
        if self.config.normalize_vectors:
            matrix = normalize_rows(matrix, name="centroids")
        elif np.any(np.linalg.norm(matrix, axis=1) <= np.finfo(np.float64).eps):
            raise ValueError("centroids contains a zero-norm row")

        if shard_metadata is not None and len(shard_metadata) != matrix.shape[0]:
            raise ValueError("shard_metadata length must match the number of centroids")

        route_order = (
            validate_order(order, matrix.shape[0])
            if order is not None
            else build_order(matrix, self.config.ordering_method, self.config.random_seed)
        )
        route_order = validate_order(route_order, matrix.shape[0])
        inverse = np.empty_like(route_order)
        inverse[route_order] = np.arange(route_order.size, dtype=np.int64)

        self.centroids = np.ascontiguousarray(matrix)
        self.order = route_order
        self.inverse_order = inverse
        self.ordered_centroids = np.ascontiguousarray(matrix[route_order])
        self.shard_metadata = shard_metadata
        self.bands = sanitize_bands(self.config.spectral_bands, matrix.shape[0])
        self._full_spectrum = np.fft.fft(self.ordered_centroids, axis=0)
        self._refresh_spectral_artifacts()
        self._full_spectrum = None
        self._fitted = True
        return self

    def _refresh_spectral_artifacts(self) -> None:
        if self._full_spectrum is None:
            raise RuntimeError("full spectrum is unavailable")
        rfft_size = self.ordered_centroids.shape[0] // 2 + 1
        full_rfft = np.ascontiguousarray(self._full_spectrum[:rfft_size])
        full_band = self.ordered_centroids.shape[0] // 2
        partial_bands = [band for band in self.bands if band < full_band]
        maximum_stored_band = max(partial_bands, default=-1)
        self._rfft_payload = (
            np.ascontiguousarray(full_rfft[: maximum_stored_band + 1])
            if maximum_stored_band >= 0
            else np.empty((0, self.ordered_centroids.shape[1]), dtype=np.complex128)
        )
        self.frequency_map = {}
        self.spectral_payloads = {}
        self.residuals = {}
        for band in self.bands:
            indices = np.arange(band + 1, dtype=np.int64)
            self.frequency_map[band] = indices
            if band >= full_band:
                self.spectral_payloads[band] = np.empty(
                    (0, self.ordered_centroids.shape[1]), dtype=np.complex128
                )
                self.residuals[band] = np.zeros(
                    self.ordered_centroids.shape[0], dtype=np.float64
                )
                continue
            self.spectral_payloads[band] = self._rfft_payload[: band + 1]
            truncated = np.zeros_like(full_rfft)
            truncated[: band + 1] = full_rfft[: band + 1]
            reconstructed = np.fft.irfft(
                truncated, n=self.ordered_centroids.shape[0], axis=0
            )
            ordered_residuals = residual_norms(self.ordered_centroids, reconstructed)
            original_residuals = np.empty_like(ordered_residuals)
            original_residuals[self.order] = ordered_residuals
            self.residuals[band] = original_residuals

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("SSFRRouter must be fitted before use")

    def _prepare_query(self, query: np.ndarray) -> tuple[np.ndarray, float]:
        self._require_fitted()
        vector = np.asarray(query, dtype=np.float64)
        if vector.shape != (self.dimension,):
            raise ValueError(
                f"query must have shape ({self.dimension},), received {vector.shape}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("query contains non-finite values")
        norm = float(np.linalg.norm(vector))
        if norm <= np.finfo(np.float64).eps:
            raise ValueError("query must have non-zero L2 norm")
        if self.config.normalize_vectors:
            return normalize_vector(vector, name="query"), 1.0
        return vector, norm

    def _approximate_scores(self, query: np.ndarray, band: int) -> np.ndarray:
        if band >= self.shard_count // 2:
            return self.centroids @ np.asarray(query, dtype=np.float64)
        half_spectrum = np.zeros(self.shard_count // 2 + 1, dtype=np.complex128)
        half_spectrum[: band + 1] = self.spectral_payloads[band] @ query
        ordered = np.fft.irfft(
            half_spectrum,
            n=self.shard_count,
        )
        original = np.empty_like(ordered)
        original[self.order] = ordered
        return original

    def route(
        self,
        query: np.ndarray,
        probe_shards: int | None = None,
    ) -> RouteResult:
        started = perf_counter()
        vector, query_norm = self._prepare_query(query)
        count = self.config.probe_shards if probe_shards is None else int(probe_shards)
        if not 1 <= count <= self.shard_count:
            raise ValueError(f"probe_shards must be between 1 and {self.shard_count}")

        last: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None = None
        full_band = self.shard_count // 2
        if self._rfft_payload is None:
            raise RuntimeError("spectral payload is unavailable")
        half_spectrum = np.zeros(self.shard_count // 2 + 1, dtype=np.complex128)
        previous_band = -1
        for band in self.bands:
            if band >= full_band:
                exact_scores = self.centroids @ vector
                selected = top_indices(exact_scores, count)
                return RouteResult(
                    shard_ids=selected,
                    approximate_scores=exact_scores,
                    lower_bounds=exact_scores.copy(),
                    upper_bounds=exact_scores.copy(),
                    used_band=band,
                    centroid_ranking_certified=True,
                    vector_pruning_certified=False,
                    used_exact_fallback=False,
                    latency_ms=(perf_counter() - started) * 1000.0,
                    route_mode="full_band_exact_fast_path",
                )
            start_frequency = previous_band + 1
            if start_frequency <= band:
                half_spectrum[start_frequency : band + 1] = (
                    self._rfft_payload[start_frequency : band + 1] @ vector
                )
            ordered_scores = np.fft.irfft(
                half_spectrum,
                n=self.shard_count,
            )
            scores = np.empty_like(ordered_scores)
            scores[self.order] = ordered_scores
            lower, upper = score_intervals(scores, self.residuals[band], query_norm)
            selected, certified, _ = certify_top_b(scores, lower, upper, count)
            last = selected, scores, lower, upper, band
            previous_band = band
            if certified:
                return RouteResult(
                    shard_ids=selected,
                    approximate_scores=scores,
                    lower_bounds=lower,
                    upper_bounds=upper,
                    used_band=band,
                    centroid_ranking_certified=True,
                    vector_pruning_certified=False,
                    used_exact_fallback=False,
                    latency_ms=(perf_counter() - started) * 1000.0,
                    route_mode="spectral_certified",
                )

        if self.config.exact_fallback:
            exact_scores = self.centroids @ vector
            selected = top_indices(exact_scores, count)
            return RouteResult(
                shard_ids=selected,
                approximate_scores=exact_scores,
                lower_bounds=exact_scores.copy(),
                upper_bounds=exact_scores.copy(),
                used_band=self.bands[-1],
                centroid_ranking_certified=True,
                vector_pruning_certified=False,
                used_exact_fallback=True,
                latency_ms=(perf_counter() - started) * 1000.0,
                route_mode="exact_fallback",
            )

        if last is None:  # defensive; fit always creates at least one band
            raise RuntimeError("no spectral band is available")
        selected, scores, lower, upper, band = last
        return RouteResult(
            shard_ids=selected,
            approximate_scores=scores,
            lower_bounds=lower,
            upper_bounds=upper,
            used_band=band,
            centroid_ranking_certified=False,
            vector_pruning_certified=False,
            used_exact_fallback=False,
            latency_ms=(perf_counter() - started) * 1000.0,
            route_mode="spectral_approximate",
        )

    def exact_route(
        self,
        query: np.ndarray,
        probe_shards: int | None = None,
    ) -> RouteResult:
        started = perf_counter()
        vector, _ = self._prepare_query(query)
        count = self.config.probe_shards if probe_shards is None else int(probe_shards)
        if not 1 <= count <= self.shard_count:
            raise ValueError(f"probe_shards must be between 1 and {self.shard_count}")
        scores = self.centroids @ vector
        selected = top_indices(scores, count)
        return RouteResult(
            shard_ids=selected,
            approximate_scores=scores,
            lower_bounds=scores.copy(),
            upper_bounds=scores.copy(),
            used_band=0,
            centroid_ranking_certified=True,
            vector_pruning_certified=False,
            used_exact_fallback=True,
            latency_ms=(perf_counter() - started) * 1000.0,
            route_mode="exact_route",
        )

    def route_batch(
        self,
        queries: np.ndarray,
        probe_shards: int | None = None,
    ) -> list[RouteResult]:
        self._require_fitted()
        matrix = np.asarray(queries, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.dimension:
            raise ValueError(f"queries must have shape (batch, {self.dimension})")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("queries contain non-finite values")
        count = self.config.probe_shards if probe_shards is None else int(probe_shards)
        if not 1 <= count <= self.shard_count:
            raise ValueError(f"probe_shards must be between 1 and {self.shard_count}")
        norms = np.linalg.norm(matrix, axis=1)
        if np.any(norms <= np.finfo(np.float64).eps):
            raise ValueError("every query must have non-zero L2 norm")
        if self.config.normalize_vectors:
            matrix = matrix / norms[:, None]
            error_norms = np.ones(matrix.shape[0], dtype=np.float64)
        else:
            error_norms = norms

        started = perf_counter()
        batch_size = matrix.shape[0]
        results: list[RouteResult | None] = [None] * batch_size
        unresolved = np.arange(batch_size, dtype=np.int64)
        last_values: dict[
            int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]
        ] = {}

        if self._rfft_payload is None:
            raise RuntimeError("spectral payload is unavailable")
        half_spectra = np.zeros(
            (batch_size, self.shard_count // 2 + 1), dtype=np.complex128
        )
        previous_band = -1
        full_band = self.shard_count // 2
        for band in self.bands:
            if unresolved.size == 0:
                break
            if band >= full_band:
                exact_scores = matrix[unresolved] @ self.centroids.T
                elapsed_per_query = (perf_counter() - started) * 1000.0 / batch_size
                for local_position, query_id in enumerate(unresolved):
                    scores = exact_scores[local_position]
                    selected = top_indices(scores, count)
                    results[int(query_id)] = RouteResult(
                        shard_ids=selected,
                        approximate_scores=scores.copy(),
                        lower_bounds=scores.copy(),
                        upper_bounds=scores.copy(),
                        used_band=band,
                        centroid_ranking_certified=True,
                        vector_pruning_certified=False,
                        used_exact_fallback=False,
                        latency_ms=elapsed_per_query,
                        route_mode="full_band_exact_fast_path",
                    )
                unresolved = np.empty(0, dtype=np.int64)
                break
            active_queries = matrix[unresolved]
            start_frequency = previous_band + 1
            coefficients = (
                self._rfft_payload[start_frequency : band + 1] @ active_queries.T
            )
            half_spectra[
                unresolved, start_frequency : band + 1
            ] = coefficients.T
            ordered_scores = np.fft.irfft(
                half_spectra[unresolved],
                n=self.shard_count,
                axis=1,
            )
            scores = np.empty_like(ordered_scores)
            scores[:, self.order] = ordered_scores
            errors = error_norms[unresolved, None] * self.residuals[band][None, :]
            lower = scores - errors
            upper = scores + errors

            next_unresolved: list[int] = []
            elapsed_per_query = (perf_counter() - started) * 1000.0 / batch_size
            for local_position, query_id in enumerate(unresolved):
                selected, certified, _ = certify_top_b(
                    scores[local_position],
                    lower[local_position],
                    upper[local_position],
                    count,
                )
                last_values[int(query_id)] = (
                    selected,
                    scores[local_position].copy(),
                    lower[local_position].copy(),
                    upper[local_position].copy(),
                    band,
                )
                if certified:
                    results[int(query_id)] = RouteResult(
                        shard_ids=selected,
                        approximate_scores=scores[local_position].copy(),
                        lower_bounds=lower[local_position].copy(),
                        upper_bounds=upper[local_position].copy(),
                        used_band=band,
                        centroid_ranking_certified=True,
                        vector_pruning_certified=False,
                        used_exact_fallback=False,
                        latency_ms=elapsed_per_query,
                        route_mode="spectral_certified",
                    )
                else:
                    next_unresolved.append(int(query_id))
            unresolved = np.asarray(next_unresolved, dtype=np.int64)
            previous_band = band

        if unresolved.size and self.config.exact_fallback:
            exact_scores = matrix[unresolved] @ self.centroids.T
            elapsed_per_query = (perf_counter() - started) * 1000.0 / batch_size
            for local_position, query_id in enumerate(unresolved):
                scores = exact_scores[local_position]
                selected = top_indices(scores, count)
                results[int(query_id)] = RouteResult(
                    shard_ids=selected,
                    approximate_scores=scores.copy(),
                    lower_bounds=scores.copy(),
                    upper_bounds=scores.copy(),
                    used_band=self.bands[-1],
                    centroid_ranking_certified=True,
                    vector_pruning_certified=False,
                    used_exact_fallback=True,
                    latency_ms=elapsed_per_query,
                    route_mode="exact_fallback",
                )
        elif unresolved.size:
            elapsed_per_query = (perf_counter() - started) * 1000.0 / batch_size
            for query_id in unresolved:
                selected, scores, lower, upper, band = last_values[int(query_id)]
                results[int(query_id)] = RouteResult(
                    shard_ids=selected,
                    approximate_scores=scores,
                    lower_bounds=lower,
                    upper_bounds=upper,
                    used_band=band,
                    centroid_ranking_certified=False,
                    vector_pruning_certified=False,
                    used_exact_fallback=False,
                    latency_ms=elapsed_per_query,
                    route_mode="spectral_approximate",
                )
        return [result for result in results if result is not None]

    def memory_report(self) -> dict[str, int]:
        self._require_fitted()
        payload = 0 if self._rfft_payload is None else self._rfft_payload.nbytes
        residual = sum(value.nbytes for value in self.residuals.values())
        ordering = self.order.nbytes + self.inverse_order.nbytes
        full = self.centroids.nbytes
        return {
            "full_centroid_matrix_bytes": int(full),
            "spectral_payload_bytes": int(payload),
            "residual_bytes": int(residual),
            "ordering_bytes": int(ordering),
            "router_bytes_without_exact_centroids": int(payload + residual + ordering),
            "router_bytes_with_exact_fallback": int(payload + residual + ordering + full),
        }

    def spectral_energy_report(self) -> dict[int, float]:
        self._require_fitted()
        full_spectrum = (
            self._full_spectrum
            if self._full_spectrum is not None
            else np.fft.fft(self.ordered_centroids, axis=0)
        )
        return {
            band: spectral_energy_fraction(full_spectrum, band) for band in self.bands
        }

    def ordering_report(self) -> dict[str, float]:
        self._require_fitted()
        return ordering_quality(self.ordered_centroids)

    def update_centroid(
        self,
        shard_id: int,
        centroid: np.ndarray,
        *,
        incremental: bool = False,
    ) -> dict[str, float | str]:
        self._require_fitted()
        if not 0 <= shard_id < self.shard_count:
            raise IndexError("shard_id is outside the router")
        started = perf_counter()
        value = np.asarray(centroid, dtype=np.float64)
        if value.shape != (self.dimension,):
            raise ValueError(f"centroid must have shape ({self.dimension},)")
        if self.config.normalize_vectors:
            value = normalize_vector(value, name="centroid")
        if incremental:
            position = int(self.inverse_order[shard_id])
            delta = value - self.ordered_centroids[position]
            frequencies = np.arange(self.shard_count, dtype=np.float64)
            phase = np.exp(-2j * np.pi * frequencies * position / self.shard_count)
            if self._full_spectrum is None:
                self._full_spectrum = np.fft.fft(self.ordered_centroids, axis=0)
            self._full_spectrum += phase[:, None] * delta[None, :]
            self.centroids[shard_id] = value
            self.ordered_centroids[position] = value
            self._refresh_spectral_artifacts()
            self._full_spectrum = None
            mode = "incremental_dft"
        else:
            updated = self.centroids.copy()
            updated[shard_id] = value
            self.fit(updated, self.shard_metadata)
            mode = "full_rebuild"
        return {"mode": mode, "latency_ms": (perf_counter() - started) * 1000.0}

    def add_shard(
        self,
        centroid: np.ndarray,
        metadata: ShardMetadata | None = None,
    ) -> dict[str, float | str]:
        self._require_fitted()
        started = perf_counter()
        updated = np.vstack((self.centroids, np.asarray(centroid, dtype=np.float64)))
        metadata_list = None
        if self.shard_metadata is not None:
            metadata_list = list(self.shard_metadata)
            metadata_list.append(
                metadata
                or ShardMetadata(
                    shard_id=len(metadata_list),
                    item_count=0,
                    centroid=np.asarray(centroid, dtype=np.float64),
                    euclidean_radius=0.0,
                )
            )
        self.fit(updated, metadata_list)
        return {"mode": "full_rebuild", "latency_ms": (perf_counter() - started) * 1000.0}

    def remove_shard(self, shard_id: int) -> dict[str, float | str]:
        self._require_fitted()
        if self.shard_count <= 1:
            raise ValueError("cannot remove the only shard")
        if not 0 <= shard_id < self.shard_count:
            raise IndexError("shard_id is outside the router")
        started = perf_counter()
        updated = np.delete(self.centroids, shard_id, axis=0)
        metadata_list = None
        if self.shard_metadata is not None:
            metadata_list = []
            for new_id, item in enumerate(
                metadata for metadata in self.shard_metadata if metadata.shard_id != shard_id
            ):
                metadata_list.append(replace(item, shard_id=new_id))
        self.fit(updated, metadata_list)
        return {"mode": "full_rebuild", "latency_ms": (perf_counter() - started) * 1000.0}

    def split_shard(
        self,
        shard_id: int,
        first_centroid: np.ndarray,
        second_centroid: np.ndarray,
    ) -> dict[str, float | str]:
        self.update_centroid(shard_id, first_centroid, incremental=False)
        return self.add_shard(second_centroid)

    def merge_shards(
        self,
        first_shard: int,
        second_shard: int,
        merged_centroid: np.ndarray,
    ) -> dict[str, float | str]:
        if first_shard == second_shard:
            raise ValueError("cannot merge a shard with itself")
        keep, remove = sorted((first_shard, second_shard))
        self.update_centroid(keep, merged_centroid, incremental=False)
        return self.remove_shard(remove)

    def save(self, path: str) -> None:
        from .persistence import save_router

        save_router(self, path)

    @classmethod
    def load(cls, path: str) -> "SSFRRouter":
        from .persistence import load_router

        return load_router(path)

    def stats(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "algorithm": "SSFR",
            "shard_count": self.shard_count,
            "embedding_dimension": self.dimension,
            "bands": list(self.bands),
            "ordering_method": self.config.ordering_method,
            "distance_metric": self.config.distance_metric,
            "memory": self.memory_report(),
            "spectral_energy": self.spectral_energy_report(),
            "ordering": self.ordering_report(),
        }
