"""Capacity and latency estimates for deployments larger than one machine."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ScaleEstimate:
    estimated_items: int
    shard_count: int
    embedding_dimension: int
    fourier_band: int
    probe_shards: int
    parallel_shard_requests: int
    items_per_shard_mean: float
    routed_item_universe: float
    shard_reduction_factor: float
    exact_centroid_multiply_adds: int
    spectral_projection_complex_terms: int
    approximate_ifft_operations: int
    centroid_matrix_float32_bytes: int
    spectral_payload_complex64_bytes: int
    residuals_float32_bytes: int
    estimated_router_latency_ms: float | None
    estimated_local_and_network_latency_ms: float | None
    estimated_merge_latency_ms: float | None
    estimated_end_to_end_latency_ms: float | None
    physical_benchmark: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["warning"] = (
            "Capacity model only; no one-billion-vector corpus was physically indexed."
        )
        return payload


def estimate_scale(
    *,
    items: int = 1_000_000_000,
    shards: int = 16_384,
    dimensions: int = 768,
    fourier_band: int = 256,
    probe_shards: int = 32,
    spectral_band_count: int = 7,
    parallel_shard_requests: int = 32,
    measured_router_ms: float | None = None,
    measured_local_shard_p95_ms: float | None = None,
    measured_network_p95_ms: float | None = None,
    measured_merge_ms: float | None = None,
) -> ScaleEstimate:
    for name, value in (
        ("items", items),
        ("shards", shards),
        ("dimensions", dimensions),
        ("probe_shards", probe_shards),
        ("spectral_band_count", spectral_band_count),
        ("parallel_shard_requests", parallel_shard_requests),
    ):
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
    if probe_shards > shards:
        raise ValueError("probe_shards cannot exceed shards")
    maximum_band = shards // 2
    band = min(max(int(fourier_band), 0), maximum_band)
    items_per_shard = items / shards
    routed_universe = items_per_shard * probe_shards
    exact_terms = shards * dimensions
    spectral_terms = (band + 1) * dimensions
    ifft_operations = int(5 * shards * math.log2(max(shards, 2)))
    centroid_bytes = shards * dimensions * 4
    spectral_bytes = (band + 1) * dimensions * 8
    residual_bytes = spectral_band_count * shards * 4

    router = measured_router_ms
    local_network = None
    merge = measured_merge_ms
    total = None
    if (
        measured_local_shard_p95_ms is not None
        and measured_network_p95_ms is not None
    ):
        waves = math.ceil(probe_shards / parallel_shard_requests)
        local_network = waves * (
            measured_local_shard_p95_ms + measured_network_p95_ms
        )
    if router is not None and local_network is not None and merge is not None:
        total = router + local_network + merge

    return ScaleEstimate(
        estimated_items=items,
        shard_count=shards,
        embedding_dimension=dimensions,
        fourier_band=band,
        probe_shards=probe_shards,
        parallel_shard_requests=parallel_shard_requests,
        items_per_shard_mean=items_per_shard,
        routed_item_universe=routed_universe,
        shard_reduction_factor=shards / probe_shards,
        exact_centroid_multiply_adds=exact_terms,
        spectral_projection_complex_terms=spectral_terms,
        approximate_ifft_operations=ifft_operations,
        centroid_matrix_float32_bytes=centroid_bytes,
        spectral_payload_complex64_bytes=spectral_bytes,
        residuals_float32_bytes=residual_bytes,
        estimated_router_latency_ms=router,
        estimated_local_and_network_latency_ms=local_network,
        estimated_merge_latency_ms=merge,
        estimated_end_to_end_latency_ms=total,
    )
