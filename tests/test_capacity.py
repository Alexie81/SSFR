from __future__ import annotations

import pytest

from ssfr.capacity import estimate_scale


def test_billion_scale_estimate_is_labelled_and_consistent() -> None:
    estimate = estimate_scale()
    payload = estimate.to_dict()
    assert estimate.physical_benchmark is False
    assert estimate.items_per_shard_mean == pytest.approx(
        1_000_000_000 / 16_384
    )
    assert estimate.shard_reduction_factor == 512
    assert "Capacity model only" in payload["warning"]


def test_latency_estimate_requires_supplied_measurements() -> None:
    without = estimate_scale()
    assert without.estimated_end_to_end_latency_ms is None
    with_values = estimate_scale(
        measured_router_ms=0.8,
        measured_local_shard_p95_ms=8.0,
        measured_network_p95_ms=1.5,
        measured_merge_ms=0.5,
    )
    assert with_values.estimated_end_to_end_latency_ms == pytest.approx(10.8)
