from __future__ import annotations

import numpy as np
import pytest

from ssfr import SSFRConfig, SSFRRouter
from ssfr.certificates import intervals_cover_exact
from ssfr.metrics import normalize_rows


def test_exact_fallback_matches_matrix_product(
    centroids: np.ndarray,
    rng: np.random.Generator,
) -> None:
    router = SSFRRouter(
        SSFRConfig(
            spectral_bands=(0,),
            probe_shards=3,
            exact_fallback=True,
            ordering_method="identity",
        )
    ).fit(centroids)
    query = rng.normal(size=centroids.shape[1])
    route = router.route(query)
    exact = router.exact_route(query)
    assert np.array_equal(route.shard_ids, exact.shard_ids)
    assert route.used_exact_fallback


def test_zero_query_is_controlled(centroids: np.ndarray) -> None:
    router = SSFRRouter().fit(centroids)
    with pytest.raises(ValueError, match="non-zero"):
        router.route(np.zeros(centroids.shape[1]))


def test_wrong_query_dimension_is_explicit(centroids: np.ndarray) -> None:
    router = SSFRRouter().fit(centroids)
    with pytest.raises(ValueError, match="shape"):
        router.route(np.ones(centroids.shape[1] + 1))


def test_full_band_route_equals_exact(
    centroids: np.ndarray,
    rng: np.random.Generator,
) -> None:
    router = SSFRRouter(
        SSFRConfig(
            spectral_bands=(centroids.shape[0] // 2,),
            probe_shards=5,
            exact_fallback=True,
        )
    ).fit(centroids)
    query = rng.normal(size=centroids.shape[1])
    route = router.route(query)
    exact = router.exact_route(query)
    assert np.array_equal(route.shard_ids, exact.shard_ids)
    assert not route.used_exact_fallback
    assert route.centroid_ranking_certified


def test_unnormalized_query_intervals_cover_scores(
    rng: np.random.Generator,
) -> None:
    centroids = rng.normal(size=(12, 7))
    router = SSFRRouter(
        SSFRConfig(
            spectral_bands=(1,),
            probe_shards=3,
            exact_fallback=False,
            normalize_vectors=False,
        )
    ).fit(centroids)
    query = rng.normal(size=7) * 3.0
    route = router.route(query)
    assert intervals_cover_exact(
        centroids @ query, route.lower_bounds, route.upper_bounds, tolerance=1e-10
    )


def test_route_batch_matches_individual(
    centroids: np.ndarray,
    rng: np.random.Generator,
) -> None:
    router = SSFRRouter(
        SSFRConfig(spectral_bands=(2, 4, 8), probe_shards=4)
    ).fit(centroids)
    queries = normalize_rows(rng.normal(size=(5, centroids.shape[1])))
    batch = router.route_batch(queries)
    individual = [router.route(query) for query in queries]
    assert all(
        np.array_equal(first.shard_ids, second.shard_ids)
        for first, second in zip(batch, individual, strict=True)
    )


def test_incremental_dft_update_matches_full_rebuild(
    centroids: np.ndarray,
    rng: np.random.Generator,
) -> None:
    replacement = normalize_rows(rng.normal(size=(1, centroids.shape[1])))[0]
    incremental = SSFRRouter(
        SSFRConfig(spectral_bands=(2, 4, 8), probe_shards=4, ordering_method="identity")
    ).fit(centroids.copy())
    rebuilt = SSFRRouter(
        SSFRConfig(spectral_bands=(2, 4, 8), probe_shards=4, ordering_method="identity")
    ).fit(centroids.copy())
    incremental.update_centroid(3, replacement, incremental=True)
    rebuilt.update_centroid(3, replacement, incremental=False)
    query = rng.normal(size=centroids.shape[1])
    assert np.array_equal(
        incremental.exact_route(query).shard_ids,
        rebuilt.exact_route(query).shard_ids,
    )
    for band in incremental.bands:
        assert np.allclose(
            incremental.spectral_payloads[band],
            rebuilt.spectral_payloads[band],
            atol=1e-10,
        )
