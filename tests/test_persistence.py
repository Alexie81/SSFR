from __future__ import annotations

import numpy as np

from ssfr import SSFRConfig, SSFRRouter


def test_save_load_preserves_routes(
    tmp_path,
    centroids: np.ndarray,
    rng: np.random.Generator,
) -> None:
    router = SSFRRouter(
        SSFRConfig(spectral_bands=(2, 4, 8), probe_shards=4)
    ).fit(centroids)
    query = rng.normal(size=centroids.shape[1])
    expected = router.route(query)
    router.save(str(tmp_path / "router"))
    loaded = SSFRRouter.load(str(tmp_path / "router"))
    actual = loaded.route(query)
    assert np.array_equal(actual.shard_ids, expected.shard_ids)
    assert np.allclose(actual.approximate_scores, expected.approximate_scores)
    assert loaded.memory_report() == router.memory_report()
