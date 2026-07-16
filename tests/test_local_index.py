from __future__ import annotations

import numpy as np
import pytest

from ssfr.local_index import LocalShardIndex
from ssfr.metrics import normalize_rows


def test_exact_local_index_search_and_persistence(
    tmp_path,
    rng: np.random.Generator,
) -> None:
    vectors = normalize_rows(rng.normal(size=(20, 8)))
    ids = np.asarray([f"I{value:03d}" for value in range(20)])
    index = LocalShardIndex("exact", "cosine")
    index.build(vectors, ids)
    found, scores = index.search(vectors[4], 3)
    assert found[0] == ids[4]
    assert scores[0] == pytest.approx(1.0, abs=1e-6)
    index.save(tmp_path / "local")
    loaded = LocalShardIndex.load(tmp_path / "local")
    loaded_found, loaded_scores = loaded.search(vectors[4], 3)
    assert np.array_equal(found, loaded_found)
    assert np.allclose(scores, loaded_scores)


def test_exact_local_index_filtering(rng: np.random.Generator) -> None:
    vectors = normalize_rows(rng.normal(size=(10, 5)))
    ids = np.arange(10)
    index = LocalShardIndex("exact")
    index.build(vectors, ids)
    found, _ = index.search(vectors[0], 5, allowed_ids=np.array([0, 2]))
    assert set(found.tolist()) <= {0, 2}


def test_auto_backend_keeps_small_shards_exact(
    rng: np.random.Generator,
) -> None:
    vectors = normalize_rows(rng.normal(size=(20, 6)))
    index = LocalShardIndex("auto", auto_hnsw_threshold=100)
    index.build(vectors, np.arange(20))
    assert index.backend == "exact"
