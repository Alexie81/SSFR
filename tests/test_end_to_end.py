from __future__ import annotations

import numpy as np

from ssfr import DistributedSSFRSearch, LocalShardIndex, SSFRConfig, SSFRRouter
from ssfr.distributed_search import exact_global_search
from ssfr.metrics import normalize_rows
from ssfr.sharding import build_shard_metadata, build_shards


def test_distributed_search_equals_oracle_when_all_shards_are_accessed(
    rng: np.random.Generator,
) -> None:
    vectors = normalize_rows(rng.normal(size=(80, 10)))
    ids = np.arange(vectors.shape[0])
    shards = build_shards(vectors, 8, random_seed=7)
    metadata = build_shard_metadata(shards)
    router = SSFRRouter(
        SSFRConfig(
            spectral_bands=(4,),
            probe_shards=8,
            exact_fallback=True,
            ordering_method="recursive_pca",
        )
    ).fit(shards.centroids, metadata)
    indexes = {}
    for shard_id in range(8):
        positions = np.flatnonzero(shards.assignments == shard_id)
        index = LocalShardIndex("exact")
        index.build(vectors[positions], ids[positions])
        indexes[shard_id] = index
    query = normalize_rows(rng.normal(size=(1, 10)))[0]
    result = DistributedSSFRSearch(router, indexes).search(
        query, top_k=10, probe_shards=8
    )
    oracle_ids, oracle_scores = exact_global_search(query, vectors, ids, 10)
    assert np.array_equal(result.item_ids, oracle_ids)
    assert np.allclose(result.scores, oracle_scores)
