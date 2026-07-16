from __future__ import annotations

import numpy as np

from ssfr.calibration import calibrate_probe_count
from ssfr.metrics import normalize_rows
from ssfr.sharding import build_shards


def test_probe_calibration_selects_a_valid_budget(
    rng: np.random.Generator,
) -> None:
    themes = normalize_rows(rng.normal(size=(4, 12)))
    labels = np.arange(200) % 4
    embeddings = normalize_rows(
        themes[labels] + 0.1 * rng.normal(size=(200, 12))
    )
    shards = build_shards(embeddings, 8, random_seed=9)
    queries = normalize_rows(themes + 0.02 * rng.normal(size=themes.shape))
    result = calibrate_probe_count(
        embeddings=embeddings,
        assignments=shards.assignments,
        centroids=shards.centroids,
        validation_queries=queries,
        top_k=5,
        probe_values=(1, 2, 4, 8),
        target_recall=0.9,
    )
    assert result.selected_probe_shards in {1, 2, 4, 8}
    assert result.mean_recall_by_probe[8] == 1.0
