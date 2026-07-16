from __future__ import annotations

import numpy as np
import pytest

from ssfr.metrics import normalize_rows


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def centroids(rng: np.random.Generator) -> np.ndarray:
    return normalize_rows(rng.normal(size=(16, 12)), name="centroids")
