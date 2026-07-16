from __future__ import annotations

import numpy as np
import pytest

from ssfr.ordering import (
    build_order,
    identity_order,
    recursive_pca_order,
    validate_order,
)


def test_recursive_pca_order_is_valid(centroids: np.ndarray) -> None:
    order = recursive_pca_order(centroids)
    assert np.array_equal(np.sort(order), np.arange(centroids.shape[0]))


def test_identity_order(centroids: np.ndarray) -> None:
    assert np.array_equal(identity_order(centroids), np.arange(centroids.shape[0]))


def test_all_ordering_dispatch_methods_are_permutations(centroids: np.ndarray) -> None:
    for method in (
        "identity",
        "random",
        "recursive_pca",
        "nearest_neighbor_chain",
        "hierarchical",
        "spectral_seriation",
    ):
        order = build_order(centroids, method, seed=7)
        validate_order(order, centroids.shape[0])


def test_invalid_order_is_rejected() -> None:
    with pytest.raises(ValueError, match="permutation"):
        validate_order(np.array([0, 0, 1]), 3)
