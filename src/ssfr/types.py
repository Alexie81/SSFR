"""Typed data models shared by SSFR components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SSFRConfig:
    spectral_bands: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512)
    probe_shards: int = 16
    exact_fallback: bool = True
    ordering_method: str = "recursive_pca"
    distance_metric: str = "cosine"
    normalize_vectors: bool = True
    random_seed: int = 42

    def __post_init__(self) -> None:
        bands = tuple(sorted({int(band) for band in self.spectral_bands if int(band) >= 0}))
        if not bands:
            raise ValueError("spectral_bands must contain at least one non-negative band")
        if self.probe_shards < 1:
            raise ValueError("probe_shards must be at least 1")
        if self.distance_metric not in {"cosine", "euclidean"}:
            raise ValueError("distance_metric must be 'cosine' or 'euclidean'")
        object.__setattr__(self, "spectral_bands", bands)


@dataclass(frozen=True)
class RouteResult:
    shard_ids: np.ndarray
    approximate_scores: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    used_band: int
    centroid_ranking_certified: bool
    vector_pruning_certified: bool
    used_exact_fallback: bool
    latency_ms: float
    route_mode: str = "spectral"


@dataclass(frozen=True)
class ShardMetadata:
    shard_id: int
    item_count: int
    centroid: np.ndarray
    euclidean_radius: float
    angular_radius: float | None = None
    index_path: str = ""


@dataclass(frozen=True)
class SearchResult:
    item_ids: np.ndarray
    scores: np.ndarray
    route: RouteResult
    routing_latency_ms: float
    local_search_latency_ms: float
    merge_latency_ms: float
    total_latency_ms: float
    shards_accessed: int
    candidate_vectors_evaluated: int


@dataclass(frozen=True)
class ProductRecord:
    product_id: str
    title: str
    description: str
    category: str | None = None
    brand: str | None = None
    price_ron: float | None = None
    color: str | None = None
    audience: str | None = None
    in_stock: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "brand": self.brand,
            "price_ron": self.price_ron,
            "color": self.color,
            "audience": self.audience,
            "in_stock": self.in_stock,
        }


@dataclass(frozen=True)
class ImportReport:
    source: str
    rows_read: int
    rows_valid: int
    rows_invalid: int
    duplicate_ids: int
    encoding: str
    invalid_rows: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "rows_read": self.rows_read,
            "rows_valid": self.rows_valid,
            "rows_invalid": self.rows_invalid,
            "duplicate_ids": self.duplicate_ids,
            "encoding": self.encoding,
            "invalid_rows": list(self.invalid_rows),
        }


@dataclass(frozen=True)
class ShardBuildResult:
    assignments: np.ndarray
    centroids: np.ndarray
    euclidean_radii: np.ndarray
    item_counts: np.ndarray


@dataclass(frozen=True)
class CatalogSearchResult:
    query: str
    products: tuple[ProductRecord, ...]
    scores: np.ndarray
    search: SearchResult
    oracle_product_ids: np.ndarray
    recall_at_k: float
    precision_at_k: float
    filter_strategy: str
