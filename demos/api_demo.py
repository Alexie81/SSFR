"""FastAPI surface for persistent SSFR catalogs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from ssfr.catalog import CatalogIndex


app = FastAPI(title="SSFR API", version="0.1.0")
_catalog: CatalogIndex | None = None


class BuildRequest(BaseModel):
    csv: str
    output: str
    shards: int = Field(default=256, ge=1)
    bands: list[int] = Field(default_factory=lambda: [8, 16, 32, 64, 128])
    probe_shards: int = Field(default=16, ge=1)
    embedding_provider: str = "hash"
    local_index: str = "exact"


class SearchRequest(BaseModel):
    index: str | None = None
    query: str | None = None
    query_vector: list[float] | None = None
    top_k: int = Field(default=20, ge=1)
    probe_shards: int | None = Field(default=None, ge=1)
    price_min: float | None = None
    price_max: float | None = None
    category: str | None = None
    brand: str | None = None
    color: str | None = None
    audience: str | None = None
    in_stock_only: bool = False
    filter_strategy: str = "post"

    @model_validator(mode="after")
    def validate_query(self) -> "SearchRequest":
        if (self.query is None) == (self.query_vector is None):
            raise ValueError("provide exactly one of query or query_vector")
        return self


def _ensure_catalog(path: str | None = None) -> CatalogIndex:
    global _catalog
    requested = path or os.environ.get("SSFR_INDEX_PATH")
    if _catalog is None:
        if not requested:
            raise HTTPException(status_code=503, detail="no catalog is loaded")
        _catalog = CatalogIndex.load(requested)
    elif requested and _catalog.path.resolve() != Path(requested).resolve():
        _catalog = CatalogIndex.load(requested)
    return _catalog


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "catalog_loaded": _catalog is not None}


@app.post("/index/build")
def build_index(request: BuildRequest) -> dict[str, Any]:
    global _catalog
    try:
        _catalog, report = CatalogIndex.build(
            request.csv,
            request.output,
            shard_count=request.shards,
            bands=tuple(request.bands),
            probe_shards=request.probe_shards,
            embedding_provider=request.embedding_provider,
            local_index_backend=request.local_index,
        )
        return report
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/search")
def search(request: SearchRequest) -> dict[str, Any]:
    try:
        catalog = _ensure_catalog(request.index)
        common = {
            "top_k": request.top_k,
            "probe_shards": request.probe_shards,
            "price_min": request.price_min,
            "price_max": request.price_max,
            "category": request.category,
            "brand": request.brand,
            "color": request.color,
            "audience": request.audience,
            "in_stock_only": request.in_stock_only,
            "filter_strategy": request.filter_strategy,
        }
        if request.query is not None:
            result = catalog.search_text(request.query, **common)
        else:
            result = catalog.search_vector(
                request.query_vector,
                query="<API vector query>",
                **common,
            )
        return {
            "results": [
                {**product.to_dict(), "score": float(score)}
                for product, score in zip(result.products, result.scores, strict=True)
            ],
            "route": {
                "selected_shards": result.search.route.shard_ids.tolist(),
                "used_band": result.search.route.used_band,
                "route_mode": result.search.route.route_mode,
                "centroid_ranking_certified": (
                    result.search.route.centroid_ranking_certified
                ),
                "vector_pruning_certified": result.search.route.vector_pruning_certified,
                "exact_fallback": result.search.route.used_exact_fallback,
                "routing_latency_ms": result.search.routing_latency_ms,
            },
            "local_search_latency_ms": result.search.local_search_latency_ms,
            "total_latency_ms": result.search.total_latency_ms,
            "recall_at_k": result.recall_at_k,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/router/stats")
def router_stats(index: str | None = None) -> dict[str, Any]:
    return _ensure_catalog(index).stats()
