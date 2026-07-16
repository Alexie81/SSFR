"""Persistent e-commerce catalog indexing and search."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .csv_loader import ProductCSVLoader, build_semantic_text
from .distributed_search import DistributedSSFRSearch, exact_global_search
from .embeddings import (
    EmbeddingProvider,
    create_embedding_provider,
    file_sha256,
    load_or_create_embeddings,
)
from .local_index import LocalShardIndex
from .metrics import precision_at_k, recall_at_k
from .router import SSFRRouter
from .sharding import angular_radii, build_shard_metadata, build_shards
from .types import CatalogSearchResult, ProductRecord, SearchResult, SSFRConfig


def _safe_text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value)
    return text if text else None


def _product_from_mapping(item: dict[str, Any]) -> ProductRecord:
    price = item.get("price_ron")
    if price is None or (isinstance(price, float) and np.isnan(price)):
        parsed_price = None
    else:
        parsed_price = float(price)
    return ProductRecord(
        product_id=str(item["product_id"]),
        title=str(item["title"]),
        description=str(item["description"]),
        category=_safe_text(item.get("category")),
        brand=_safe_text(item.get("brand")),
        price_ron=parsed_price,
        color=_safe_text(item.get("color")),
        audience=_safe_text(item.get("audience")),
        in_stock=bool(item.get("in_stock", True)),
    )


class CatalogIndex:
    def __init__(
        self,
        path: Path,
        products: list[ProductRecord],
        embeddings: np.ndarray,
        product_ids: np.ndarray,
        assignments: np.ndarray,
        provider: EmbeddingProvider,
        router: SSFRRouter,
        local_indexes: dict[int, LocalShardIndex],
        manifest: dict[str, Any],
    ) -> None:
        self.path = path
        self.products = products
        self.embeddings = embeddings
        self.product_ids = product_ids
        self.assignments = assignments
        self.provider = provider
        self.router = router
        self.local_indexes = local_indexes
        self.manifest = manifest
        self._product_by_id = {product.product_id: product for product in products}
        self._position_by_id = {
            str(product_id): position for position, product_id in enumerate(product_ids)
        }
        self.distributed = DistributedSSFRSearch(router, local_indexes)

    @classmethod
    def build(
        cls,
        csv_path: str | Path,
        output_path: str | Path,
        *,
        shard_count: int = 256,
        bands: tuple[int, ...] = (8, 16, 32, 64, 128),
        probe_shards: int = 16,
        ordering_method: str = "recursive_pca",
        embedding_provider: str = "hash",
        embedding_model: str = (
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
        embedding_dimension: int = 384,
        local_index_backend: str = "exact",
        tolerant_csv: bool = True,
        force_embeddings: bool = False,
        random_seed: int = 42,
        max_spectral_attempts: int | None = 2,
    ) -> tuple["CatalogIndex", dict[str, Any]]:
        started = perf_counter()
        output = Path(output_path)
        output.mkdir(parents=True, exist_ok=True)

        loader = ProductCSVLoader(tolerant=tolerant_csv)
        products = loader.load(csv_path)
        assert loader.last_report is not None
        (output / "import_report.json").write_text(
            json.dumps(loader.last_report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        texts = [build_semantic_text(product) for product in products]
        with (output / "semantic_texts.jsonl").open("w", encoding="utf-8") as handle:
            for product, text in zip(products, texts, strict=True):
                handle.write(
                    json.dumps(
                        {"product_id": product.product_id, "semantic_text": text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        provider = create_embedding_provider(
            embedding_provider,
            model_name=embedding_model,
            dimension=embedding_dimension,
        )
        embeddings, cache_hit = load_or_create_embeddings(
            texts,
            provider,
            output,
            source_checksum=file_sha256(csv_path),
            force=force_embeddings,
        )
        product_ids = np.asarray([product.product_id for product in products], dtype=str)
        np.save(output / "product_ids.npy", product_ids, allow_pickle=False)

        frame = pd.DataFrame([product.to_dict() for product in products])
        metadata_format = "parquet"
        try:
            frame.to_parquet(output / "products.parquet", index=False)
        except (ImportError, ModuleNotFoundError):
            metadata_format = "csv"
            frame.to_csv(output / "products.csv", index=False, encoding="utf-8")

        shard_result = build_shards(
            embeddings, shard_count, random_seed=random_seed
        )
        np.save(output / "shard_assignments.npy", shard_result.assignments, allow_pickle=False)
        np.save(output / "shard_centroids.npy", shard_result.centroids, allow_pickle=False)
        np.save(output / "shard_radii.npy", shard_result.euclidean_radii, allow_pickle=False)
        np.save(output / "shard_item_counts.npy", shard_result.item_counts, allow_pickle=False)

        angular = angular_radii(embeddings, shard_result)
        np.save(output / "shard_angular_radii.npy", angular, allow_pickle=False)
        metadata = build_shard_metadata(
            shard_result, index_root="local_indexes", angular=angular
        )
        router = SSFRRouter(
            SSFRConfig(
                spectral_bands=bands,
                probe_shards=min(probe_shards, shard_count),
                exact_fallback=True,
                ordering_method=ordering_method,
                distance_metric="cosine",
                normalize_vectors=True,
                random_seed=random_seed,
                max_spectral_attempts=max_spectral_attempts,
            )
        ).fit(shard_result.centroids, metadata)
        router.save(str(output / "ssfr_router"))

        local_root = output / "local_indexes"
        local_indexes: dict[int, LocalShardIndex] = {}
        actual_backend = local_index_backend
        for shard_id in range(shard_count):
            positions = np.flatnonzero(shard_result.assignments == shard_id)
            index = LocalShardIndex(
                backend=local_index_backend,
                distance_metric="cosine",
            )
            index.build(embeddings[positions], product_ids[positions])
            index.save(local_root / f"shard_{shard_id:05d}")
            local_indexes[shard_id] = index
            actual_backend = index.backend

        manifest = {
            "algorithm": "SSFR CSV Catalog",
            "version": "0.2.0",
            "created_at": datetime.now(UTC).isoformat(),
            "source_csv": str(Path(csv_path).resolve()),
            "source_checksum": file_sha256(csv_path),
            "product_count": len(products),
            "embedding_provider": (
                "sentence-transformers"
                if provider.provider_id.startswith("sentence-transformers:")
                else "openai"
                if provider.provider_id.startswith("openai:")
                else "hash"
            ),
            "embedding_provider_id": provider.provider_id,
            "embedding_model": embedding_model,
            "embedding_dimension": provider.dimension,
            "shard_count": shard_count,
            "bands": list(router.bands),
            "probe_shards": router.config.probe_shards,
            "ordering_method": ordering_method,
            "local_index_backend": actual_backend,
            "metadata_format": metadata_format,
            "embedding_cache_hit": cache_hit,
        }
        (output / "catalog_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        index = cls(
            output,
            products,
            embeddings,
            product_ids,
            shard_result.assignments,
            provider,
            router,
            local_indexes,
            manifest,
        )
        report = {
            "products_loaded": len(products),
            "invalid_rows": loader.last_report.rows_invalid,
            "embedding_dimension": provider.dimension,
            "embedding_cache_hit": cache_hit,
            "shards_built": shard_count,
            "bands": list(router.bands),
            "local_index_backend": actual_backend,
            "build_time_seconds": perf_counter() - started,
            "artifact_path": str(output.resolve()),
        }
        return index, report

    @classmethod
    def load(cls, path: str | Path) -> "CatalogIndex":
        directory = Path(path)
        manifest = json.loads(
            (directory / "catalog_manifest.json").read_text(encoding="utf-8")
        )
        if manifest["metadata_format"] == "parquet":
            frame = pd.read_parquet(directory / "products.parquet")
        else:
            frame = pd.read_csv(directory / "products.csv")
        products = [_product_from_mapping(item) for item in frame.to_dict(orient="records")]
        provider = create_embedding_provider(
            manifest["embedding_provider"],
            model_name=manifest.get("embedding_model", ""),
            dimension=int(manifest["embedding_dimension"]),
        )
        embeddings = np.load(directory / "embeddings.npy", mmap_mode="r", allow_pickle=False)
        product_ids = np.load(directory / "product_ids.npy", allow_pickle=False)
        assignments = np.load(directory / "shard_assignments.npy", allow_pickle=False)
        router = SSFRRouter.load(str(directory / "ssfr_router"))
        local_indexes = {
            shard_id: LocalShardIndex.load(
                directory / "local_indexes" / f"shard_{shard_id:05d}"
            )
            for shard_id in range(router.shard_count)
        }
        return cls(
            directory,
            products,
            embeddings,
            product_ids,
            assignments,
            provider,
            router,
            local_indexes,
            manifest,
        )

    def _filter_mask(
        self,
        *,
        price_min: float | None = None,
        price_max: float | None = None,
        category: str | None = None,
        brand: str | None = None,
        color: str | None = None,
        audience: str | None = None,
        in_stock_only: bool = False,
    ) -> np.ndarray:
        mask = np.ones(len(self.products), dtype=bool)
        for position, product in enumerate(self.products):
            if price_min is not None and (
                product.price_ron is None or product.price_ron < price_min
            ):
                mask[position] = False
            if price_max is not None and (
                product.price_ron is None or product.price_ron > price_max
            ):
                mask[position] = False
            for expected, actual in (
                (category, product.category),
                (brand, product.brand),
                (color, product.color),
                (audience, product.audience),
            ):
                if expected is not None and (
                    actual is None or actual.casefold() != expected.casefold()
                ):
                    mask[position] = False
            if in_stock_only and not product.in_stock:
                mask[position] = False
        return mask

    def search_text(
        self,
        query: str,
        *,
        top_k: int = 10,
        probe_shards: int | None = None,
        filter_strategy: str = "post",
        report_path: str | Path | None = "reports/csv_search_evaluation.csv",
        **filters: Any,
    ) -> CatalogSearchResult:
        vector = self.provider.encode_query(query)
        return self.search_vector(
            vector,
            query=query,
            top_k=top_k,
            probe_shards=probe_shards,
            filter_strategy=filter_strategy,
            report_path=report_path,
            **filters,
        )

    def search_vector(
        self,
        query_vector: np.ndarray,
        *,
        query: str = "<vector query>",
        top_k: int = 10,
        probe_shards: int | None = None,
        filter_strategy: str = "post",
        report_path: str | Path | None = "reports/csv_search_evaluation.csv",
        price_min: float | None = None,
        price_max: float | None = None,
        category: str | None = None,
        brand: str | None = None,
        color: str | None = None,
        audience: str | None = None,
        in_stock_only: bool = False,
    ) -> CatalogSearchResult:
        if filter_strategy not in {"pre", "post"}:
            raise ValueError("filter_strategy must be 'pre' or 'post'")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        mask = self._filter_mask(
            price_min=price_min,
            price_max=price_max,
            category=category,
            brand=brand,
            color=color,
            audience=audience,
            in_stock_only=in_stock_only,
        )
        vector = np.asarray(query_vector, dtype=np.float64)
        vector /= np.linalg.norm(vector)

        if filter_strategy == "pre":
            allowed = {
                shard_id: self.product_ids[
                    (self.assignments == shard_id) & mask
                ]
                for shard_id in range(self.router.shard_count)
            }
            raw = self.distributed.search(
                vector,
                top_k,
                probe_shards=probe_shards,
                allowed_ids_by_shard=allowed,
            )
            filtered_ids = raw.item_ids
            filtered_scores = raw.scores
        else:
            budget = min(len(self.products), max(top_k * 5, top_k))
            while True:
                raw = self.distributed.search(
                    vector,
                    budget,
                    probe_shards=probe_shards,
                    local_top_k=budget,
                )
                keep = np.asarray(
                    [mask[self._position_by_id[str(item_id)]] for item_id in raw.item_ids],
                    dtype=bool,
                )
                filtered_ids = raw.item_ids[keep][:top_k]
                filtered_scores = raw.scores[keep][:top_k]
                selected_capacity = sum(
                    self.local_indexes[int(shard_id)].item_count
                    for shard_id in raw.route.shard_ids
                )
                if filtered_ids.size >= top_k or budget >= selected_capacity:
                    break
                budget = min(selected_capacity, budget * 2)

        search = replace(
            raw,
            item_ids=np.asarray(filtered_ids),
            scores=np.asarray(filtered_scores, dtype=np.float64),
        )
        oracle_ids, _ = exact_global_search(
            vector,
            self.embeddings,
            self.product_ids,
            top_k,
            allowed_mask=mask,
        )
        products = tuple(self._product_by_id[str(item_id)] for item_id in search.item_ids)
        recall = recall_at_k(search.item_ids.tolist(), oracle_ids.tolist(), top_k)
        precision = precision_at_k(search.item_ids.tolist(), oracle_ids.tolist(), top_k)
        result = CatalogSearchResult(
            query=query,
            products=products,
            scores=search.scores,
            search=search,
            oracle_product_ids=oracle_ids,
            recall_at_k=recall,
            precision_at_k=precision,
            filter_strategy=filter_strategy,
        )
        if report_path is not None:
            self._append_evaluation(result, report_path)
        return result

    @staticmethod
    def _append_evaluation(
        result: CatalogSearchResult,
        report_path: str | Path,
    ) -> None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "query": result.query,
            "selected_shards": json.dumps(result.search.route.shard_ids.tolist()),
            "used_band": result.search.route.used_band,
            "route_mode": result.search.route.route_mode,
            "certified": result.search.route.centroid_ranking_certified,
            "vector_pruning_certified": result.search.route.vector_pruning_certified,
            "fallback": result.search.route.used_exact_fallback,
            "routing_latency_ms": result.search.routing_latency_ms,
            "local_search_latency_ms": result.search.local_search_latency_ms,
            "total_latency_ms": result.search.total_latency_ms,
            "result_product_ids": json.dumps(result.search.item_ids.tolist(), ensure_ascii=False),
            "oracle_product_ids": json.dumps(
                result.oracle_product_ids.tolist(), ensure_ascii=False
            ),
            "recall_at_k": result.recall_at_k,
            "precision_at_k": result.precision_at_k,
            "filter_strategy": result.filter_strategy,
        }
        write_header = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def stats(self) -> dict[str, Any]:
        return {
            "catalog": self.manifest,
            "router": self.router.stats(),
            "local_index_items": sum(index.item_count for index in self.local_indexes.values()),
        }


def format_catalog_search(result: CatalogSearchResult) -> str:
    route = result.search.route
    lines = [
        f"Query:\n{result.query}",
        "",
        "SSFR route:",
        f"- total shards: {route.approximate_scores.size}",
        f"- selected shards: {route.shard_ids.tolist()}",
        f"- used Fourier band: {route.used_band}",
        f"- route mode: {route.route_mode}",
        f"- centroid ranking certified: {str(route.centroid_ranking_certified).lower()}",
        f"- vector pruning certified: {str(route.vector_pruning_certified).lower()}",
        f"- exact fallback: {str(route.used_exact_fallback).lower()}",
        f"- routing latency: {result.search.routing_latency_ms:.3f} ms",
        "",
        "Local search:",
        f"- shards accessed: {result.search.shards_accessed}",
        f"- candidate vectors available: {result.search.candidate_vectors_evaluated}",
        f"- local index latency: {result.search.local_search_latency_ms:.3f} ms",
        f"- merge latency: {result.search.merge_latency_ms:.3f} ms",
        f"- total latency: {result.search.total_latency_ms:.3f} ms",
        f"- Recall@k vs exact oracle: {result.recall_at_k:.4f}",
        f"- Precision@k vs exact oracle: {result.precision_at_k:.4f}",
        "",
        "Top results:",
    ]
    for rank, (product, score) in enumerate(
        zip(result.products, result.scores, strict=True), start=1
    ):
        price = "n/a" if product.price_ron is None else f"{product.price_ron:.2f} RON"
        lines.append(
            f"{rank}. {product.product_id} — {product.title} — {price} — score {score:.4f}"
        )
    if not result.products:
        lines.append("(no products matched the filters in the routed shards)")
    return "\n".join(lines)
