"""Persistent e-commerce catalog indexing and search."""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
from collections.abc import Callable
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
from .metrics import normalize_rows, precision_at_k, recall_at_k
from .router import SSFRRouter
from .sharding import angular_radii, build_shard_metadata, build_shards
from .types import (
    CatalogSearchResult,
    ProductRecord,
    SearchResult,
    ShardBuildResult,
    SSFRConfig,
)


_PRODUCT_CSV_FIELDS = (
    "product_id",
    "title",
    "description",
    "category",
    "brand",
    "price_ron",
    "color",
    "audience",
    "in_stock",
)


def _embedding_provider_name(provider_id: str) -> str:
    if provider_id.startswith("multilingual-e5:"):
        return "multilingual-e5"
    if provider_id.startswith("sentence-transformers:"):
        return "sentence-transformers"
    if provider_id.startswith("openai:"):
        return "openai"
    if provider_id.startswith("fast-hash"):
        return "fast-hash"
    return "hash"


def _repair_streaming_empty_clusters(
    assignments: np.memmap,
    embeddings: np.memmap,
    centroid_sums: np.ndarray,
    item_counts: np.ndarray,
    *,
    batch_size: int,
) -> int:
    """Move existing points into empty clusters without loading the catalog."""

    empty_clusters = np.flatnonzero(item_counts == 0)
    if empty_clusters.size == 0:
        return 0

    donor_heap = [
        (-int(count), int(shard_id))
        for shard_id, count in enumerate(item_counts)
        if count > 1
    ]
    heapq.heapify(donor_heap)
    donor_targets: dict[int, list[int]] = {}
    for empty in empty_clusters:
        if not donor_heap:
            raise RuntimeError("unable to repair an empty shard")
        negative_count, donor = heapq.heappop(donor_heap)
        count = -negative_count
        donor_targets.setdefault(donor, []).append(int(empty))
        count -= 1
        if count > 1:
            heapq.heappush(donor_heap, (-count, donor))

    donor_positions: dict[int, list[int]] = {
        donor: [] for donor in donor_targets
    }
    pending = set(donor_targets)
    for start in range(0, assignments.shape[0], batch_size):
        stop = min(start + batch_size, assignments.shape[0])
        labels = np.asarray(assignments[start:stop])
        completed: list[int] = []
        for donor in pending:
            needed = len(donor_targets[donor]) - len(donor_positions[donor])
            matches = np.flatnonzero(labels == donor)[:needed]
            donor_positions[donor].extend((start + matches).tolist())
            if len(donor_positions[donor]) == len(donor_targets[donor]):
                completed.append(donor)
        pending.difference_update(completed)
        if not pending:
            break
    if pending:
        raise RuntimeError("unable to locate donor points for empty shards")

    for donor, targets in donor_targets.items():
        for position, empty in zip(donor_positions[donor], targets, strict=True):
            vector = np.asarray(embeddings[position], dtype=np.float64)
            assignments[position] = empty
            centroid_sums[donor] -= vector
            centroid_sums[empty] += vector
            item_counts[donor] -= 1
            item_counts[empty] += 1
    return int(empty_clusters.size)


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
        embedding_model: str | None = None,
        embedding_dimension: int = 384,
        embedding_batch_size: int = 64,
        local_index_backend: str = "exact",
        tolerant_csv: bool = True,
        force_embeddings: bool = False,
        random_seed: int = 42,
        max_spectral_attempts: int | None = 2,
    ) -> tuple["CatalogIndex", dict[str, Any]]:
        started = perf_counter()
        if embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be at least 1")
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
            batch_size=embedding_batch_size,
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
            "embedding_provider": _embedding_provider_name(provider.provider_id),
            "embedding_provider_id": provider.provider_id,
            "embedding_model": getattr(provider, "model_name", embedding_model or ""),
            "embedding_dimension": provider.dimension,
            "embedding_device": getattr(provider, "device", "cpu"),
            "embedding_precision": getattr(provider, "precision", "fp32"),
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
            "embedding_device": getattr(provider, "device", "cpu"),
            "embedding_precision": getattr(provider, "precision", "fp32"),
            "embedding_cache_hit": cache_hit,
            "shards_built": shard_count,
            "bands": list(router.bands),
            "local_index_backend": actual_backend,
            "build_time_seconds": perf_counter() - started,
            "artifact_path": str(output.resolve()),
        }
        return index, report

    @classmethod
    def build_streaming(
        cls,
        csv_path: str | Path,
        output_path: str | Path,
        *,
        shard_count: int = 256,
        bands: tuple[int, ...] = (8, 16, 32, 64, 128),
        probe_shards: int = 16,
        ordering_method: str = "recursive_pca",
        embedding_provider: str = "hash",
        embedding_model: str | None = None,
        embedding_dimension: int = 384,
        embedding_batch_size: int = 64,
        local_index_backend: str = "exact",
        tolerant_csv: bool = True,
        force_embeddings: bool = False,
        random_seed: int = 42,
        max_spectral_attempts: int | None = 2,
        batch_size: int = 10_000,
        kmeans_epochs: int = 1,
        progress: Callable[[str, int, int], None] | None = None,
        progress_every: int = 50_000,
        load_after_build: bool = False,
    ) -> tuple["CatalogIndex | None", dict[str, Any]]:
        """Build large catalog artifacts with bounded per-batch working memory.

        The CSV is parsed exactly twice. The first pass validates and counts
        products; the second writes metadata, semantic text, IDs, and embeddings.
        """

        if shard_count < 1:
            raise ValueError("shard_count must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be at least 1")
        if kmeans_epochs < 1:
            raise ValueError("kmeans_epochs must be at least 1")
        if progress_every < 1:
            raise ValueError("progress_every must be at least 1")
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable or None")

        started = perf_counter()
        source = Path(csv_path)
        output = Path(output_path)
        output.mkdir(parents=True, exist_ok=True)
        products_path = output / "products.csv"
        if source.resolve() == products_path.resolve():
            raise ValueError("output products.csv must not overwrite the source CSV")

        last_progress: dict[str, int] = {}

        def report_progress(
            phase: str,
            current: int,
            total: int,
            *,
            force: bool = False,
        ) -> None:
            if progress is None:
                return
            previous = last_progress.get(phase)
            if (
                force
                or previous is None
                or current - previous >= progress_every
            ):
                progress(phase, int(current), int(total))
                last_progress[phase] = int(current)

        metadata_loader = ProductCSVLoader(tolerant=tolerant_csv)
        product_count = 0
        max_product_id_length = 1
        report_progress("csv_pass_1", 0, 0, force=True)
        for product_batch in metadata_loader.iter_batches(
            source, batch_size=batch_size
        ):
            max_product_id_length = max(
                max_product_id_length,
                max(len(product.product_id) for product in product_batch),
            )
            product_count += len(product_batch)
            report_progress("csv_pass_1", product_count, 0)

        if metadata_loader.last_report is None:
            raise RuntimeError("CSV import did not produce an import report")
        if product_count == 0:
            raise ValueError("CSV contains no valid product rows")
        if shard_count > product_count:
            raise ValueError(
                "shard_count must be between 1 and the number of embeddings"
            )
        report_progress(
            "csv_pass_1", product_count, product_count, force=True
        )
        (output / "import_report.json").write_text(
            json.dumps(
                metadata_loader.last_report.to_dict(),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        source_checksum = file_sha256(source)
        report_progress("embedding_model", 0, 1, force=True)
        provider = create_embedding_provider(
            embedding_provider,
            model_name=embedding_model,
            dimension=embedding_dimension,
        )
        report_progress("embedding_model", 1, 1, force=True)
        cache_key = hashlib.sha256(
            (
                f"{source_checksum}|{provider.provider_id}|{product_count}"
            ).encode("utf-8")
        ).hexdigest()
        embedding_path = output / "embeddings.npy"
        cache_path = output / "embedding_cache.json"
        cache_hit = False
        embeddings: np.memmap
        if not force_embeddings and embedding_path.exists() and cache_path.exists():
            try:
                cache_manifest = json.loads(
                    cache_path.read_text(encoding="utf-8")
                )
                if cache_manifest.get("cache_key") == cache_key:
                    cached_embeddings = np.lib.format.open_memmap(
                        embedding_path, mode="r+"
                    )
                    if (
                        cached_embeddings.shape
                        == (product_count, provider.dimension)
                        and cached_embeddings.dtype == np.dtype(np.float32)
                    ):
                        embeddings = cached_embeddings
                        del cached_embeddings
                        cache_hit = True
                    else:
                        del cached_embeddings
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                cache_hit = False
        if not cache_hit:
            embeddings = np.lib.format.open_memmap(
                embedding_path,
                mode="w+",
                dtype=np.float32,
                shape=(product_count, provider.dimension),
            )

        product_ids = np.lib.format.open_memmap(
            output / "product_ids.npy",
            mode="w+",
            dtype=f"<U{max_product_id_length}",
            shape=(product_count,),
        )

        def checkpoint_embeddings() -> None:
            """Flush and remap embeddings so Windows can release old pages."""

            nonlocal embeddings
            embeddings.flush()
            del embeddings
            embeddings = np.lib.format.open_memmap(
                embedding_path,
                mode="r+",
            )

        def crossed_checkpoint(start: int, stop: int, total: int) -> bool:
            return stop == total or stop // progress_every > start // progress_every

        embedding_loader = ProductCSVLoader(tolerant=tolerant_csv)
        second_pass_count = 0
        report_progress("csv_pass_2", 0, product_count, force=True)
        with (
            products_path.open(
                "w", encoding="utf-8", newline=""
            ) as products_handle,
            (output / "semantic_texts.jsonl").open(
                "w", encoding="utf-8"
            ) as semantic_handle,
        ):
            writer = csv.DictWriter(
                products_handle,
                fieldnames=list(_PRODUCT_CSV_FIELDS),
            )
            writer.writeheader()
            for product_batch in embedding_loader.iter_batches(
                source, batch_size=batch_size
            ):
                stop = second_pass_count + len(product_batch)
                if stop > product_count:
                    raise RuntimeError(
                        "CSV changed between streaming build passes"
                    )
                batch_ids = [
                    product.product_id for product in product_batch
                ]
                if any(
                    len(product_id) > max_product_id_length
                    for product_id in batch_ids
                ):
                    raise RuntimeError(
                        "CSV changed between streaming build passes"
                    )
                writer.writerows(
                    product.to_dict() for product in product_batch
                )
                texts = [build_semantic_text(product) for product in product_batch]
                for product, text in zip(
                    product_batch, texts, strict=True
                ):
                    semantic_handle.write(
                        json.dumps(
                            {
                                "product_id": product.product_id,
                                "semantic_text": text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                product_ids[second_pass_count:stop] = batch_ids
                if not cache_hit:
                    encoded = provider.encode_texts(
                        texts,
                        batch_size=embedding_batch_size,
                    )
                    normalized = normalize_rows(
                        encoded, name="embedding batch"
                    ).astype(np.float32)
                    if normalized.shape != (
                        len(product_batch),
                        provider.dimension,
                    ):
                        raise ValueError(
                            "embedding provider returned an unexpected matrix shape"
                        )
                    embeddings[second_pass_count:stop] = normalized
                    del encoded, normalized
                second_pass_count = stop
                if crossed_checkpoint(
                    second_pass_count - len(product_batch),
                    second_pass_count,
                    product_count,
                ):
                    checkpoint_embeddings()
                report_progress(
                    "csv_pass_2", second_pass_count, product_count
                )

        if (
            embedding_loader.last_report is None
            or second_pass_count != product_count
            or embedding_loader.last_report.rows_valid != product_count
        ):
            raise RuntimeError("CSV changed between streaming build passes")
        del batch_ids, product_batch, texts
        embeddings.flush()
        product_ids.flush()
        report_progress(
            "csv_pass_2", product_count, product_count, force=True
        )
        cache_path.write_text(
            json.dumps(
                {
                    "cache_key": cache_key,
                    "source_checksum": source_checksum,
                    "provider_id": provider.provider_id,
                    "dimension": provider.dimension,
                    "row_count": product_count,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError as exc:
            raise RuntimeError(
                "CatalogIndex.build_streaming requires scikit-learn"
            ) from exc

        kmeans_batch_size = min(
            product_count, max(batch_size, shard_count)
        )
        kmeans = MiniBatchKMeans(
            n_clusters=shard_count,
            random_state=random_seed,
            batch_size=kmeans_batch_size,
            n_init=3,
            max_no_improvement=20,
            reassignment_ratio=0.01,
        )
        kmeans_total = product_count * kmeans_epochs
        report_progress("kmeans", 0, kmeans_total, force=True)
        for epoch in range(kmeans_epochs):
            for start in range(0, product_count, kmeans_batch_size):
                stop = min(start + kmeans_batch_size, product_count)
                kmeans.partial_fit(
                    np.asarray(embeddings[start:stop], dtype=np.float32)
                )
                if crossed_checkpoint(start, stop, product_count):
                    checkpoint_embeddings()
                report_progress(
                    "kmeans",
                    epoch * product_count + stop,
                    kmeans_total,
                )
        report_progress(
            "kmeans", kmeans_total, kmeans_total, force=True
        )

        assignments = np.lib.format.open_memmap(
            output / "shard_assignments.npy",
            mode="w+",
            dtype=np.int64,
            shape=(product_count,),
        )
        centroid_sums = np.zeros(
            (shard_count, provider.dimension), dtype=np.float64
        )
        item_counts = np.zeros(shard_count, dtype=np.int64)
        report_progress("assignments", 0, product_count, force=True)
        for start in range(0, product_count, batch_size):
            stop = min(start + batch_size, product_count)
            vectors = np.asarray(
                embeddings[start:stop], dtype=np.float32
            )
            labels = np.asarray(kmeans.predict(vectors), dtype=np.int64)
            assignments[start:stop] = labels
            np.add.at(centroid_sums, labels, vectors)
            np.add.at(item_counts, labels, 1)
            del labels, vectors
            if crossed_checkpoint(start, stop, product_count):
                checkpoint_embeddings()
            report_progress("assignments", stop, product_count)

        empty_clusters_repaired = _repair_streaming_empty_clusters(
            assignments,
            embeddings,
            centroid_sums,
            item_counts,
            batch_size=batch_size,
        )
        assignments.flush()
        if np.any(item_counts == 0):
            raise RuntimeError("empty shard repair failed")
        report_progress(
            "assignments", product_count, product_count, force=True
        )

        centroids = normalize_rows(
            centroid_sums / item_counts[:, None],
            name="shard centroids",
        )
        euclidean_radii = np.zeros(shard_count, dtype=np.float64)
        angular = np.zeros(shard_count, dtype=np.float64)
        report_progress("radii", 0, product_count, force=True)
        for start in range(0, product_count, batch_size):
            stop = min(start + batch_size, product_count)
            labels = np.asarray(assignments[start:stop], dtype=np.int64)
            vectors_64 = np.asarray(
                embeddings[start:stop], dtype=np.float64
            )
            local_centroids = centroids[labels]
            distances = np.linalg.norm(
                vectors_64 - local_centroids, axis=1
            )
            cosines = np.einsum(
                "ij,ij->i", vectors_64, local_centroids
            )
            angles = np.arccos(np.clip(cosines, -1.0, 1.0))
            np.maximum.at(euclidean_radii, labels, distances)
            np.maximum.at(angular, labels, angles)
            del angles, cosines, distances, labels, local_centroids, vectors_64
            if crossed_checkpoint(start, stop, product_count):
                checkpoint_embeddings()
            report_progress("radii", stop, product_count)
        report_progress("radii", product_count, product_count, force=True)

        np.save(output / "shard_centroids.npy", centroids, allow_pickle=False)
        np.save(
            output / "shard_radii.npy",
            euclidean_radii,
            allow_pickle=False,
        )
        np.save(
            output / "shard_item_counts.npy",
            item_counts,
            allow_pickle=False,
        )
        np.save(
            output / "shard_angular_radii.npy",
            angular,
            allow_pickle=False,
        )
        shard_result = ShardBuildResult(
            assignments=assignments,
            centroids=centroids,
            euclidean_radii=euclidean_radii,
            item_counts=item_counts,
        )
        metadata = build_shard_metadata(
            shard_result,
            index_root="local_indexes",
            angular=angular,
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
        ).fit(centroids, metadata)
        router.save(str(output / "ssfr_router"))

        report_progress("local_indexes", 0, product_count, force=True)
        sorted_positions = np.argsort(assignments, kind="stable")
        shard_offsets = np.empty(shard_count + 1, dtype=np.int64)
        shard_offsets[0] = 0
        np.cumsum(item_counts, out=shard_offsets[1:])
        local_root = output / "local_indexes"
        actual_backend = local_index_backend
        indexed_count = 0
        for shard_id in range(shard_count):
            previous_indexed_count = indexed_count
            start = int(shard_offsets[shard_id])
            stop = int(shard_offsets[shard_id + 1])
            positions = sorted_positions[start:stop]
            index = LocalShardIndex(
                backend=local_index_backend,
                distance_metric="cosine",
            )
            index.build(embeddings[positions], product_ids[positions])
            index.save(local_root / f"shard_{shard_id:05d}")
            actual_backend = index.backend
            indexed_count += stop - start
            del index, positions
            if crossed_checkpoint(
                previous_indexed_count,
                indexed_count,
                product_count,
            ):
                checkpoint_embeddings()
            report_progress(
                "local_indexes", indexed_count, product_count
            )
        report_progress(
            "local_indexes", product_count, product_count, force=True
        )

        manifest = {
            "algorithm": "SSFR CSV Catalog",
            "version": "0.2.0",
            "created_at": datetime.now(UTC).isoformat(),
            "source_csv": str(source.resolve()),
            "source_checksum": source_checksum,
            "product_count": product_count,
            "embedding_provider": _embedding_provider_name(
                provider.provider_id
            ),
            "embedding_provider_id": provider.provider_id,
            "embedding_model": getattr(provider, "model_name", embedding_model or ""),
            "embedding_dimension": provider.dimension,
            "embedding_device": getattr(provider, "device", "cpu"),
            "embedding_precision": getattr(provider, "precision", "fp32"),
            "shard_count": shard_count,
            "bands": list(router.bands),
            "probe_shards": router.config.probe_shards,
            "ordering_method": ordering_method,
            "local_index_backend": actual_backend,
            "metadata_format": "csv",
            "embedding_cache_hit": cache_hit,
            "streaming_build": True,
            "batch_size": batch_size,
            "embedding_batch_size": embedding_batch_size,
            "kmeans_epochs": kmeans_epochs,
        }
        (output / "catalog_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        build_time_seconds = perf_counter() - started
        report = {
            "products_loaded": product_count,
            "invalid_rows": metadata_loader.last_report.rows_invalid,
            "embedding_dimension": provider.dimension,
            "embedding_device": getattr(provider, "device", "cpu"),
            "embedding_precision": getattr(provider, "precision", "fp32"),
            "embedding_cache_hit": cache_hit,
            "shards_built": shard_count,
            "bands": list(router.bands),
            "local_index_backend": actual_backend,
            "metadata_format": "csv",
            "streaming_build": True,
            "batch_size": batch_size,
            "embedding_batch_size": embedding_batch_size,
            "kmeans_epochs": kmeans_epochs,
            "empty_clusters_repaired": empty_clusters_repaired,
            "build_time_seconds": build_time_seconds,
            "artifact_path": str(output.resolve()),
        }

        embeddings.flush()
        product_ids.flush()
        assignments.flush()
        del sorted_positions
        del shard_result
        del embeddings
        del product_ids
        del assignments

        loaded_index: CatalogIndex | None = None
        if load_after_build:
            report_progress("load", 0, 1, force=True)
            loaded_index = cls.load(output)
            report_progress("load", 1, 1, force=True)
        report_progress(
            "complete", product_count, product_count, force=True
        )
        return loaded_index, report

    @classmethod
    def load(cls, path: str | Path) -> "CatalogIndex":
        directory = Path(path)
        manifest = json.loads(
            (directory / "catalog_manifest.json").read_text(encoding="utf-8")
        )
        if manifest["metadata_format"] == "parquet":
            frame = pd.read_parquet(directory / "products.parquet")
        else:
            frame = pd.read_csv(
                directory / "products.csv",
                dtype={"product_id": str},
            )
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
        if (
            price_min is None
            and price_max is None
            and category is None
            and brand is None
            and color is None
            and audience is None
            and not in_stock_only
        ):
            return mask
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
        all_results: bool = False,
        evaluate: bool = True,
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
            all_results=all_results,
            evaluate=evaluate,
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
        all_results: bool = False,
        evaluate: bool = True,
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
        filters_active = any(
            value is not None
            for value in (
                price_min,
                price_max,
                category,
                brand,
                color,
                audience,
            )
        ) or in_stock_only
        mask = self._filter_mask(
            price_min=price_min,
            price_max=price_max,
            category=category,
            brand=brand,
            color=color,
            audience=audience,
            in_stock_only=in_stock_only,
        )
        if all_results:
            # "All" is deliberately exact for the local catalog: every shard is
            # searched and every product passing the filters is returned. Using
            # pre-filtering also forces an exact scan inside approximate local
            # backends, so no matching item is dropped by HNSW/FAISS.
            top_k = max(1, int(np.count_nonzero(mask)))
            probe_shards = self.router.shard_count
            filter_strategy = "pre"
        vector = np.asarray(query_vector, dtype=np.float64)
        vector /= np.linalg.norm(vector)

        if filter_strategy == "pre" and filters_active:
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

        if all_results and filtered_ids.size:
            # Keep pagination stable even when multiple products have equal
            # similarity scores.
            stable_order = np.lexsort(
                (
                    np.asarray(filtered_ids, dtype=str),
                    -np.asarray(filtered_scores, dtype=np.float64),
                )
            )
            filtered_ids = np.asarray(filtered_ids)[stable_order]
            filtered_scores = np.asarray(filtered_scores)[stable_order]

        search = replace(
            raw,
            item_ids=np.asarray(filtered_ids),
            scores=np.asarray(filtered_scores, dtype=np.float64),
        )
        if evaluate:
            oracle_ids, _ = exact_global_search(
                vector,
                self.embeddings,
                self.product_ids,
                top_k,
                allowed_mask=mask,
            )
            recall = recall_at_k(search.item_ids.tolist(), oracle_ids.tolist(), top_k)
            precision = precision_at_k(
                search.item_ids.tolist(), oracle_ids.tolist(), top_k
            )
        elif all_results:
            # Complete all-shard exact retrieval is its own correctness guarantee;
            # avoid scanning the global embedding matrix a second time.
            oracle_ids = search.item_ids.copy()
            recall = 1.0
            precision = 1.0
        else:
            oracle_ids = np.empty(0, dtype=self.product_ids.dtype)
            recall = float("nan")
            precision = float("nan")
        products = tuple(self._product_by_id[str(item_id)] for item_id in search.item_ids)
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
