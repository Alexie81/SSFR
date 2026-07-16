"""Synthetic e-commerce demo with 100,000 products by default."""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

from ssfr import DistributedSSFRSearch, LocalShardIndex, SSFRConfig, SSFRRouter
from ssfr.console import configure_utf8_output
from ssfr.distributed_search import exact_global_search
from ssfr.metrics import normalize_rows, recall_at_k
from ssfr.sharding import angular_radii, build_shard_metadata, build_shards


CATEGORIES = (
    "încălțăminte",
    "haine",
    "electronice",
    "casă și grădină",
    "produse pentru copii",
    "sport",
    "beauty",
    "auto",
)

QUERY_CATEGORY = {
    "adidași negri comozi pentru alergare": "încălțăminte",
    "telefon cu baterie mare și cameră bună": "electronice",
    "rochie verde elegantă pentru nuntă": "haine",
    "cărucior compact pentru nou-născut": "produse pentru copii",
    "laptop pentru programare și editare video": "electronice",
}


def main() -> None:
    configure_utf8_output()
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=100_000)
    parser.add_argument("--dimensions", type=int, default=96)
    parser.add_argument("--shards", type=int, default=256)
    parser.add_argument("--probe-shards", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--local-index",
        choices=("auto", "exact", "hnsw"),
        default="auto",
    )
    parser.add_argument("--latency-runs", type=int, default=100)
    parser.add_argument(
        "--query", default="adidași negri comozi pentru alergare"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.items < len(CATEGORIES):
        raise ValueError(f"items must be at least {len(CATEGORIES)}")
    if args.shards > args.items:
        raise ValueError("shards cannot exceed items")

    rng = np.random.default_rng(args.seed)
    category_centers = normalize_rows(
        rng.normal(size=(len(CATEGORIES), args.dimensions)), name="category centers"
    )
    category_ids = np.arange(args.items, dtype=np.int64) % len(CATEGORIES)
    rng.shuffle(category_ids)
    embeddings = normalize_rows(
        category_centers[category_ids]
        + 0.16 * rng.normal(size=(args.items, args.dimensions)),
        name="product embeddings",
    ).astype(np.float32)
    product_ids = np.asarray([f"SYN{value:07d}" for value in range(args.items)])
    brands = np.asarray([f"Brand{value % 32:02d}" for value in range(args.items)])
    prices = rng.uniform(20.0, 8_000.0, size=args.items)

    build_started = perf_counter()
    shard_result = build_shards(embeddings, args.shards, random_seed=args.seed)
    metadata = build_shard_metadata(
        shard_result,
        angular=angular_radii(embeddings, shard_result),
    )
    router = SSFRRouter(
        SSFRConfig(
            spectral_bands=(8, 16, 32, 64, 128),
            probe_shards=min(args.probe_shards, args.shards),
            exact_fallback=True,
            ordering_method="recursive_pca",
        )
    ).fit(shard_result.centroids, metadata)
    indexes = {}
    for shard_id in range(args.shards):
        positions = np.flatnonzero(shard_result.assignments == shard_id)
        local = LocalShardIndex(args.local_index, "cosine")
        local.build(embeddings[positions], product_ids[positions])
        indexes[shard_id] = local
    build_seconds = perf_counter() - build_started

    category_name = QUERY_CATEGORY.get(args.query, "sport")
    category_index = CATEGORIES.index(category_name)
    query_vector = normalize_rows(
        category_centers[category_index][None, :]
        + 0.04 * rng.normal(size=(1, args.dimensions)),
        name="query",
    )[0]
    searcher = DistributedSSFRSearch(router, indexes)
    result = searcher.search(
        query_vector,
        top_k=args.top_k,
        probe_shards=min(args.probe_shards, args.shards),
    )
    warm_latencies = []
    if args.latency_runs > 0:
        for _ in range(min(10, args.latency_runs)):
            searcher.search(
                query_vector,
                top_k=args.top_k,
                probe_shards=min(args.probe_shards, args.shards),
            )
        for _ in range(args.latency_runs):
            measured = searcher.search(
                query_vector,
                top_k=args.top_k,
                probe_shards=min(args.probe_shards, args.shards),
            )
            warm_latencies.append(measured.total_latency_ms)
    oracle_ids, _ = exact_global_search(
        query_vector, embeddings, product_ids, args.top_k
    )
    recall = recall_at_k(result.item_ids, oracle_ids, args.top_k)
    positions = {product_id: index for index, product_id in enumerate(product_ids)}

    print(f"Catalog: {args.items:,} synthetic products, {args.shards} shards")
    print(f"Offline build time: {build_seconds:.3f} s")
    print(f"Local index backend: {sorted({index.backend for index in indexes.values()})}")
    print(f"Query: {args.query}")
    print(f"Fourier band: {result.route.used_band}")
    print(f"Route mode: {result.route.route_mode}")
    print(f"Selected shards: {result.route.shard_ids.tolist()}")
    print(
        "Centroid certificate: "
        f"{result.route.centroid_ranking_certified}; "
        f"exact fallback: {result.route.used_exact_fallback}"
    )
    print(f"SSFR latency: {result.routing_latency_ms:.3f} ms")
    print(f"Local search latency: {result.local_search_latency_ms:.3f} ms")
    print(f"Total latency: {result.total_latency_ms:.3f} ms")
    if warm_latencies:
        print(
            f"Warm search P50/P95 over {len(warm_latencies)} runs: "
            f"{np.percentile(warm_latencies, 50):.3f}/"
            f"{np.percentile(warm_latencies, 95):.3f} ms"
        )
    print(f"Recall@{args.top_k} vs global exact oracle: {recall:.4f}")
    print("Top products:")
    for rank, (product_id, score) in enumerate(
        zip(result.item_ids, result.scores, strict=True), start=1
    ):
        position = positions[str(product_id)]
        print(
            f"{rank}. {product_id} — Categorie: {CATEGORIES[category_ids[position]]}; "
            f"Brand: {brands[position]}; Preț: {prices[position]:.2f} RON; "
            f"score {score:.4f}; shard {shard_result.assignments[position]}"
        )


if __name__ == "__main__":
    main()
