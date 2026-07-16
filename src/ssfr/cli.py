"""Command-line interface for catalog build, search, and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmarking import run_csv_catalog_benchmark
from .catalog import CatalogIndex, format_catalog_search
from .console import configure_utf8_output


def _integer_tuple(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list") from exc
    if not result:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ssfr")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build a persistent SSFR index from CSV")
    build.add_argument("--csv", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--shards", type=int, default=256)
    build.add_argument("--bands", type=_integer_tuple, default=(8, 16, 32, 64, 128))
    build.add_argument("--probe-shards", type=int, default=16)
    build.add_argument("--ordering", default="recursive_pca")
    build.add_argument(
        "--embedding-provider",
        choices=("hash", "auto", "sentence-transformers", "openai"),
        default="hash",
    )
    build.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    build.add_argument("--embedding-dimension", type=int, default=384)
    build.add_argument(
        "--local-index",
        choices=("auto", "exact", "hnsw", "faiss"),
        default="exact",
    )
    build.add_argument("--strict-csv", action="store_true")
    build.add_argument("--force-embeddings", action="store_true")
    build.add_argument("--seed", type=int, default=42)

    search = subparsers.add_parser("search", help="search a previously built catalog")
    search.add_argument("--index", required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=10)
    search.add_argument("--probe-shards", type=int)
    search.add_argument("--price-min", type=float)
    search.add_argument("--price-max", type=float)
    search.add_argument("--category")
    search.add_argument("--brand")
    search.add_argument("--color")
    search.add_argument("--audience")
    search.add_argument("--in-stock-only", action="store_true")
    search.add_argument(
        "--filter-strategy", choices=("pre", "post"), default="post"
    )
    search.add_argument(
        "--report", default="reports/csv_search_evaluation.csv"
    )
    search.add_argument("--json", action="store_true", dest="as_json")

    benchmark = subparsers.add_parser(
        "benchmark-csv", help="run measured baselines on a CSV catalog"
    )
    benchmark.add_argument("--csv", required=True)
    benchmark.add_argument("--queries", required=True)
    benchmark.add_argument("--artifacts", default="artifacts/benchmark_products")
    benchmark.add_argument("--output", default="reports")
    benchmark.add_argument("--shards", type=int, default=8)
    benchmark.add_argument("--probe-values", type=_integer_tuple, default=(2, 4, 8))
    benchmark.add_argument("--bands", type=_integer_tuple, default=(1, 2, 4))
    benchmark.add_argument("--top-k", type=int, default=5)
    benchmark.add_argument("--seed", type=int, default=42)
    return parser


def _command_build(args: argparse.Namespace) -> int:
    print("[SSFR] Loading and validating CSV...")
    _, report = CatalogIndex.build(
        args.csv,
        args.output,
        shard_count=args.shards,
        bands=args.bands,
        probe_shards=args.probe_shards,
        ordering_method=args.ordering,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_dimension=args.embedding_dimension,
        local_index_backend=args.local_index,
        tolerant_csv=not args.strict_csv,
        force_embeddings=args.force_embeddings,
        random_seed=args.seed,
    )
    print(f"[SSFR] CSV loaded: {report['products_loaded']:,} valid products")
    cache = "cache hit" if report["embedding_cache_hit"] else "generated"
    print(
        f"[SSFR] Embeddings {cache}: {report['products_loaded']:,} × "
        f"{report['embedding_dimension']}"
    )
    print(f"[SSFR] Shards built: {report['shards_built']}")
    print(f"[SSFR] Spectral router fitted with bands {report['bands']}")
    print(f"[SSFR] Local indexes built: {report['local_index_backend']}")
    print(f"[SSFR] Artifacts saved to {report['artifact_path']}")
    print(f"[SSFR] Total build time: {report['build_time_seconds']:.3f} s")
    return 0


def _command_search(args: argparse.Namespace) -> int:
    catalog = CatalogIndex.load(args.index)
    result = catalog.search_text(
        args.query,
        top_k=args.top_k,
        probe_shards=args.probe_shards,
        price_min=args.price_min,
        price_max=args.price_max,
        category=args.category,
        brand=args.brand,
        color=args.color,
        audience=args.audience,
        in_stock_only=args.in_stock_only,
        filter_strategy=args.filter_strategy,
        report_path=args.report,
    )
    if args.as_json:
        payload = {
            "query": result.query,
            "results": [
                {**product.to_dict(), "score": float(score)}
                for product, score in zip(result.products, result.scores, strict=True)
            ],
            "route": {
                "selected_shards": result.search.route.shard_ids.tolist(),
                "used_band": result.search.route.used_band,
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
            "precision_at_k": result.precision_at_k,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(format_catalog_search(result))
    return 0


def _command_benchmark(args: argparse.Namespace) -> int:
    report = run_csv_catalog_benchmark(
        csv_path=args.csv,
        queries_path=args.queries,
        artifact_path=args.artifacts,
        output_directory=args.output,
        shards=args.shards,
        probe_values=args.probe_values,
        bands=args.bands,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_utf8_output()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "build":
        return _command_build(args)
    if args.command == "search":
        return _command_search(args)
    if args.command == "benchmark-csv":
        return _command_benchmark(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
