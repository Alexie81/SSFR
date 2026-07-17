"""Command-line interface for catalog build, search, and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmarking import run_csv_catalog_benchmark
from .catalog import CatalogIndex, format_catalog_search
from .console import configure_utf8_output
from .interactive import InteractiveState, run_interactive
from .performance import limit_native_threads


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
    build.add_argument("--max-spectral-attempts", type=int, default=2)
    build.add_argument("--native-threads", type=int, default=1)

    search = subparsers.add_parser("search", help="search a previously built catalog")
    search.add_argument("--index", required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=10)
    search.add_argument("--probe-shards", type=int)
    search.add_argument(
        "--all-results",
        action="store_true",
        help="return every filtered product from every shard, sorted by score",
    )
    search.add_argument(
        "--all-shards",
        action="store_true",
        help="search the requested top-k in every shard",
    )
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
    search.add_argument("--native-threads", type=int, default=1)

    interactive = subparsers.add_parser(
        "interactive",
        aliases=("live", "cauta"),
        help="open a friendly live search session",
    )
    interactive.add_argument("--index", default="artifacts/products")
    interactive.add_argument("--top-k", type=int, default=10)
    interactive.add_argument("--probe-shards", type=int)
    interactive.add_argument(
        "--top-only",
        action="store_true",
        help="start in fast top-k mode instead of exact all-results mode",
    )
    interactive.add_argument("--page-size", type=int, default=10)
    interactive.add_argument("--price-min", type=float)
    interactive.add_argument("--price-max", type=float)
    interactive.add_argument("--category")
    interactive.add_argument("--brand")
    interactive.add_argument("--color")
    interactive.add_argument("--audience")
    interactive.add_argument("--in-stock-only", action="store_true")
    interactive.add_argument(
        "--report", default="reports/csv_search_evaluation.csv"
    )
    interactive.add_argument("--native-threads", type=int, default=1)

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
    benchmark.add_argument("--native-threads", type=int, default=1)
    benchmark.add_argument("--max-spectral-attempts", type=int, default=0)
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
        max_spectral_attempts=args.max_spectral_attempts,
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
    probe_shards = (
        catalog.router.shard_count
        if args.all_results or args.all_shards
        else args.probe_shards
    )
    result = catalog.search_text(
        args.query,
        top_k=args.top_k,
        probe_shards=probe_shards,
        all_results=args.all_results,
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
            "returned_results": len(result.products),
            "all_results_requested": args.all_results,
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
            "precision_at_k": result.precision_at_k,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(format_catalog_search(result))
    return 0


def _command_interactive(args: argparse.Namespace) -> int:
    catalog = CatalogIndex.load(args.index)
    state = InteractiveState(
        all_results=not args.top_only,
        top_k=args.top_k,
        probe_shards=args.probe_shards,
        page_size=args.page_size,
        price_min=args.price_min,
        price_max=args.price_max,
        category=args.category,
        brand=args.brand,
        color=args.color,
        audience=args.audience,
        in_stock_only=args.in_stock_only,
    )
    return run_interactive(
        catalog,
        state=state,
        report_path=args.report,
    )


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
        max_spectral_attempts=args.max_spectral_attempts,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_utf8_output()
    parser = build_parser()
    args = parser.parse_args(argv)
    native_thread_limiter = limit_native_threads(args.native_threads)
    if args.command == "build":
        result = _command_build(args)
        del native_thread_limiter
        return result
    if args.command == "search":
        result = _command_search(args)
        del native_thread_limiter
        return result
    if args.command in {"interactive", "live", "cauta"}:
        result = _command_interactive(args)
        del native_thread_limiter
        return result
    if args.command == "benchmark-csv":
        result = _command_benchmark(args)
        del native_thread_limiter
        return result
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
