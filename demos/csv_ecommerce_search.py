"""One-command build/search demo for a real product CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from ssfr.catalog import CatalogIndex, format_catalog_search
from ssfr.console import configure_utf8_output
from ssfr.performance import limit_native_threads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/products.csv")
    parser.add_argument("--index", default="artifacts/products")
    parser.add_argument(
        "--query",
        default="adidași negri impermeabili pentru alergare pe munte",
    )
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--bands", default="1,2,4")
    parser.add_argument("--probe-shards", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--local-index", default="exact")
    parser.add_argument("--native-threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    configure_utf8_output()
    args = parse_args()
    native_thread_limiter = limit_native_threads(args.native_threads)
    do_build = args.build or (not args.build and not args.search)
    do_search = args.search or (not args.build and not args.search)
    bands = tuple(int(value) for value in args.bands.split(","))
    if do_build:
        _, report = CatalogIndex.build(
            args.csv,
            args.index,
            shard_count=args.shards,
            bands=bands,
            probe_shards=args.probe_shards,
            embedding_provider="hash",
            local_index_backend=args.local_index,
        )
        print(f"Built {report['products_loaded']} products in {report['build_time_seconds']:.3f}s")
    if do_search:
        if not Path(args.index, "catalog_manifest.json").exists():
            raise FileNotFoundError("index does not exist; run with --build first")
        catalog = CatalogIndex.load(args.index)
        result = catalog.search_text(
            args.query,
            top_k=args.top_k,
            probe_shards=args.probe_shards,
        )
        print(format_catalog_search(result))
    del native_thread_limiter


if __name__ == "__main__":
    main()
