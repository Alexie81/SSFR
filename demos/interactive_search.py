"""Interactive terminal search over a prebuilt catalog."""

from __future__ import annotations

import argparse

from ssfr.catalog import CatalogIndex, format_catalog_search
from ssfr.console import configure_utf8_output


def main() -> None:
    configure_utf8_output()
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/products")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--probe-shards", type=int, default=4)
    args = parser.parse_args()
    catalog = CatalogIndex.load(args.index)
    while True:
        query = input("Query (empty to quit): ").strip()
        if not query:
            return
        result = catalog.search_text(
            query, top_k=args.top_k, probe_shards=args.probe_shards
        )
        print(format_catalog_search(result))


if __name__ == "__main__":
    main()
