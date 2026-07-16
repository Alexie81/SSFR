"""Compare SSFR with all required centroid-routing baselines."""

from __future__ import annotations

import argparse
import json

from ssfr.benchmarking import run_synthetic_benchmark
from ssfr.console import configure_utf8_output


def main() -> None:
    configure_utf8_output()
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=1_000_000)
    parser.add_argument("--shards", type=int, default=1024)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--probe-shards", type=int, default=16)
    parser.add_argument("--bands", default="8,16,32,64,128")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    report = run_synthetic_benchmark(
        shards=args.shards,
        dimensions=args.dimensions,
        queries=args.queries,
        probe_shards=args.probe_shards,
        bands=tuple(int(value) for value in args.bands.split(",")),
        low_rank=args.rank,
        seed=args.seed,
        output_directory=args.output,
        estimated_catalog_items=args.items,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
