"""Measured benchmark over a real product CSV."""

from __future__ import annotations

import argparse
import json

from ssfr.benchmarking import run_csv_catalog_benchmark
from ssfr.console import configure_utf8_output


def main() -> None:
    configure_utf8_output()
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--artifacts", default="artifacts/benchmark_products")
    parser.add_argument("--output", default="reports")
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--probe-values", default="2,4,8")
    parser.add_argument("--bands", default="1,2,4")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-spectral-attempts", type=int, default=0)
    args = parser.parse_args()
    report = run_csv_catalog_benchmark(
        csv_path=args.csv,
        queries_path=args.queries,
        artifact_path=args.artifacts,
        output_directory=args.output,
        shards=args.shards,
        probe_values=tuple(int(value) for value in args.probe_values.split(",")),
        bands=tuple(int(value) for value in args.bands.split(",")),
        top_k=args.top_k,
        seed=args.seed,
        max_spectral_attempts=args.max_spectral_attempts,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
