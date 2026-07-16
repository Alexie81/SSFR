"""Run the standard comparison across increasing shard counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ssfr.benchmarking import run_synthetic_benchmark
from ssfr.console import configure_utf8_output


def main() -> None:
    configure_utf8_output()
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", default="128,256,512,1024")
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--output", default="reports/scale")
    args = parser.parse_args()
    reports = {}
    for shards in (int(value) for value in args.shards.split(",")):
        reports[str(shards)] = run_synthetic_benchmark(
            shards=shards,
            dimensions=args.dimensions,
            queries=args.queries,
            probe_shards=min(16, shards),
            output_directory=Path(args.output) / str(shards),
        )
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "scale_summary.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
