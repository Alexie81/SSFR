"""Physical router-only benchmark for large shard-centroid matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ssfr import SSFRConfig, SSFRRouter
from ssfr.console import configure_utf8_output
from ssfr.metrics import latency_summary, normalize_rows
from ssfr.performance import limit_native_threads


def main() -> None:
    configure_utf8_output()
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=16_384)
    parser.add_argument("--dimensions", type=int, default=768)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--probe-shards", type=int, default=32)
    parser.add_argument("--bands", default="32,64,128,256,512")
    parser.add_argument("--max-spectral-attempts", type=int, default=0)
    parser.add_argument("--native-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="reports/large_router")
    args = parser.parse_args()
    limiter = limit_native_threads(args.native_threads)
    rng = np.random.default_rng(args.seed)
    positions = np.arange(args.shards, dtype=np.float64)[:, None]
    frequencies = np.arange(1, 9, dtype=np.float64)[None, :]
    features = np.concatenate(
        (
            np.sin(2 * np.pi * positions * frequencies / args.shards),
            np.cos(2 * np.pi * positions * frequencies / args.shards),
        ),
        axis=1,
    )
    basis = rng.normal(size=(features.shape[1], args.dimensions))
    centroids = normalize_rows(
        features @ basis
        + 0.002 * rng.normal(size=(args.shards, args.dimensions)),
        name="centroids",
    )
    query_rows = rng.integers(0, args.shards, size=args.queries)
    queries = normalize_rows(
        centroids[query_rows]
        + 0.01 * rng.normal(size=(args.queries, args.dimensions)),
        name="queries",
    )
    started = perf_counter()
    router = SSFRRouter(
        SSFRConfig(
            spectral_bands=tuple(int(value) for value in args.bands.split(",")),
            probe_shards=args.probe_shards,
            exact_fallback=True,
            ordering_method="identity",
            max_spectral_attempts=args.max_spectral_attempts,
        )
    ).fit(centroids)
    fit_seconds = perf_counter() - started
    for query in queries[: min(20, len(queries))]:
        router.route(query)
    routes = [router.route(query) for query in queries]
    report = {
        "benchmark": "large_router_only",
        "physical_centroids": args.shards,
        "embedding_dimension": args.dimensions,
        "physical_item_vectors": 0,
        "queries": args.queries,
        "probe_shards": args.probe_shards,
        "bands": list(router.bands),
        "max_spectral_attempts": args.max_spectral_attempts,
        "native_threads": args.native_threads,
        "fit_seconds": fit_seconds,
        "latency_ms": latency_summary([route.latency_ms for route in routes]),
        "fraction_certified": float(
            np.mean([route.centroid_ranking_certified for route in routes])
        ),
        "fraction_exact_fallback": float(
            np.mean([route.used_exact_fallback for route in routes])
        ),
        "route_modes": {
            mode: sum(route.route_mode == mode for route in routes)
            for mode in sorted({route.route_mode for route in routes})
        },
        "memory": router.memory_report(),
        "warning": (
            "Router-only physical benchmark; no item-vector corpus or network "
            "shards were loaded."
        ),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    del limiter


if __name__ == "__main__":
    main()
