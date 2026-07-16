"""Generate an explicitly labelled billion-scale capacity estimate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ssfr.capacity import estimate_scale
from ssfr.console import configure_utf8_output


def main() -> None:
    configure_utf8_output()
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=1_000_000_000)
    parser.add_argument("--shards", type=int, default=16_384)
    parser.add_argument("--dimensions", type=int, default=768)
    parser.add_argument("--band", type=int, default=256)
    parser.add_argument("--probe-shards", type=int, default=32)
    parser.add_argument("--parallel-shards", type=int, default=32)
    parser.add_argument("--router-ms", type=float)
    parser.add_argument("--local-shard-p95-ms", type=float)
    parser.add_argument("--network-p95-ms", type=float)
    parser.add_argument("--merge-ms", type=float)
    parser.add_argument("--output", default="reports/billion_scale_estimate")
    args = parser.parse_args()
    estimate = estimate_scale(
        items=args.items,
        shards=args.shards,
        dimensions=args.dimensions,
        fourier_band=args.band,
        probe_shards=args.probe_shards,
        parallel_shard_requests=args.parallel_shards,
        measured_router_ms=args.router_ms,
        measured_local_shard_p95_ms=args.local_shard_p95_ms,
        measured_network_p95_ms=args.network_p95_ms,
        measured_merge_ms=args.merge_ms,
    )
    payload = estimate.to_dict()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "estimate.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Billion-scale capacity estimate",
        "",
        "> Estimate only. No one-billion-vector corpus was physically indexed.",
        "",
        f"- Items: {payload['estimated_items']:,}",
        f"- Shards: {payload['shard_count']:,}",
        f"- Mean items/shard: {payload['items_per_shard_mean']:,.1f}",
        f"- Probed shards: {payload['probe_shards']:,}",
        f"- Routed item universe: {payload['routed_item_universe']:,.1f}",
        f"- Shard fan-out reduction: {payload['shard_reduction_factor']:,.1f}×",
        f"- Exact centroid multiply-adds: {payload['exact_centroid_multiply_adds']:,}",
        (
            "- Spectral projection complex terms: "
            f"{payload['spectral_projection_complex_terms']:,}"
        ),
        f"- Approximate IFFT operations: {payload['approximate_ifft_operations']:,}",
        (
            "- Float32 centroid matrix: "
            f"{payload['centroid_matrix_float32_bytes'] / 2**20:.2f} MiB"
        ),
        (
            "- Complex64 spectral payload: "
            f"{payload['spectral_payload_complex64_bytes'] / 2**20:.2f} MiB"
        ),
        (
            "- Float32 residuals: "
            f"{payload['residuals_float32_bytes'] / 2**20:.2f} MiB"
        ),
    ]
    if payload["estimated_end_to_end_latency_ms"] is not None:
        lines.extend(
            [
                "",
                "## Latency model from supplied measurements",
                "",
                (
                    f"- Router: {payload['estimated_router_latency_ms']:.3f} ms"
                ),
                (
                    "- Local + network waves: "
                    f"{payload['estimated_local_and_network_latency_ms']:.3f} ms"
                ),
                f"- Merge: {payload['estimated_merge_latency_ms']:.3f} ms",
                (
                    "- Estimated end-to-end: "
                    f"{payload['estimated_end_to_end_latency_ms']:.3f} ms"
                ),
            ]
        )
    (output / "estimate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
