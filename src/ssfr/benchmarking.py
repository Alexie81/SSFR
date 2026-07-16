"""Reproducible synthetic and CSV benchmark runners."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np

from .baselines import (
    HierarchicalCentroidRouter,
    LowRankCentroidRouter,
    exhaustive_centroid_route,
    random_shard_route,
)
from .catalog import CatalogIndex
from .certificates import top_indices
from .local_index import LocalShardIndex, backend_available
from .metrics import latency_summary, normalize_rows, recall_at_k
from .router import SSFRRouter
from .types import SSFRConfig


def _write_reports(report: dict[str, Any], output_directory: str | Path) -> None:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# SSFR benchmark report",
        "",
        f"Generated from measured runs. Seed: `{report.get('seed', 'n/a')}`.",
        "",
    ]
    if "configuration" in report:
        lines.extend(["## Configuration", "", "```json"])
        lines.append(json.dumps(report["configuration"], indent=2))
        lines.extend(["```", ""])
    lines.extend(
        [
            "## Results",
            "",
            "| Method | Mean ms | P95 ms | Centroid recall | Vector recall | "
            "Certified | Fallback | Memory bytes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    def metric(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.4f}"

    for name, result in report.get("methods", {}).items():
        latency = result.get("latency_ms", {})
        lines.append(
            "| {name} | {mean:.4f} | {p95:.4f} | {centroid} | {vector} | "
            "{cert:.4f} | {fallback:.4f} | {memory} |".format(
                name=name,
                mean=float(latency.get("mean", 0.0)),
                p95=float(latency.get("p95", 0.0)),
                centroid=metric(result.get("centroid_top_b_recall")),
                vector=metric(result.get("vector_top_k_recall")),
                cert=float(result.get("fraction_certified", 0.0)),
                fallback=float(result.get("fraction_exact_fallback", 0.0)),
                memory=int(result.get("memory_bytes", 0)),
            )
        )
    if report.get("kill_criteria"):
        lines.extend(["", "## Triggered kill criteria", ""])
        lines.extend(f"- {item}" for item in report["kill_criteria"])
    (output / "benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_synthetic(report: dict[str, Any], output_directory: str | Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    methods = report["methods"]
    names = list(methods)
    latencies = [methods[name]["latency_ms"]["mean"] for name in names]
    recalls = [methods[name]["centroid_top_b_recall"] for name in names]
    plots = Path(output_directory) / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(names, latencies)
    axis.set_ylabel("Mean routing latency (ms)")
    axis.tick_params(axis="x", rotation=40)
    figure.tight_layout()
    figure.savefig(plots / "ssfr_vs_baselines_latency.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(names, recalls)
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Centroid top-B recall")
    axis.tick_params(axis="x", rotation=40)
    figure.tight_layout()
    figure.savefig(plots / "ssfr_vs_baselines_recall.png", dpi=150)
    plt.close(figure)

    ssfr = methods.get("ssfr")
    if ssfr and ssfr.get("spectral_energy"):
        bands = [int(value) for value in ssfr["spectral_energy"]]
        energy = [ssfr["spectral_energy"][str(value)] for value in bands]
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(bands, energy, marker="o")
        axis.set_xlabel("Fourier band")
        axis.set_ylabel("Cumulative spectral energy")
        axis.set_ylim(0, 1.02)
        figure.tight_layout()
        figure.savefig(plots / "spectral_energy.png", dpi=150)
        plt.close(figure)


def run_synthetic_benchmark(
    *,
    shards: int = 1024,
    dimensions: int = 128,
    queries: int = 100,
    probe_shards: int = 16,
    bands: tuple[int, ...] = (8, 16, 32, 64, 128),
    low_rank: int = 32,
    seed: int = 42,
    output_directory: str | Path = "reports",
    estimated_catalog_items: int | None = None,
) -> dict[str, Any]:
    if probe_shards > shards:
        raise ValueError("probe_shards cannot exceed shards")
    rng = np.random.default_rng(seed)
    theme_count = min(32, max(4, int(np.sqrt(shards))))
    themes = normalize_rows(rng.normal(size=(theme_count, dimensions)), name="themes")
    labels = np.arange(shards) % theme_count
    centroids = normalize_rows(
        themes[labels] + 0.12 * rng.normal(size=(shards, dimensions)),
        name="centroids",
    )
    rng.shuffle(centroids, axis=0)
    query_labels = rng.integers(0, theme_count, size=queries)
    query_vectors = normalize_rows(
        themes[query_labels] + 0.08 * rng.normal(size=(queries, dimensions)),
        name="queries",
    )
    oracle = [
        exhaustive_centroid_route(centroids, query, probe_shards)
        for query in query_vectors
    ]

    pca_router = LowRankCentroidRouter(centroids, low_rank, centered=True)
    svd_router = LowRankCentroidRouter(centroids, low_rank, centered=False)
    hierarchy = HierarchicalCentroidRouter(centroids, random_seed=seed)
    ssfr = SSFRRouter(
        SSFRConfig(
            spectral_bands=bands,
            probe_shards=probe_shards,
            exact_fallback=True,
            ordering_method="recursive_pca",
            random_seed=seed,
        )
    ).fit(centroids)
    ssfr_identity = SSFRRouter(
        SSFRConfig(
            spectral_bands=bands,
            probe_shards=probe_shards,
            exact_fallback=True,
            ordering_method="identity",
            random_seed=seed,
        )
    ).fit(centroids)
    ssfr_no_certificate = SSFRRouter(
        SSFRConfig(
            spectral_bands=(bands[0],),
            probe_shards=probe_shards,
            exact_fallback=False,
            ordering_method="recursive_pca",
            random_seed=seed,
        )
    ).fit(centroids)

    results: dict[str, dict[str, Any]] = {}

    def evaluate(
        name: str,
        route: Callable[[np.ndarray, int], np.ndarray],
        *,
        memory_bytes: int,
    ) -> None:
        latencies = []
        recalls = []
        for query, expected in zip(query_vectors, oracle, strict=True):
            started = perf_counter()
            selected = route(query, probe_shards)
            latencies.append((perf_counter() - started) * 1000.0)
            recalls.append(recall_at_k(selected, expected, probe_shards))
        results[name] = {
            "latency_ms": latency_summary(latencies),
            "queries_per_second": (
                1000.0 / float(np.mean(latencies)) if np.mean(latencies) > 0 else 0.0
            ),
            "centroid_top_b_recall": float(np.mean(recalls)),
            "fraction_certified": 0.0,
            "fraction_exact_fallback": 0.0,
            "mean_used_spectral_band": 0.0,
            "memory_bytes": int(memory_bytes),
        }

    evaluate(
        "exhaustive",
        lambda query, count: exhaustive_centroid_route(centroids, query, count),
        memory_bytes=centroids.nbytes,
    )
    random_rng = np.random.default_rng(seed + 1)
    evaluate(
        "random",
        lambda _query, count: random_shard_route(shards, count, random_rng),
        memory_bytes=0,
    )
    evaluate("pca_low_rank", pca_router.route, memory_bytes=pca_router.memory_bytes)
    evaluate("truncated_svd", svd_router.route, memory_bytes=svd_router.memory_bytes)
    evaluate("hierarchical", hierarchy.route, memory_bytes=hierarchy.memory_bytes)
    evaluate(
        "ivf_centroid_probing",
        lambda query, count: exhaustive_centroid_route(centroids, query, count),
        memory_bytes=centroids.nbytes,
    )

    def evaluate_ssfr(name: str, router: SSFRRouter) -> None:
        latencies = []
        recalls = []
        certified = []
        fallback = []
        used_bands = []
        for query, expected in zip(query_vectors, oracle, strict=True):
            route = router.route(query, probe_shards)
            latencies.append(route.latency_ms)
            recalls.append(recall_at_k(route.shard_ids, expected, probe_shards))
            certified.append(route.centroid_ranking_certified)
            fallback.append(route.used_exact_fallback)
            used_bands.append(route.used_band)
        results[name] = {
            "latency_ms": latency_summary(latencies),
            "queries_per_second": (
                1000.0 / float(np.mean(latencies)) if np.mean(latencies) > 0 else 0.0
            ),
            "centroid_top_b_recall": float(np.mean(recalls)),
            "fraction_certified": float(np.mean(certified)),
            "fraction_exact_fallback": float(np.mean(fallback)),
            "mean_used_spectral_band": float(np.mean(used_bands)),
            "memory_bytes": router.memory_report()["router_bytes_with_exact_fallback"],
            "spectral_energy": {
                str(band): energy
                for band, energy in router.spectral_energy_report().items()
            },
            "ordering": router.ordering_report(),
        }

    evaluate_ssfr("ssfr", ssfr)
    evaluate_ssfr("ssfr_without_ordering", ssfr_identity)
    no_certificate_band = ssfr_no_certificate.bands[0]
    evaluate(
        "ssfr_without_certificate",
        lambda query, count: top_indices(
            ssfr_no_certificate._approximate_scores(query, no_certificate_band),
            count,
        ),
        memory_bytes=ssfr_no_certificate.memory_report()[
            "router_bytes_without_exact_centroids"
        ],
    )
    results["ssfr_without_certificate"]["mean_used_spectral_band"] = float(
        no_certificate_band
    )
    results["ssfr_without_certificate"]["spectral_energy"] = {
        str(band): energy
        for band, energy in ssfr_no_certificate.spectral_energy_report().items()
    }

    kill_criteria = []
    ssfr_result = results["ssfr"]
    if ssfr_result["mean_used_spectral_band"] >= shards * 0.4:
        kill_criteria.append("The mean Fourier band is close to the shard count.")
    if ssfr_result["fraction_exact_fallback"] > 0.9:
        kill_criteria.append("More than 90% of queries used exact fallback.")
    if (
        ssfr_result["latency_ms"]["mean"]
        >= results["exhaustive"]["latency_ms"]["mean"]
    ):
        kill_criteria.append("Measured SSFR routing was not faster than matrix multiplication.")
    if (
        ssfr.memory_report()["router_bytes_without_exact_centroids"]
        > centroids.nbytes
    ):
        kill_criteria.append("Spectral payload plus residuals exceeded centroid matrix memory.")

    report = {
        "benchmark": "synthetic_centroid_routing",
        "seed": seed,
        "configuration": {
            "shards": shards,
            "dimensions": dimensions,
            "queries": queries,
            "probe_shards": probe_shards,
            "bands": list(bands),
            "low_rank": low_rank,
            "physical_vectors_loaded": False,
            "billion_scale_estimate": False,
            "estimated_catalog_items": estimated_catalog_items,
            "estimated_only": estimated_catalog_items is not None,
        },
        "methods": results,
        "kill_criteria": kill_criteria,
    }
    _write_reports(report, output_directory)
    _plot_synthetic(report, output_directory)
    return report


def _merge_local_results(
    catalog: CatalogIndex,
    query: np.ndarray,
    shard_ids: np.ndarray,
    top_k: int,
) -> np.ndarray:
    candidates: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for shard_id in shard_ids:
        local_ids, local_scores = catalog.local_indexes[int(shard_id)].search(query, top_k)
        candidates.append(local_ids)
        scores.append(local_scores)
    all_ids = np.concatenate(candidates)
    all_scores = np.concatenate(scores)
    return all_ids[top_indices(all_scores, min(top_k, all_scores.size))]


def run_csv_catalog_benchmark(
    *,
    csv_path: str | Path,
    queries_path: str | Path,
    artifact_path: str | Path = "artifacts/benchmark_products",
    output_directory: str | Path = "reports",
    shards: int = 8,
    probe_values: tuple[int, ...] = (2, 4, 8),
    bands: tuple[int, ...] = (1, 2, 4),
    top_k: int = 5,
    seed: int = 42,
    max_spectral_attempts: int | None = 0,
) -> dict[str, Any]:
    local_backend = "hnsw" if backend_available("hnsw") else "exact"
    catalog, build_report = CatalogIndex.build(
        csv_path,
        artifact_path,
        shard_count=shards,
        bands=bands,
        probe_shards=min(probe_values),
        embedding_provider="hash",
        local_index_backend=local_backend,
        random_seed=seed,
        max_spectral_attempts=max_spectral_attempts,
    )
    with Path(queries_path).open("r", encoding="utf-8-sig", newline="") as handle:
        query_texts = [
            row["query"].strip()
            for row in csv.DictReader(handle)
            if row.get("query", "").strip()
        ]
    hierarchy = HierarchicalCentroidRouter(
        catalog.router.centroids, random_seed=seed
    )

    methods: dict[str, dict[str, list[Any]]] = {}
    by_probe: dict[str, dict[str, dict[str, list[Any]]]] = {}
    global_hnsw: LocalShardIndex | None = None
    if backend_available("hnsw"):
        global_hnsw = LocalShardIndex("hnsw", "cosine")
        global_hnsw.build(catalog.embeddings, catalog.product_ids)

    def record(
        name: str,
        probe: int,
        latency_ms: float,
        vector_recall: float,
        *,
        certified: bool = False,
        fallback: bool = False,
        used_band: int = 0,
        shards_accessed: int = 0,
        candidates: int = 0,
        route_mode: str = "",
    ) -> None:
        values = methods.setdefault(
            name,
            {
                "latency": [],
                "vector_recall": [],
                "certified": [],
                "fallback": [],
                "used_band": [],
                "shards_accessed": [],
                "candidates": [],
                "route_mode": [],
            },
        )
        probe_values = by_probe.setdefault(str(probe), {}).setdefault(
            name,
            {
                "latency": [],
                "vector_recall": [],
                "certified": [],
                "fallback": [],
                "used_band": [],
                "shards_accessed": [],
                "candidates": [],
                "route_mode": [],
            },
        )
        for destination in (values, probe_values):
            destination["latency"].append(latency_ms)
            destination["vector_recall"].append(vector_recall)
            destination["certified"].append(certified)
            destination["fallback"].append(fallback)
            destination["used_band"].append(used_band)
            destination["shards_accessed"].append(shards_accessed)
            destination["candidates"].append(candidates)
            destination["route_mode"].append(route_mode)

    for probe in probe_values:
        if probe > shards:
            continue
        for text in query_texts:
            vector = catalog.provider.encode_query(text).astype(np.float64)
            oracle_started = perf_counter()
            oracle_scores = catalog.embeddings @ vector
            oracle_ids = catalog.product_ids[
                top_indices(oracle_scores, min(top_k, oracle_scores.size))
            ]
            oracle_ms = (perf_counter() - oracle_started) * 1000.0
            record(
                "global_exact",
                probe,
                oracle_ms,
                1.0,
                shards_accessed=shards,
                candidates=len(catalog.products),
            )

            if global_hnsw is not None:
                started = perf_counter()
                hnsw_ids, _ = global_hnsw.search(vector, top_k)
                elapsed = (perf_counter() - started) * 1000.0
                record(
                    "global_hnsw",
                    probe,
                    elapsed,
                    recall_at_k(hnsw_ids, oracle_ids, top_k),
                    shards_accessed=1,
                    candidates=len(catalog.products),
                )

            started = perf_counter()
            exact_shards = exhaustive_centroid_route(
                catalog.router.centroids, vector, probe
            )
            exact_ids = _merge_local_results(catalog, vector, exact_shards, top_k)
            elapsed = (perf_counter() - started) * 1000.0
            record(
                "exhaustive_centroid_plus_local_hnsw"
                if local_backend == "hnsw"
                else "exhaustive_centroid_plus_local_exact",
                probe,
                elapsed,
                recall_at_k(exact_ids, oracle_ids, top_k),
                certified=True,
                shards_accessed=probe,
                candidates=sum(
                    catalog.local_indexes[int(shard_id)].item_count
                    for shard_id in exact_shards
                ),
            )

            started = perf_counter()
            hierarchical_shards = hierarchy.route(vector, probe)
            hierarchical_ids = _merge_local_results(
                catalog, vector, hierarchical_shards, top_k
            )
            elapsed = (perf_counter() - started) * 1000.0
            record(
                "hierarchical_plus_local_hnsw"
                if local_backend == "hnsw"
                else "hierarchical_plus_local_exact",
                probe,
                elapsed,
                recall_at_k(hierarchical_ids, oracle_ids, top_k),
                shards_accessed=probe,
                candidates=sum(
                    catalog.local_indexes[int(shard_id)].item_count
                    for shard_id in hierarchical_shards
                ),
            )

            result = catalog.search_text(
                text,
                top_k=top_k,
                probe_shards=probe,
                report_path=None,
            )
            record(
                "ssfr_plus_local_hnsw"
                if local_backend == "hnsw"
                else "ssfr_plus_local_exact",
                probe,
                result.search.total_latency_ms,
                result.recall_at_k,
                certified=result.search.route.centroid_ranking_certified,
                fallback=result.search.route.used_exact_fallback,
                used_band=result.search.route.used_band,
                shards_accessed=result.search.shards_accessed,
                candidates=result.search.candidate_vectors_evaluated,
                route_mode=result.search.route.route_mode,
            )

    def summarize(values: dict[str, list[Any]]) -> dict[str, Any]:
        latency = latency_summary(values["latency"])
        return {
            "latency_ms": latency,
            "queries_per_second": (
                1000.0 / latency["mean"] if latency["mean"] > 0 else 0.0
            ),
            "centroid_top_b_recall": None,
            "vector_top_k_recall": float(np.mean(values["vector_recall"])),
            "fraction_certified": float(np.mean(values["certified"])),
            "fraction_exact_fallback": float(np.mean(values["fallback"])),
            "mean_used_spectral_band": float(np.mean(values["used_band"])),
            "mean_shards_accessed": float(np.mean(values["shards_accessed"])),
            "mean_local_candidates_available": float(np.mean(values["candidates"])),
            "memory_bytes": 0,
            "route_mode_counts": {
                mode: values["route_mode"].count(mode)
                for mode in sorted(set(values["route_mode"]))
                if mode
            },
        }

    summarized = {name: summarize(values) for name, values in methods.items()}
    probe_summary = {
        probe: {name: summarize(values) for name, values in probe_methods.items()}
        for probe, probe_methods in by_probe.items()
    }
    ssfr_name = (
        "ssfr_plus_local_hnsw" if local_backend == "hnsw" else "ssfr_plus_local_exact"
    )
    exact_name = (
        "exhaustive_centroid_plus_local_hnsw"
        if local_backend == "hnsw"
        else "exhaustive_centroid_plus_local_exact"
    )
    local_array_bytes = int(
        sum(
            (0 if index.vectors is None else index.vectors.nbytes)
            + (0 if index.ids is None else index.ids.nbytes)
            for index in catalog.local_indexes.values()
        )
    )
    global_array_bytes = int(catalog.embeddings.nbytes + catalog.product_ids.nbytes)
    summarized["global_exact"]["memory_bytes"] = global_array_bytes
    if "global_hnsw" in summarized:
        summarized["global_hnsw"]["memory_bytes"] = global_array_bytes
        summarized["global_hnsw"][
            "memory_note"
        ] = "Python arrays only; native HNSW graph allocation is not included."
    summarized[exact_name]["memory_bytes"] = int(
        local_array_bytes + catalog.router.centroids.nbytes
    )
    hierarchy_name = (
        "hierarchical_plus_local_hnsw"
        if local_backend == "hnsw"
        else "hierarchical_plus_local_exact"
    )
    summarized[hierarchy_name]["memory_bytes"] = int(
        local_array_bytes + hierarchy.memory_bytes
    )
    summarized[ssfr_name]["memory_bytes"] = int(
        local_array_bytes
        + catalog.router.memory_report()["router_bytes_with_exact_fallback"]
    )
    for probe_methods in probe_summary.values():
        for name, values in probe_methods.items():
            values["memory_bytes"] = summarized[name]["memory_bytes"]
            if "memory_note" in summarized[name]:
                values["memory_note"] = summarized[name]["memory_note"]
    kill_criteria = []
    if summarized[ssfr_name]["vector_top_k_recall"] < 0.95:
        kill_criteria.append("Measured SSFR vector Recall@k was below 0.95.")
    if summarized[ssfr_name]["fraction_exact_fallback"] > 0.9:
        kill_criteria.append("More than 90% of CSV queries used exact centroid fallback.")
    if (
        summarized[ssfr_name]["latency_ms"]["mean"]
        >= summarized[exact_name]["latency_ms"]["mean"]
    ):
        kill_criteria.append(
            "SSFR end-to-end latency did not improve on exhaustive centroid routing."
        )
    report = {
        "benchmark": "csv_catalog",
        "seed": seed,
        "configuration": {
            "csv": str(csv_path),
            "queries": str(queries_path),
            "products": len(catalog.products),
            "shards": shards,
            "probe_values": list(probe_values),
            "bands": list(bands),
            "top_k": top_k,
            "physical_vectors_loaded": True,
            "local_index_backend": local_backend,
            "max_spectral_attempts": max_spectral_attempts,
        },
        "build": build_report,
        "methods": summarized,
        "by_probe_shards": probe_summary,
        "kill_criteria": kill_criteria,
    }
    _write_reports(report, output_directory)
    return report
