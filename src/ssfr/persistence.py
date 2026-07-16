"""Persistence for fitted SSFR routers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .types import ShardMetadata, SSFRConfig

if TYPE_CHECKING:
    from .router import SSFRRouter


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _checksum(directory: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(path for path in directory.iterdir() if path.name != "manifest.json"):
        if file_path.is_file():
            digest.update(file_path.name.encode("utf-8"))
            digest.update(file_path.read_bytes())
    return digest.hexdigest()


def save_router(router: "SSFRRouter", path: str | Path) -> None:
    router._require_fitted()
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)

    config_payload = asdict(router.config)
    config_payload["spectral_bands"] = list(router.config.spectral_bands)
    (directory / "config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.save(directory / "order.npy", router.order, allow_pickle=False)
    np.save(directory / "inverse_order.npy", router.inverse_order, allow_pickle=False)
    np.save(directory / "exact_centroids.npy", router.centroids, allow_pickle=False)
    full_rfft = np.fft.rfft(router.ordered_centroids, axis=0)
    for band in router.bands:
        np.save(
            directory / f"spectral_payload_{band}.npy",
            full_rfft[: band + 1],
            allow_pickle=False,
        )
        np.save(directory / f"residuals_{band}.npy", router.residuals[band], allow_pickle=False)

    metadata_payload = []
    if router.shard_metadata is not None:
        metadata_payload = [asdict(item) for item in router.shard_metadata]
    (directory / "shard_metadata.json").write_text(
        json.dumps(metadata_payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )

    manifest = {
        "algorithm": "SSFR",
        "version": "0.2.0",
        "spectral_layout": "rfft_prefix_v1",
        "embedding_dimension": router.dimension,
        "shard_count": router.shard_count,
        "distance_metric": router.config.distance_metric,
        "bands": list(router.bands),
        "ordering_method": router.config.ordering_method,
        "created_at": datetime.now(UTC).isoformat(),
        "checksum": _checksum(directory),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_router(path: str | Path) -> "SSFRRouter":
    from .router import SSFRRouter

    directory = Path(path)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("algorithm") != "SSFR":
        raise ValueError("artifact is not an SSFR router")
    expected_checksum = manifest.get("checksum")
    if expected_checksum and _checksum(directory) != expected_checksum:
        raise ValueError("router artifact checksum mismatch")

    config_payload = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    config_payload["spectral_bands"] = tuple(config_payload["spectral_bands"])
    router = SSFRRouter(SSFRConfig(**config_payload))
    centroids = np.load(directory / "exact_centroids.npy", allow_pickle=False)
    order = np.load(directory / "order.npy", allow_pickle=False)
    metadata_payload = json.loads(
        (directory / "shard_metadata.json").read_text(encoding="utf-8")
    )
    shard_metadata = [
        ShardMetadata(
            shard_id=int(item["shard_id"]),
            item_count=int(item["item_count"]),
            centroid=np.asarray(item["centroid"], dtype=np.float64),
            euclidean_radius=float(item["euclidean_radius"]),
            angular_radius=(
                None if item.get("angular_radius") is None else float(item["angular_radius"])
            ),
            index_path=str(item.get("index_path", "")),
        )
        for item in metadata_payload
    ] or None
    if manifest.get("spectral_layout") != "rfft_prefix_v1":
        return router.fit(centroids, shard_metadata=shard_metadata, order=order)

    router.centroids = centroids
    router.order = order
    router.inverse_order = np.load(directory / "inverse_order.npy", allow_pickle=False)
    router.ordered_centroids = np.ascontiguousarray(router.centroids[router.order])
    router.bands = tuple(int(value) for value in manifest["bands"])
    router.frequency_map = {
        band: np.arange(band + 1, dtype=np.int64) for band in router.bands
    }
    router.spectral_payloads = {
        band: np.empty((0, router.centroids.shape[1]), dtype=np.complex128)
        for band in router.bands
    }
    full_band = router.centroids.shape[0] // 2
    partial_bands = [band for band in router.bands if band < full_band]
    attempted_partial_bands = (
        partial_bands
        if router.config.max_spectral_attempts is None
        else partial_bands[: router.config.max_spectral_attempts]
    )
    if attempted_partial_bands:
        largest_partial = max(attempted_partial_bands)
        largest_payload = np.load(
            directory / f"spectral_payload_{largest_partial}.npy", allow_pickle=False
        )
        router._rfft_payload = np.ascontiguousarray(largest_payload)
    else:
        router._rfft_payload = np.empty(
            (0, router.centroids.shape[1]), dtype=np.complex128
        )
    router.spectral_payloads = {
        band: (
            router._rfft_payload[: band + 1]
            if band in attempted_partial_bands
            else np.empty((0, router.centroids.shape[1]), dtype=np.complex128)
        )
        for band in router.bands
    }
    router.residuals = {
        band: np.load(directory / f"residuals_{band}.npy", allow_pickle=False)
        for band in router.bands
    }
    router.shard_metadata = shard_metadata
    router._full_spectrum = None
    router._fitted = True
    return router
