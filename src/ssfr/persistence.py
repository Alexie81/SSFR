"""Persistence for fitted SSFR routers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .spectral import frequency_indices
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
    for band in router.bands:
        np.save(
            directory / f"spectral_payload_{band}.npy",
            router.spectral_payloads[band],
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
        "version": "0.1.0",
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
    router.centroids = np.load(directory / "exact_centroids.npy", allow_pickle=False)
    router.order = np.load(directory / "order.npy", allow_pickle=False)
    router.inverse_order = np.load(directory / "inverse_order.npy", allow_pickle=False)
    router.ordered_centroids = np.ascontiguousarray(router.centroids[router.order])
    router.bands = tuple(int(value) for value in manifest["bands"])
    router.frequency_map = {
        band: frequency_indices(router.centroids.shape[0], band) for band in router.bands
    }
    router.spectral_payloads = {
        band: np.load(directory / f"spectral_payload_{band}.npy", allow_pickle=False)
        for band in router.bands
    }
    router.residuals = {
        band: np.load(directory / f"residuals_{band}.npy", allow_pickle=False)
        for band in router.bands
    }
    metadata_payload = json.loads(
        (directory / "shard_metadata.json").read_text(encoding="utf-8")
    )
    router.shard_metadata = [
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
    router._full_spectrum = np.fft.fft(router.ordered_centroids, axis=0)
    router._fitted = True
    return router
