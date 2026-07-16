"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .types import SSFRConfig


def load_config(path: str | Path) -> SSFRConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle) or {}
    if "spectral_bands" in payload:
        payload["spectral_bands"] = tuple(payload["spectral_bands"])
    return SSFRConfig(**payload)
