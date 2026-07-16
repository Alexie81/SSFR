"""Compare full rebuild and incremental DFT centroid update cost."""

from __future__ import annotations

import argparse
import json

import numpy as np

from ssfr import SSFRRouter
from ssfr.console import configure_utf8_output
from ssfr.metrics import normalize_rows


def main() -> None:
    configure_utf8_output()
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=1024)
    parser.add_argument("--dimensions", type=int, default=128)
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    matrix = normalize_rows(rng.normal(size=(args.shards, args.dimensions)))
    replacement = normalize_rows(rng.normal(size=(1, args.dimensions)))[0]
    incremental = SSFRRouter().fit(matrix.copy())
    full = SSFRRouter().fit(matrix.copy())
    report = {
        "incremental": incremental.update_centroid(0, replacement, incremental=True),
        "full_rebuild": full.update_centroid(0, replacement, incremental=False),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
