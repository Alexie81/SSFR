"""Small mathematical SSFR demonstration."""

from __future__ import annotations

import argparse

import numpy as np

from ssfr import SSFRConfig, SSFRRouter
from ssfr.console import configure_utf8_output
from ssfr.metrics import normalize_rows


def main() -> None:
    configure_utf8_output()
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=128)
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument("--probe-shards", type=int, default=8)
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    centroids = normalize_rows(rng.normal(size=(args.shards, args.dimensions)))
    query = normalize_rows(rng.normal(size=(1, args.dimensions)))[0]
    router = SSFRRouter(
        SSFRConfig(
            spectral_bands=(4, 8, 16, 32, 64),
            probe_shards=args.probe_shards,
            ordering_method="recursive_pca",
        )
    ).fit(centroids)
    route = router.route(query)
    print(route)
    print(router.stats())


if __name__ == "__main__":
    main()
