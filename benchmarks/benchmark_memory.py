"""Print router memory accounting for a reproducible synthetic setup."""

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
    matrix = normalize_rows(
        np.random.default_rng(42).normal(size=(args.shards, args.dimensions))
    )
    router = SSFRRouter().fit(matrix)
    print(json.dumps(router.memory_report(), indent=2))


if __name__ == "__main__":
    main()
