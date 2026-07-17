"""Compatibility launcher for the main interactive SSFR CLI."""

from __future__ import annotations

import sys

from ssfr.cli import main as ssfr_main


def main() -> int:
    return ssfr_main(["interactive", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
