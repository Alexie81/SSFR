"""Cross-platform console output helpers."""

from __future__ import annotations

import sys


def configure_utf8_output() -> None:
    """Use UTF-8 for Romanian catalog text on legacy Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError, ValueError):
                pass
