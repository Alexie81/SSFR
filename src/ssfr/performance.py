"""Runtime performance controls for native numerical libraries."""

from __future__ import annotations

from typing import Any


def limit_native_threads(threads: int | None) -> Any | None:
    """Limit BLAS/OpenMP pools and return a live limiter object.

    Small, latency-sensitive matrix-vector operations often become slower and more
    variable when a native library starts many worker threads. Keep the returned
    object alive for as long as the limit should remain active.
    """

    if threads is None or threads <= 0:
        return None
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return None
    return threadpool_limits(limits=int(threads))
