"""Public API for the SpectraShard Fourier Router."""

from .config import load_config
from .distributed_search import DistributedSSFRSearch
from .local_index import LocalShardIndex
from .router import SSFRRouter
from .types import (
    ImportReport,
    ProductRecord,
    RouteResult,
    SearchResult,
    ShardBuildResult,
    ShardMetadata,
    SSFRConfig,
)

__all__ = [
    "DistributedSSFRSearch",
    "ImportReport",
    "LocalShardIndex",
    "ProductRecord",
    "RouteResult",
    "SSFRConfig",
    "SSFRRouter",
    "SearchResult",
    "ShardBuildResult",
    "ShardMetadata",
    "load_config",
]

__version__ = "0.1.0"
