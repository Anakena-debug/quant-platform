"""Data-plane adapters: DuckDB / parquet ⇄ quantengine in-memory contracts.

This package is the *only* place in ``quantengine`` that touches vendor
data storage. Every other module receives typed, validated in-memory
objects (``MarketSnapshot``, ``AlphaSignal``) built by these loaders.

Modules
-------
- ``snapshot``: PIT-safe loader ``DuckDB → MarketSnapshot``.
- ``signal``:   disk ⇄ ``AlphaSignal`` round-trip (parquet or JSON).
- ``universe``: session-date → ticker tuple.
"""

from quantengine.data.signal import SignalArtifact
from quantengine.data.snapshot import (
    DataFrameSnapshotLoader,
    DuckDBSnapshotLoader,
    SnapshotSource,
    pit_filter,
)
from quantengine.data.universe import (
    DataFrameUniverseResolver,
    DuckDBUniverseResolver,
    UniverseSource,
)

__all__ = [
    "DataFrameSnapshotLoader",
    "DataFrameUniverseResolver",
    "DuckDBSnapshotLoader",
    "DuckDBUniverseResolver",
    "SignalArtifact",
    "SnapshotSource",
    "UniverseSource",
    "pit_filter",
]
