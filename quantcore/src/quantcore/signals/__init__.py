"""
quantcore.signals
-----------------

Producer-side of the quantcore → quantengine handoff.

Writes a ``SignalArtifact`` (signal.{parquet|json} + manifest.json) whose
on-disk contract matches ``quantengine.data.signal.SignalArtifact.read()``
byte-for-byte. quantcore has **no** runtime dependency on quantengine —
the disk layout is the sole coupling surface.

See Also
--------
- quantengine.contracts.signal.AlphaSignal
- quantengine.data.signal.SignalArtifact  (reader side)
"""

from __future__ import annotations

from .producer import (
    SCHEMA_VERSION,
    Format,
    write_alpha_signal,
    _signal_to_dataframe,  # exported for tests
    _build_manifest,  # exported for tests
)

__all__ = [
    "SCHEMA_VERSION",
    "Format",
    "write_alpha_signal",
    "_signal_to_dataframe",
    "_build_manifest",
]
