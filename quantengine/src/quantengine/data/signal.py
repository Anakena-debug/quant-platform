"""Signal artifact — disk contract between quantcore and quantengine.

Why this exists
---------------
``quantcore`` produces ``AlphaSignal`` objects on a *research cadence*
(weekly training, daily inference). ``quantengine`` consumes them on an
*execution cadence* (once per session close). The two must be able to
run in separate processes, on separate machines, even at separate times.

This module is the serialization boundary: one function to write, one to
read, and a tiny manifest that pins provenance (model hash, run id, alpha).

Format choice
-------------
Two on-disk formats are supported, chosen at construction:

- ``parquet`` (default): small, typed, fast. Production choice.
  Requires ``pyarrow`` or ``fastparquet``.
- ``json``: dependency-free. Good for unit tests and environments where
  parquet engines aren't installed (e.g., minimal dev containers).

Both formats persist identical information. The manifest lives next to
the payload and is always JSON.

Directory layout
----------------
    <path>/
      signal.parquet   # or signal.json
      manifest.json

Manifest schema
---------------
    {
      "run_id":    <uuid string>,
      "model_sha": <git / model digest string>,
      "alpha":     <conformal miscoverage float in (0,1)>,
      "as_of":     <ISO timestamp or null>,
      "n":         <int, number of rows>,
      "format":    "parquet" | "json",
      "schema_version": 1
    }

Rationale
---------
Parquet is the correct production target — columnar, typed, composable
with downstream analytics. JSON is the fallback for test environments.
The manifest is always JSON because it is tiny and must be readable by
shell tools (``jq``, grep) during ops triage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from quantengine.contracts.signal import AlphaSignal, build_alpha_signal


SCHEMA_VERSION = 1
Format = Literal["parquet", "json"]


@dataclass(frozen=True)
class SignalArtifact:
    """On-disk handoff for a single ``AlphaSignal``.

    Usage (producer, ``quantcore`` side)::

        art = SignalArtifact(path=Path("signals/2026-04-17"), fmt="parquet")
        art.write(sig, run_id="r-2026-04-17-xgb-v3", model_sha="a1b2c3")

    Usage (consumer, ``quantengine`` side)::

        sig = SignalArtifact(path=Path("signals/2026-04-17")).read()

    The constructor does NOT auto-detect ``fmt``; pass it at write time.
    ``read`` honors the ``format`` field of the manifest, so consumers
    don't need to know which format the producer used.
    """

    path: Path
    fmt: Format = "parquet"

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def write(
        self,
        signal: AlphaSignal,
        *,
        run_id: str,
        model_sha: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Serialize ``signal`` to ``<path>/signal.<fmt>`` + ``<path>/manifest.json``.

        Overwrites any existing files at those paths. Directory is
        created if it doesn't exist.
        """
        self.path.mkdir(parents=True, exist_ok=True)
        df = _signal_to_dataframe(signal)
        if self.fmt == "parquet":
            df.to_parquet(self._payload_path("parquet"), index=False)
        elif self.fmt == "json":
            # Use stdlib json, not pandas.to_json: pandas truncates to
            # `double_precision=10` by default, which loses float64 bits and
            # breaks exact round-trip. Python's repr-based json is bit-exact
            # for finite float64.
            records = [
                {col: _scalar_to_json(v) for col, v in row.items()}
                for row in df.to_dict(orient="records")
            ]
            self._payload_path("json").write_text(json.dumps(records))
        else:  # pragma: no cover — Literal narrows this at the type level
            raise ValueError(f"Unsupported format: {self.fmt!r}")

        manifest = {
            "run_id": run_id,
            "model_sha": model_sha,
            "alpha": float(signal.alpha),
            "as_of": signal.timestamp,
            "n": int(signal.n),
            "format": self.fmt,
            "schema_version": SCHEMA_VERSION,
            "has_kelly": signal.kelly_weights is not None,
        }
        if extra:
            # Guard: never let `extra` clobber core fields.
            overlap = set(extra) & set(manifest)
            if overlap:
                raise ValueError(f"`extra` cannot override manifest keys: {sorted(overlap)}")
            manifest.update(extra)
        (self.path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def read(self) -> AlphaSignal:
        """Deserialize an ``AlphaSignal`` from disk.

        Format is read from the manifest (``format`` field), so callers
        need not know which format the producer chose.
        """
        manifest = self.read_manifest()
        fmt = manifest.get("format", "parquet")
        if fmt == "parquet":
            df = pd.read_parquet(self._payload_path("parquet"))
        elif fmt == "json":
            # Read the stdlib-json payload (list of records) and build the
            # dataframe ourselves so no value goes through pandas' lossy
            # double-precision JSON parser.
            records = json.loads(self._payload_path("json").read_text())
            df = pd.DataFrame.from_records(records)
        else:
            raise ValueError(f"Unsupported format in manifest: {fmt!r}")

        # Validate schema.
        expected = {"ticker", "expected_return", "lower", "upper"}
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(f"signal file missing columns: {sorted(missing)}")

        tickers = tuple(df["ticker"].astype(str).tolist())
        expected_return = df["expected_return"].to_numpy(dtype=np.float64)
        lower = df["lower"].to_numpy(dtype=np.float64)
        upper = df["upper"].to_numpy(dtype=np.float64)

        kelly = None
        if manifest.get("has_kelly", "kelly_weight" in df.columns) and "kelly_weight" in df.columns:
            kelly = df["kelly_weight"].to_numpy(dtype=np.float64).tolist()

        return build_alpha_signal(
            tickers=tickers,
            expected_return=expected_return.tolist(),
            lower=lower.tolist(),
            upper=upper.tolist(),
            alpha=float(manifest["alpha"]),
            kelly_weights=kelly,
            timestamp=manifest.get("as_of"),
            metadata={
                "run_id": manifest.get("run_id"),
                "model_sha": manifest.get("model_sha"),
            },
        )

    def read_manifest(self) -> dict[str, Any]:
        """Return the manifest dict (small, always JSON). Cheap to call."""
        p = self.path / "manifest.json"
        if not p.exists():
            raise FileNotFoundError(f"manifest.json not found at {self.path}")
        return json.loads(p.read_text())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _payload_path(self, fmt: Format) -> Path:
        return self.path / f"signal.{fmt}"


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------
def _scalar_to_json(v: Any) -> Any:
    """Coerce numpy/pandas scalars to native Python for stdlib json.

    For float64 we return the Python float, which ``json`` serializes with
    ``repr`` — bit-exact for finite values under IEEE-754 round-trip.
    """
    # numpy scalar → Python scalar
    if isinstance(v, np.generic):
        return v.item()
    return v


def _signal_to_dataframe(sig: AlphaSignal) -> pd.DataFrame:
    """Flatten an ``AlphaSignal`` to a 4- or 5-column dataframe.

    Column contract (stable):
        ticker, expected_return, lower, upper[, kelly_weight]
    """
    cols: dict[str, Any] = {
        "ticker": list(sig.tickers),
        "expected_return": sig.expected_return.astype(np.float64),
        "lower": sig.lower.astype(np.float64),
        "upper": sig.upper.astype(np.float64),
    }
    if sig.kelly_weights is not None:
        cols["kelly_weight"] = sig.kelly_weights.astype(np.float64)
    return pd.DataFrame(cols)


__all__ = ["SCHEMA_VERSION", "Format", "SignalArtifact"]
