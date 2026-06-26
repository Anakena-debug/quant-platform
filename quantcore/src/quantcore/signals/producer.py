"""
Producer for the quantcore → quantengine alpha-signal handoff.

Contract
========

Outputs
-------
An output directory ``out_dir`` containing:

* ``signal.parquet`` (default) or ``signal.json`` — one row per ticker::

      ticker            : str
      expected_return   : float64
      lower             : float64
      upper             : float64
      kelly_weight      : float64   (optional — present iff kelly_weights given)

  Parquet path uses ``pandas.DataFrame.to_parquet(..., index=False)``.
  JSON path uses stdlib ``json.dumps(records)`` with ``sort_keys=False`` to
  guarantee **bit-exact** float round-trip (mirrors quantengine's reader).

* ``manifest.json`` — single JSON object with **exactly** the core keys::

      run_id          : str
      model_sha       : str
      alpha           : float   (0 < alpha < 1)
      as_of           : str     (ISO-8601)
      n               : int     (len(tickers))
      format          : "parquet" | "json"
      schema_version  : int     (== SCHEMA_VERSION == 1)
      has_kelly       : bool

  Optional ``extra`` dict is merged in; ``extra`` may not override any core
  key (raises ValueError — matches quantengine's reader expectations).

Invariants validated on write
-----------------------------
1.  ``0 < alpha < 1``
2.  ``len(tickers) == n >= 1`` and tickers are unique, non-empty strings
3.  ``expected_return, lower, upper`` are 1-D float64 arrays of length ``n``
4.  ``lower[i] <= upper[i]`` for all i  (matches AlphaSignal invariant)
5.  All arrays finite (no NaN / inf) — refuse silent garbage downstream
6.  If ``kelly_weights`` is a ``pd.Series``, it is reindexed to ``tickers``
    order (raises KeyError on missing ticker); if ndarray, length must match

Rationale for producer living in quantcore
------------------------------------------
* **DAG direction.** quantengine consumes quantcore research outputs — the
  reverse dependency would make quantcore un-installable in a pure-research
  environment. Disk layout is a *versioned* contract (``schema_version``).
* **Testability.** Producer round-trip is validated against a stub reader
  that mirrors the quantengine schema, independent of the execution stack.

Limits
------
* Parquet engine is whichever pandas picks (``pyarrow`` preferred). The
  sandbox may lack pyarrow → use ``fmt="json"`` for local tests. Production
  deployments of both quantcore and quantengine require pyarrow.
* Bit-exact round-trip is guaranteed for ``fmt="json"`` (stdlib serialiser,
  IEEE-754 repr). Parquet round-trip is exact for float64 modulo engine
  compliance.

References
----------
* López de Prado, M. (2018). *Advances in Financial Machine Learning*.
  Wiley. ISBN 978-1119482086 — Ch. 7 (sample weights) and Ch. 9 (sizing).
* quantengine source: ``src/quantengine/contracts/signal.py``,
  ``src/quantengine/data/signal.py`` (as supplied by the team).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Contract constants — MUST match quantengine.data.signal
# -----------------------------------------------------------------------------
SCHEMA_VERSION: int = 1
Format = Literal["parquet", "json"]

_CORE_MANIFEST_KEYS: frozenset[str] = frozenset(
    {
        "run_id",
        "model_sha",
        "alpha",
        "as_of",
        "n",
        "format",
        "schema_version",
        "has_kelly",
    }
)


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _as_float64_1d(name: str, arr: np.ndarray | pd.Series | Sequence[float]) -> np.ndarray:
    """Coerce to contiguous 1-D float64 ndarray; refuse non-finite."""
    if isinstance(arr, pd.Series):
        arr = arr.to_numpy()
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    if a.ndim != 1:
        raise ValueError(f"{name}: expected 1-D, got shape {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: contains non-finite values (NaN / inf)")
    return a


def _validate_tickers(tickers: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(tickers, (list, tuple, pd.Index)):
        raise TypeError(f"tickers must be list/tuple/Index; got {type(tickers).__name__}")
    t = tuple(str(x) for x in tickers)
    if len(t) == 0:
        raise ValueError("tickers is empty")
    for x in t:
        if not x:
            raise ValueError("tickers contains empty string")
    if len(set(t)) != len(t):
        dup = [x for x in set(t) if t.count(x) > 1]
        raise ValueError(f"tickers contains duplicates: {sorted(dup)[:5]}")
    return t


def _align_kelly(
    kelly: pd.Series | np.ndarray | Sequence[float] | None,
    tickers: tuple[str, ...],
) -> np.ndarray | None:
    """Reindex kelly weights to ticker order (for pd.Series) or length-check."""
    if kelly is None:
        return None
    if isinstance(kelly, pd.Series):
        missing = [t for t in tickers if t not in kelly.index]
        if missing:
            raise KeyError(f"kelly_weights missing tickers: {missing[:5]}")
        aligned = kelly.reindex(list(tickers)).to_numpy(dtype=np.float64)
    else:
        aligned = np.asarray(kelly, dtype=np.float64)
        if aligned.shape != (len(tickers),):
            raise ValueError(f"kelly_weights shape {aligned.shape} != ({len(tickers)},)")
    if not np.isfinite(aligned).all():
        raise ValueError("kelly_weights contains non-finite values")
    return np.ascontiguousarray(aligned)


def _signal_to_dataframe(
    *,
    tickers: tuple[str, ...],
    expected_return: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    kelly_weights: np.ndarray | None,
) -> pd.DataFrame:
    """Reference projection used by both writer paths (parquet, json)."""
    cols: dict[str, np.ndarray | list[str]] = {
        "ticker": list(tickers),
        "expected_return": expected_return.astype(np.float64, copy=False),
        "lower": lower.astype(np.float64, copy=False),
        "upper": upper.astype(np.float64, copy=False),
    }
    if kelly_weights is not None:
        cols["kelly_weight"] = kelly_weights.astype(np.float64, copy=False)
    return pd.DataFrame(cols)


def _build_manifest(
    *,
    run_id: str,
    model_sha: str,
    alpha: float,
    as_of_iso: str,
    n: int,
    fmt: Format,
    has_kelly: bool,
    extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build manifest dict; reject ``extra`` overlapping core keys."""
    core: dict[str, Any] = {
        "run_id": run_id,
        "model_sha": model_sha,
        "alpha": float(alpha),
        "as_of": as_of_iso,
        "n": int(n),
        "format": fmt,
        "schema_version": SCHEMA_VERSION,
        "has_kelly": bool(has_kelly),
    }
    if extra:
        overlap = _CORE_MANIFEST_KEYS.intersection(extra.keys())
        if overlap:
            raise ValueError(f"extra cannot override core manifest keys: {sorted(overlap)}")
        core.update(extra)
    return core


def _as_of_to_iso(as_of: pd.Timestamp | str) -> str:
    if isinstance(as_of, str):
        # Round-trip through pd.Timestamp for validation + canonical ISO
        ts = pd.Timestamp(as_of)
    else:
        ts = pd.Timestamp(as_of)
    if ts is pd.NaT:
        raise ValueError("as_of is NaT")
    return ts.isoformat()


# -----------------------------------------------------------------------------
# Writers (mirror quantengine.data.signal.SignalArtifact.write behaviour)
# -----------------------------------------------------------------------------
def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)


def _write_json(df: pd.DataFrame, path: Path) -> None:
    # stdlib json on records — guarantees bit-exact float round-trip with reader
    records = df.to_dict(orient="records")
    # Ensure plain python floats / ints / strs for json — no numpy scalars
    clean: list[dict[str, Any]] = []
    for rec in records:
        clean.append({k: (v.item() if hasattr(v, "item") else v) for k, v in rec.items()})
    path.write_text(json.dumps(clean, sort_keys=False))


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def write_alpha_signal(
    *,
    tickers: Sequence[str],
    expected_return: np.ndarray | pd.Series | Sequence[float],
    lower: np.ndarray | pd.Series | Sequence[float],
    upper: np.ndarray | pd.Series | Sequence[float],
    alpha: float,
    kelly_weights: pd.Series | np.ndarray | Sequence[float] | None,
    as_of: pd.Timestamp | str,
    out_dir: str | Path,
    run_id: str,
    model_sha: str,
    fmt: Format = "parquet",
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """
    Emit a ``SignalArtifact`` bundle at ``out_dir``.

    Parameters
    ----------
    tickers
        Ordered, unique non-empty ticker symbols. Arrays are interpreted in
        this order. If ``kelly_weights`` is a ``pd.Series``, it is reindexed
        to this order (raises KeyError on missing ticker).
    expected_return, lower, upper
        1-D arrays of length ``len(tickers)``, finite, coerced to float64.
        Must satisfy ``lower <= upper`` elementwise.
    alpha
        Miscoverage level ∈ (0, 1). Identifies the nominal **(1 - α)**
        prediction-interval confidence (e.g. ``alpha=0.10`` ⇒ 90% PI).
    kelly_weights
        Optional position weights (typically fractional Kelly from
        ``portfolio.sizing.kelly_fraction``). ``pd.Series`` is reindexed by
        ticker; ``ndarray`` must match ``len(tickers)``.
    as_of
        Session timestamp (ISO-serialised into the manifest).
    out_dir
        Artifact root directory. Created if missing; overwritten if extant.
    run_id, model_sha
        Audit identifiers (hash-chain anchored in quantengine's journal).
    fmt
        "parquet" (default, production) or "json" (for tests / air-gapped).
    extra
        Optional extra manifest keys. **Must not** overlap core keys.

    Returns
    -------
    Path
        The artifact root (``out_dir``).

    Raises
    ------
    ValueError
        On any contract violation (shape, range, finiteness, α, lower<=upper,
        duplicate tickers, reserved manifest key, non-finite Kelly, …).
    KeyError
        If ``kelly_weights`` is a Series missing a ticker.
    TypeError
        On pathological argument types.

    Notes
    -----
    *   Contract is ``schema_version=1``; breaking changes bump this and
        require coordinated reader upgrade.
    *   Producer writes through a single reference ``_signal_to_dataframe``
        projection → parquet/json diverge only in serialiser choice.
    """
    # --- validation ---------------------------------------------------------
    t_tuple = _validate_tickers(tickers)
    n = len(t_tuple)

    er = _as_float64_1d("expected_return", expected_return)
    lo = _as_float64_1d("lower", lower)
    hi = _as_float64_1d("upper", upper)
    for name, arr in (("expected_return", er), ("lower", lo), ("upper", hi)):
        if arr.shape != (n,):
            raise ValueError(f"{name} shape {arr.shape} != ({n},)")
    if not np.all(lo <= hi):
        bad = int(np.sum(lo > hi))
        raise ValueError(f"lower > upper at {bad} positions (invariant violated)")

    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError(f"alpha must lie in (0, 1); got {a}")

    kw = _align_kelly(kelly_weights, t_tuple)

    if fmt not in ("parquet", "json"):
        raise ValueError(f"fmt must be 'parquet' or 'json'; got {fmt!r}")

    as_of_iso = _as_of_to_iso(as_of)

    run_id = str(run_id)
    model_sha = str(model_sha)
    if not run_id:
        raise ValueError("run_id is empty")
    if not model_sha:
        raise ValueError("model_sha is empty")

    # --- materialise --------------------------------------------------------
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = _signal_to_dataframe(
        tickers=t_tuple,
        expected_return=er,
        lower=lo,
        upper=hi,
        kelly_weights=kw,
    )

    manifest = _build_manifest(
        run_id=run_id,
        model_sha=model_sha,
        alpha=a,
        as_of_iso=as_of_iso,
        n=n,
        fmt=fmt,
        has_kelly=kw is not None,
        extra=extra,
    )

    if fmt == "parquet":
        _write_parquet(df, out / "signal.parquet")
    else:
        _write_json(df, out / "signal.json")

    (out / "manifest.json").write_text(json.dumps(manifest, sort_keys=False))

    return out
