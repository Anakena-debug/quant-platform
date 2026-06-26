"""S27 PR1 — Realistic DJ30 panel loader (test-local helper).

Replaces the S26 synthetic high-SNR toy panel (``_toy_afml._synthetic_closes``)
with a fixed real historical DJ30 panel read from ``quantdata/quant.duckdb``
via the ``MarketData`` view.

The loader is *test infrastructure*, not production code. It lives under
``quantstrat/tests/research/`` and is consumed by S27 PR2/PR3 to drive the
realistic AFML/conformal chain. The leading-underscore filename mirrors
the S26 ``_toy_afml.py`` convention so pytest will not collect it.

Pinned (reproducibility
contract):

* universe — DJ30 from ``quantdata/dowjones30_tickers.txt`` (32 names)
* window  — 2022-01-03 .. 2024-12-31 (verbatim from S24)
* catalog — ``quantdata/quant.duckdb`` opened ``read_only=True``
* view    — ``MarketData`` (relative-path glob; resolved via
            ``contextlib.chdir(QUANTDATA_DIR)`` — same idiom S24 uses at
            ``quantstrat/tests/test_s24_e2e_smoke.py`` lines 86–98)

The returned long-form schema (``ticker``, ``session_date``, ``price``)
matches ``DataFrameSnapshotLoader``'s default column names so the panel
can be fed downstream without renaming.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# ─── Pinned constants (sprint §6) ───────────────────────────────────
QUANTDATA_DIR: Path = Path(__file__).parents[3] / "quantdata"
DJ30_FILE: Path = QUANTDATA_DIR / "dowjones30_tickers.txt"
DUCKDB_PATH: Path = QUANTDATA_DIR / "quant.duckdb"

START_DATE: str = "2022-01-03"
END_DATE: str = "2024-12-31"


def _load_dj30_tickers() -> list[str]:
    """Parse the DJ30 ticker file, skipping comments and blank lines."""
    return [
        line.strip()
        for line in DJ30_FILE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def load_dj30_panel() -> pd.DataFrame:
    """Return the fixed DJ30 long-form panel pinned by S27 §6.

    Returns
    -------
    pd.DataFrame
        Long-form panel with columns ``{ticker (str/object), session_date
        (datetime64[ns]), price (float64)}``, sorted by
        ``(ticker, session_date)``.

    Raises
    ------
    FileNotFoundError
        If the DJ30 ticker file or the DuckDB catalog is missing.
    ValueError
        If the underlying query returns rows that violate finite-data
        invariants (NaN/inf/non-positive price, duplicate
        ``(ticker, session_date)`` pairs, or non-monotonic per-ticker
        date sequences). The loader fails loud; it does not silently
        drop or impute.
    """
    if not DJ30_FILE.exists():
        raise FileNotFoundError(f"DJ30 ticker file missing: {DJ30_FILE}")
    if not DUCKDB_PATH.exists():
        raise FileNotFoundError(f"DuckDB catalog missing: {DUCKDB_PATH}")

    tickers = sorted(_load_dj30_tickers())

    with contextlib.chdir(QUANTDATA_DIR):
        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        try:
            placeholders = ",".join(["?"] * len(tickers))
            raw = con.execute(
                f"SELECT ticker, date, close FROM MarketData "
                f"WHERE ticker IN ({placeholders}) "
                f"AND date >= ? AND date <= ? "
                f"ORDER BY ticker, date",
                [*tickers, START_DATE, END_DATE],
            ).df()
        finally:
            con.close()

    panel = raw.rename(columns={"date": "session_date", "close": "price"})
    panel["ticker"] = panel["ticker"].astype("object")
    panel["session_date"] = pd.to_datetime(panel["session_date"]).astype("datetime64[ns]")
    panel["price"] = panel["price"].astype("float64")
    panel = panel[["ticker", "session_date", "price"]]
    panel = panel.sort_values(["ticker", "session_date"], kind="mergesort").reset_index(drop=True)

    _validate_panel(panel, expected_tickers=tickers)
    return panel


def pivot_to_wide(panel: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long-form panel into wide form for feature/label work.

    Index is ``session_date``, columns are ``ticker``, values are ``price``.
    """
    wide = panel.pivot(index="session_date", columns="ticker", values="price")
    wide.columns.name = "ticker"
    wide.index.name = "session_date"
    return wide


def _validate_panel(panel: pd.DataFrame, *, expected_tickers: list[str]) -> None:
    """Fail loud on any finite-data or PIT-ordering violation."""
    expected_cols = {"ticker", "session_date", "price"}
    if set(panel.columns) != expected_cols:
        raise ValueError(f"panel columns {set(panel.columns)} != expected {expected_cols}")

    if panel.empty:
        raise ValueError("realistic DJ30 panel returned zero rows")

    # AC1.5 — finite-data invariants.
    price = panel["price"].to_numpy()
    if not np.isfinite(price).all():
        n_nan = int(panel["price"].isna().sum())
        n_inf = int(np.isinf(price).sum())
        raise ValueError(f"non-finite prices in DJ30 panel: nan={n_nan}, inf={n_inf}")
    if (price <= 0).any():
        bad = panel.loc[panel["price"] <= 0, ["ticker", "session_date", "price"]]
        raise ValueError(f"non-positive prices in DJ30 panel: {len(bad)} rows\n{bad.head()}")

    dup_mask = panel.duplicated(subset=["ticker", "session_date"], keep=False)
    if dup_mask.any():
        dup = panel.loc[dup_mask, ["ticker", "session_date"]]
        raise ValueError(
            f"duplicate (ticker, session_date) pairs in DJ30 panel: {len(dup)} rows\n{dup.head()}"
        )

    # AC1.6 — per-ticker strict monotonic session_date.
    for ticker, group in panel.groupby("ticker", sort=False):
        diffs = group["session_date"].diff().dropna()
        if (diffs <= pd.Timedelta(0)).any():
            raise ValueError(f"non-monotonic session_date for ticker {ticker!r}")

    # AC1.3 — universe equals the pinned DJ30 set exactly.
    actual_tickers = sorted(panel["ticker"].unique().tolist())
    if actual_tickers != expected_tickers:
        missing = sorted(set(expected_tickers) - set(actual_tickers))
        extra = sorted(set(actual_tickers) - set(expected_tickers))
        raise ValueError(f"DJ30 universe mismatch: missing={missing}, extra={extra}")

    # AC1.4 — window bounds.
    start_ts = pd.Timestamp(START_DATE)
    end_ts = pd.Timestamp(END_DATE)
    if panel["session_date"].min() < start_ts:
        raise ValueError(
            f"session_date.min()={panel['session_date'].min()} < START_DATE={start_ts}"
        )
    if panel["session_date"].max() > end_ts:
        raise ValueError(f"session_date.max()={panel['session_date'].max()} > END_DATE={end_ts}")
