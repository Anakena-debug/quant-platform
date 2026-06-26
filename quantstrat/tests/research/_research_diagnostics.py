"""S28 PR4 — deterministic diagnostic artifact generators (test-local).

Pure functions that derive small, JSON-friendly diagnostic artifacts (and
one optional PNG) from a research run's prediction arrays. These helpers
are agnostic to whichever tracking sink eventually ingests their output:
they do not import ``mlflow`` (per AC2.7's spirit; PR2 / PR4 stay
backend-agnostic so the schemas survive a future migration off MLflow).

The helpers accept only the producer-output slice
(``expected_return``, ``lower``, ``upper``, ``tickers``) and small
metadata dicts (``run_identity``, ``dataset_manifest``, ``metrics``). They
never accept the raw OHLCV panel, the feature matrix, or labels — that
boundary is what enforces AC4.4's "no raw data" contract by construction.

Determinism (AC4.2 / AC4.5):

* Histogram bin edges are derived from input min/max via
  ``np.linspace(min, max, N_HISTOGRAM_BINS + 1)``; ``n_bins`` is fixed.
* All sort orders are explicit and stable (lexicographic ticker tie-break).
* Markdown serialisation iterates ``sorted(dict)`` to lock key order;
  nested dict/list values are rendered via canonical-JSON.
* The optional PNG path uses fixed figsize + DPI and no ``bbox_inches=
  "tight"`` / no time-of-day annotation.

The leading-underscore filename signals "test helper, not pytest
collection target," following ``_toy_afml.py`` / ``_realistic_panel.py``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "N_HISTOGRAM_BINS",
    "predictions_distribution_summary",
    "interval_half_width_summary",
    "er_vs_halfwidth_scatter",
    "tradeable_summary",
    "top_abs_forecast_table",
    "seal_markdown_report",
    "predictions_distribution_png",
]

N_HISTOGRAM_BINS: int = 16


# ─── Internal helpers ──────────────────────────────────────────────────


def _check_1d_float(name: str, arr: np.ndarray) -> None:
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")


def _check_aligned(arrays: Mapping[str, np.ndarray]) -> int:
    """Verify every array is 1-D non-empty and shares the same length."""
    sizes = set()
    for name, arr in arrays.items():
        _check_1d_float(name, arr)
        sizes.add(int(arr.shape[0]))
    if len(sizes) > 1:
        raise ValueError(
            f"misaligned input arrays: {dict((k, int(v.shape[0])) for k, v in arrays.items())}"
        )
    return next(iter(sizes))


def _check_tickers_align(tickers: Sequence[str], n: int) -> None:
    if len(tickers) != n:
        raise ValueError(f"tickers length {len(tickers)} != aligned array length {n}.")


def _histogram_edges(values: np.ndarray) -> np.ndarray:
    """N_HISTOGRAM_BINS+1 edges spanning [min, max]; pads when degenerate."""
    lo = float(values.min())
    hi = float(values.max())
    if lo == hi:
        # Degenerate constant array: synthesize a unit-wide range so the
        # bins shape stays stable and np.histogram doesn't reject the
        # zero-width interval.
        lo -= 0.5
        hi += 0.5
    return np.linspace(lo, hi, N_HISTOGRAM_BINS + 1)


# ─── 1) predictions_distribution_summary ───────────────────────────────


def predictions_distribution_summary(
    expected_return: NDArray,
) -> dict[str, list[float] | list[int] | dict[str, float]]:
    """JSON-friendly histogram + summary stats of ``expected_return``.

    Bin edges are deterministic ``np.linspace(min, max, n_bins + 1)``
    (constant arrays use a synthetic unit-wide range so the shape stays
    stable). ``summary`` stats are computed from the original array.
    """
    arr = np.asarray(expected_return, dtype=np.float64)
    _check_1d_float("expected_return", arr)

    edges = _histogram_edges(arr)
    counts, _ = np.histogram(arr, bins=edges)
    return {
        "bin_edges": [float(x) for x in edges.tolist()],
        "counts": [int(c) for c in counts.tolist()],
        "summary": {
            "n": float(arr.size),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "median": float(np.median(arr)),
            "q25": float(np.quantile(arr, 0.25)),
            "q75": float(np.quantile(arr, 0.75)),
        },
    }


# ─── 2) interval_half_width_summary ────────────────────────────────────


def interval_half_width_summary(
    lower: NDArray,
    upper: NDArray,
) -> dict[str, float | list[float] | list[int]]:
    """Half-width = ``(upper - lower) / 2``: quantile summary + histogram."""
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    _check_aligned({"lower": lo, "upper": hi})

    half_width = (hi - lo) / 2.0
    edges = _histogram_edges(half_width)
    counts, _ = np.histogram(half_width, bins=edges)
    return {
        "n": float(half_width.size),
        "min": float(half_width.min()),
        "max": float(half_width.max()),
        "mean": float(half_width.mean()),
        "median": float(np.median(half_width)),
        "p10": float(np.quantile(half_width, 0.10)),
        "p25": float(np.quantile(half_width, 0.25)),
        "p75": float(np.quantile(half_width, 0.75)),
        "p90": float(np.quantile(half_width, 0.90)),
        "bin_edges": [float(x) for x in edges.tolist()],
        "counts": [int(c) for c in counts.tolist()],
    }


# ─── 3) er_vs_halfwidth_scatter ────────────────────────────────────────


def er_vs_halfwidth_scatter(
    expected_return: NDArray,
    lower: NDArray,
    upper: NDArray,
    tickers: Sequence[str],
) -> list[dict[str, float | str]]:
    """One row per ticker (``ticker``, ``expected_return``, ``half_width``).

    Output is sorted alphabetically by ticker for byte-determinism.
    """
    er = np.asarray(expected_return, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    n = _check_aligned({"expected_return": er, "lower": lo, "upper": hi})
    _check_tickers_align(tickers, n)

    half_width = (hi - lo) / 2.0
    rows: list[dict[str, float | str]] = [
        {
            "ticker": str(tickers[i]),
            "expected_return": float(er[i]),
            "half_width": float(half_width[i]),
        }
        for i in range(n)
    ]
    rows.sort(key=lambda row: str(row["ticker"]))
    return rows


# ─── 4) tradeable_summary ──────────────────────────────────────────────


def tradeable_summary(
    lower: NDArray,
    upper: NDArray,
    tickers: Sequence[str],
) -> dict[str, int | list[str]]:
    """Tradeable / long / short counts + sorted ticker partitions.

    * ``long`` ↔ ``lower > 0`` (prediction interval fully positive)
    * ``short`` ↔ ``upper < 0`` (prediction interval fully negative)
    * ``tradeable`` ↔ long OR short
    """
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    n = _check_aligned({"lower": lo, "upper": hi})
    _check_tickers_align(tickers, n)

    long_mask = lo > 0.0
    short_mask = hi < 0.0
    tradeable_mask = long_mask | short_mask

    long_tickers = sorted(str(tickers[i]) for i in range(n) if bool(long_mask[i]))
    short_tickers = sorted(str(tickers[i]) for i in range(n) if bool(short_mask[i]))
    tradeable_tickers = sorted(str(tickers[i]) for i in range(n) if bool(tradeable_mask[i]))
    return {
        "n_tickers": int(n),
        "tradeable_count": int(tradeable_mask.sum()),
        "long_count": int(long_mask.sum()),
        "short_count": int(short_mask.sum()),
        "tradeable_tickers": tradeable_tickers,
        "long_tickers": long_tickers,
        "short_tickers": short_tickers,
    }


# ─── 5) top_abs_forecast_table ─────────────────────────────────────────


def top_abs_forecast_table(
    expected_return: NDArray,
    tickers: Sequence[str],
    *,
    k: int,
) -> list[dict[str, float | str]]:
    """Top-k rows by ``|expected_return|``, tie-broken by ticker ascending.

    Raises ``ValueError`` for ``k <= 0`` or ``k > len(tickers)`` (fail
    loud; silent clipping would corrupt downstream report shape).
    """
    er = np.asarray(expected_return, dtype=np.float64)
    _check_1d_float("expected_return", er)
    n = int(er.size)
    _check_tickers_align(tickers, n)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")
    if k > n:
        raise ValueError(f"k ({k}) > number of tickers ({n}).")

    abs_er = np.abs(er)
    # Sort indices by (-abs_er, ticker) for descending |er| with
    # alphabetical tie-break.
    indexed = list(range(n))
    indexed.sort(key=lambda i: (-float(abs_er[i]), str(tickers[i])))
    top = indexed[:k]
    return [
        {
            "ticker": str(tickers[idx]),
            "expected_return": float(er[idx]),
            "abs_expected_return": float(abs_er[idx]),
        }
        for idx in top
    ]


# ─── 6) seal_markdown_report ───────────────────────────────────────────


def _render_value(value: Any) -> str:
    """Deterministic single-line value rendering for the markdown report."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        )
    return str(value)


def seal_markdown_report(
    *,
    run_identity: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    metrics: Mapping[str, float],
    top_abs: Sequence[Mapping[str, Any]],
) -> str:
    """Deterministic markdown body — no timestamp, no hostname, no user.

    Sections: Run identity, Dataset manifest, Metrics, Top |expected_return|.
    All dict keys are iterated in sorted order; nested values are rendered
    via canonical JSON.
    """
    lines: list[str] = ["# Research run report", ""]

    lines.append("## Run identity")
    for key in sorted(run_identity):
        lines.append(f"- **{key}**: {_render_value(run_identity[key])}")
    lines.append("")

    lines.append("## Dataset manifest")
    for key in sorted(dataset_manifest):
        lines.append(f"- **{key}**: {_render_value(dataset_manifest[key])}")
    lines.append("")

    lines.append("## Metrics")
    for key in sorted(metrics):
        lines.append(f"- **{key}**: {metrics[key]}")
    lines.append("")

    lines.append("## Top |expected_return|")
    if top_abs:
        lines.append("")
        lines.append("| ticker | expected_return | abs_expected_return |")
        lines.append("|:---|---:|---:|")
        for row in top_abs:
            lines.append(
                f"| {row['ticker']} | {row['expected_return']} | {row['abs_expected_return']} |"
            )
    lines.append("")
    return "\n".join(lines)


# ─── 7) predictions_distribution_png (optional, matplotlib-gated) ──────


def predictions_distribution_png(
    expected_return: NDArray,
    out_path: Path,
) -> bool:
    """Write a deterministic histogram PNG of ``expected_return``.

    Returns ``True`` on success, ``False`` if matplotlib is not installed.
    Other errors (e.g. disk write failure) propagate — only the
    matplotlib-availability path is silenced.

    Determinism (AC4.5): fixed figsize, fixed DPI, no ``bbox_inches=
    "tight"``, no time annotation. Within a single Python process two
    consecutive calls on the same input produce byte-equal PNGs.
    """
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    arr = np.asarray(expected_return, dtype=np.float64)
    _check_1d_float("expected_return", arr)

    er_min = float(arr.min())
    er_max = float(arr.max())
    if er_min == er_max:
        er_min -= 0.5
        er_max += 0.5

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=100)
    ax.hist(
        arr,
        bins=N_HISTOGRAM_BINS,
        range=(er_min, er_max),
        color="#444444",
        edgecolor="black",
    )
    ax.set_xlabel("expected_return")
    ax.set_ylabel("count")
    ax.set_title("Expected-return distribution")
    fig.savefig(out_path, format="png", dpi=100, metadata={"Software": ""})
    plt.close(fig)
    return True
