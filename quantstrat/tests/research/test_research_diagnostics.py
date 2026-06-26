"""S28 PR4 — diagnostic artifact tests (AC4.1–AC4.6).

Validates ``quantstrat/tests/research/_research_diagnostics.py``. Runs under
plain ``uv run --directory quantstrat pytest`` — no ``--extra research``
flag because PR4 helpers must remain backend-agnostic (no mlflow import).

The matplotlib PNG path is gated behind ``pytest.skip`` when matplotlib
is not installed (AC4.5); production CI installs the ``reporting`` extra
to exercise it.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from research._research_diagnostics import (
    N_HISTOGRAM_BINS,
    er_vs_halfwidth_scatter,
    interval_half_width_summary,
    predictions_distribution_png,
    predictions_distribution_summary,
    seal_markdown_report,
    top_abs_forecast_table,
    tradeable_summary,
)

MODULE_FILE: Path = Path(__file__).resolve().parent / "_research_diagnostics.py"

# AC4.4 — per-artifact-file size cap.
PER_FILE_SIZE_CAP_BYTES: int = 32 * 1024


# ─── Deterministic toy fixture inputs (no random seed needed) ──────────

TICKERS: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
EXPECTED_RETURN: np.ndarray = np.array(
    [0.005, -0.003, 0.012, -0.001, 0.008, -0.020], dtype=np.float64
)
LOWER: np.ndarray = np.array([-0.015, -0.025, 0.001, -0.010, -0.008, -0.040], dtype=np.float64)
UPPER: np.ndarray = np.array([0.025, 0.019, 0.023, 0.008, 0.024, -0.005], dtype=np.float64)
# Tradeable mask (long OR short):
#   CCC: lower=0.001>0 → long
#   FFF: upper=-0.005<0 → short
#   all others: PI straddles zero → not tradeable


def _sample_run_inputs() -> dict[str, Any]:
    """Compact deterministic inputs for the markdown helper test."""
    return {
        "run_identity": {
            "sprint": "s28",
            "source_sprint": "s27",
            "branch": "sprint/s28-mlflow-experiment-tracking",
            "git_sha": "0" * 40,
            "experiment_family": "test",
            "run_family": "diagnostic_smoke",
        },
        "dataset_manifest": {
            "universe_name": "TOY",
            "data_source": "synthetic",
            "data_frequency": "daily",
            "row_count": 6,
            "missing_rate_summary": {"ticker": 0.0, "price": 0.0},
        },
        "metrics": {
            "n_tickers": 6.0,
            "tradeable_count": 2.0,
            "expected_return_mean": 0.00017,
        },
    }


# ─── AC4.1 — every helper is callable / importable ─────────────────────


def test_helpers_are_callable() -> None:
    """AC4.1 — every documented helper is importable and callable."""
    for fn in (
        predictions_distribution_summary,
        interval_half_width_summary,
        er_vs_halfwidth_scatter,
        tradeable_summary,
        top_abs_forecast_table,
        seal_markdown_report,
        predictions_distribution_png,
    ):
        assert callable(fn), f"{fn.__name__} not callable"


# ─── AC4.3 — well-formed output shapes ─────────────────────────────────


def test_predictions_distribution_summary_shape() -> None:
    """AC4.3 — bin_edges / counts / summary keys present and aligned."""
    summary = predictions_distribution_summary(EXPECTED_RETURN)
    assert set(summary.keys()) == {"bin_edges", "counts", "summary"}
    assert len(summary["bin_edges"]) == N_HISTOGRAM_BINS + 1
    assert len(summary["counts"]) == N_HISTOGRAM_BINS
    assert sum(summary["counts"]) == int(EXPECTED_RETURN.size)
    stats = summary["summary"]
    assert isinstance(stats, dict)
    expected_stat_keys = {"n", "min", "max", "mean", "std", "median", "q25", "q75"}
    assert expected_stat_keys <= set(stats.keys())


def test_interval_half_width_summary_shape() -> None:
    """AC4.3 — half-width summary carries quantiles and histogram."""
    summary = interval_half_width_summary(LOWER, UPPER)
    expected_keys = {
        "n",
        "min",
        "max",
        "mean",
        "median",
        "p10",
        "p25",
        "p75",
        "p90",
        "bin_edges",
        "counts",
    }
    assert expected_keys <= set(summary.keys())
    assert len(summary["bin_edges"]) == N_HISTOGRAM_BINS + 1
    assert len(summary["counts"]) == N_HISTOGRAM_BINS
    # All half-widths must be non-negative.
    assert summary["min"] >= 0.0


def test_er_vs_halfwidth_scatter_sorted_and_aligned() -> None:
    """AC4.3 — scatter rows are sorted by ticker and preserve per-row data.

    Shuffle the inputs to verify the sort actually fires and the
    expected_return / half_width values follow the ticker correctly.
    """
    perm = [2, 0, 5, 1, 4, 3]
    sh_tickers = tuple(TICKERS[i] for i in perm)
    sh_er = np.array([EXPECTED_RETURN[i] for i in perm], dtype=np.float64)
    sh_lo = np.array([LOWER[i] for i in perm], dtype=np.float64)
    sh_hi = np.array([UPPER[i] for i in perm], dtype=np.float64)

    rows = er_vs_halfwidth_scatter(sh_er, sh_lo, sh_hi, sh_tickers)
    assert len(rows) == len(TICKERS)
    tickers_in_rows = [row["ticker"] for row in rows]
    assert tickers_in_rows == sorted(TICKERS)
    for row in rows:
        assert set(row.keys()) == {"ticker", "expected_return", "half_width"}
    # Verify data alignment after the sort.
    aaa = next(r for r in rows if r["ticker"] == "AAA")
    assert aaa["expected_return"] == EXPECTED_RETURN[0]
    expected_aaa_hw = (UPPER[0] - LOWER[0]) / 2.0
    assert aaa["half_width"] == pytest.approx(expected_aaa_hw)


def test_tradeable_summary_counts_and_sorted_lists() -> None:
    """AC4.3 — long/short partitions match the long>0/short<0 contract."""
    summary = tradeable_summary(LOWER, UPPER, TICKERS)
    assert set(summary.keys()) == {
        "n_tickers",
        "tradeable_count",
        "long_count",
        "short_count",
        "tradeable_tickers",
        "long_tickers",
        "short_tickers",
    }
    assert summary["n_tickers"] == 6
    assert summary["long_tickers"] == ["CCC"]
    assert summary["short_tickers"] == ["FFF"]
    assert summary["long_count"] == 1
    assert summary["short_count"] == 1
    assert summary["tradeable_count"] == 2
    assert summary["tradeable_tickers"] == sorted(summary["tradeable_tickers"])
    assert summary["tradeable_tickers"] == ["CCC", "FFF"]


def test_top_abs_forecast_table_descending_by_abs() -> None:
    """AC4.3 — output is sorted by abs(expected_return) descending."""
    rows = top_abs_forecast_table(EXPECTED_RETURN, TICKERS, k=3)
    assert len(rows) == 3
    abs_returns = [row["abs_expected_return"] for row in rows]
    assert abs_returns == sorted(abs_returns, reverse=True)
    for row in rows:
        assert set(row.keys()) == {"ticker", "expected_return", "abs_expected_return"}
    # |er| = (0.005, 0.003, 0.012, 0.001, 0.008, 0.020) for AAA..FFF;
    # descending order = FFF, CCC, EEE.
    assert [r["ticker"] for r in rows] == ["FFF", "CCC", "EEE"]


def test_top_abs_tie_break_alphabetical() -> None:
    """AC4.3 — tied |expected_return| resolves by ticker ascending."""
    er = np.array([0.01, 0.01, -0.01], dtype=np.float64)
    tickers = ("ZZZ", "AAA", "BBB")
    rows = top_abs_forecast_table(er, tickers, k=2)
    assert [r["ticker"] for r in rows] == ["AAA", "BBB"]


@pytest.mark.parametrize("bad_k", [0, -1, -5, 7, 100])
def test_top_abs_invalid_k_raises(bad_k: int) -> None:
    """Fail loud on non-positive k or k > n."""
    with pytest.raises(ValueError):
        top_abs_forecast_table(EXPECTED_RETURN, TICKERS, k=bad_k)


def test_seal_markdown_report_contains_required_sections() -> None:
    """AC4.3 — markdown report carries the four documented sections."""
    inputs = _sample_run_inputs()
    top_abs = top_abs_forecast_table(EXPECTED_RETURN, TICKERS, k=3)
    md = seal_markdown_report(
        run_identity=inputs["run_identity"],
        dataset_manifest=inputs["dataset_manifest"],
        metrics=inputs["metrics"],
        top_abs=top_abs,
    )
    assert "# Research run report" in md
    assert "## Run identity" in md
    assert "## Dataset manifest" in md
    assert "## Metrics" in md
    assert "## Top |expected_return|" in md


def test_seal_markdown_report_carries_no_timestamp_or_hostname() -> None:
    """AC4.5 — report body contains no time-of-day or hostname markers.

    Defensive: a careless future contributor adding ``datetime.now()`` or
    ``socket.gethostname()`` to the report body would break byte-equal
    determinism. Catch the obvious tokens at the integration level.
    """
    inputs = _sample_run_inputs()
    top_abs = top_abs_forecast_table(EXPECTED_RETURN, TICKERS, k=3)
    md = seal_markdown_report(
        run_identity=inputs["run_identity"],
        dataset_manifest=inputs["dataset_manifest"],
        metrics=inputs["metrics"],
        top_abs=top_abs,
    )
    for token in ("UTC", "GMT", "hostname", "user@", "T00:", "T01:"):
        assert token not in md, f"AC4.5: forbidden token {token!r} found in markdown report."


# ─── AC4.2 — deterministic outputs (byte-equal across two calls) ───────


def test_predictions_distribution_summary_deterministic() -> None:
    s1 = predictions_distribution_summary(EXPECTED_RETURN)
    s2 = predictions_distribution_summary(EXPECTED_RETURN)
    assert json.dumps(s1, sort_keys=True) == json.dumps(s2, sort_keys=True)


def test_interval_half_width_summary_deterministic() -> None:
    s1 = interval_half_width_summary(LOWER, UPPER)
    s2 = interval_half_width_summary(LOWER, UPPER)
    assert json.dumps(s1, sort_keys=True) == json.dumps(s2, sort_keys=True)


def test_er_vs_halfwidth_scatter_deterministic() -> None:
    r1 = er_vs_halfwidth_scatter(EXPECTED_RETURN, LOWER, UPPER, TICKERS)
    r2 = er_vs_halfwidth_scatter(EXPECTED_RETURN, LOWER, UPPER, TICKERS)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


def test_tradeable_summary_deterministic() -> None:
    s1 = tradeable_summary(LOWER, UPPER, TICKERS)
    s2 = tradeable_summary(LOWER, UPPER, TICKERS)
    assert json.dumps(s1, sort_keys=True) == json.dumps(s2, sort_keys=True)


def test_top_abs_forecast_table_deterministic() -> None:
    r1 = top_abs_forecast_table(EXPECTED_RETURN, TICKERS, k=3)
    r2 = top_abs_forecast_table(EXPECTED_RETURN, TICKERS, k=3)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


def test_seal_markdown_report_deterministic() -> None:
    inputs = _sample_run_inputs()
    top_abs = top_abs_forecast_table(EXPECTED_RETURN, TICKERS, k=3)
    md1 = seal_markdown_report(
        run_identity=inputs["run_identity"],
        dataset_manifest=inputs["dataset_manifest"],
        metrics=inputs["metrics"],
        top_abs=top_abs,
    )
    md2 = seal_markdown_report(
        run_identity=inputs["run_identity"],
        dataset_manifest=inputs["dataset_manifest"],
        metrics=inputs["metrics"],
        top_abs=top_abs,
    )
    assert md1 == md2


# ─── AC4.3 + AC4.4 — files non-empty + < 32 KiB ────────────────────────


def test_all_artifacts_write_to_tmp_path_and_under_cap(tmp_path: Path) -> None:
    """AC4.3 + AC4.4 — every diagnostic round-trips to disk under cap."""
    inputs = _sample_run_inputs()
    top_abs = top_abs_forecast_table(EXPECTED_RETURN, TICKERS, k=3)
    artifacts: dict[str, str] = {
        "predictions_distribution.json": json.dumps(
            predictions_distribution_summary(EXPECTED_RETURN),
            sort_keys=True,
            indent=2,
        ),
        "interval_half_width.json": json.dumps(
            interval_half_width_summary(LOWER, UPPER),
            sort_keys=True,
            indent=2,
        ),
        "er_vs_halfwidth_scatter.json": json.dumps(
            er_vs_halfwidth_scatter(EXPECTED_RETURN, LOWER, UPPER, TICKERS),
            sort_keys=True,
            indent=2,
        ),
        "tradeable_summary.json": json.dumps(
            tradeable_summary(LOWER, UPPER, TICKERS),
            sort_keys=True,
            indent=2,
        ),
        "top_abs_forecast.json": json.dumps(
            top_abs,
            sort_keys=True,
            indent=2,
        ),
        "seal_report.md": seal_markdown_report(
            run_identity=inputs["run_identity"],
            dataset_manifest=inputs["dataset_manifest"],
            metrics=inputs["metrics"],
            top_abs=top_abs,
        ),
    }
    for name, content in artifacts.items():
        path = tmp_path / name
        path.write_text(content)
        assert path.exists(), f"{name} missing on disk"
        size = path.stat().st_size
        assert size > 0, f"{name} is empty"
        assert size < PER_FILE_SIZE_CAP_BYTES, (
            f"{name}: {size} bytes >= {PER_FILE_SIZE_CAP_BYTES} (32 KiB) cap"
        )


# ─── Misalignment / fail-loud paths ─────────────────────────────────────


def test_er_vs_halfwidth_scatter_misaligned_tickers_raises() -> None:
    with pytest.raises(ValueError, match="tickers"):
        er_vs_halfwidth_scatter(
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([2.0, 3.0], dtype=np.float64),
            tickers=("A", "B", "C"),
        )


def test_interval_half_width_summary_misaligned_arrays_raises() -> None:
    with pytest.raises(ValueError, match="misaligned"):
        interval_half_width_summary(
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([2.0, 3.0, 4.0], dtype=np.float64),
        )


def test_predictions_distribution_summary_empty_array_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        predictions_distribution_summary(np.array([], dtype=np.float64))


# ─── AC4.5 — matplotlib-optional PNG path ───────────────────────────────


def test_predictions_distribution_png_optional(tmp_path: Path) -> None:
    """AC4.5 — PNG path returns False (skip) when matplotlib unavailable."""
    out_path = tmp_path / "predictions_distribution.png"
    result = predictions_distribution_png(EXPECTED_RETURN, out_path)
    if not result:
        # matplotlib not installed: helper must NOT have written anything.
        assert not out_path.exists()
        pytest.skip("matplotlib not installed; PNG path returned False as documented")
    # matplotlib available: assert AC4.3 + AC4.4 on the PNG.
    assert out_path.exists()
    size = out_path.stat().st_size
    assert size > 0
    assert size < PER_FILE_SIZE_CAP_BYTES, (
        f"PNG size {size} bytes >= {PER_FILE_SIZE_CAP_BYTES} (32 KiB) cap."
    )


def test_predictions_distribution_png_byte_deterministic(tmp_path: Path) -> None:
    """AC4.5 — two PNGs from identical input are byte-equal within a process."""
    out_a = tmp_path / "dist_a.png"
    out_b = tmp_path / "dist_b.png"
    if not predictions_distribution_png(EXPECTED_RETURN, out_a):
        pytest.skip("matplotlib not installed")
    assert predictions_distribution_png(EXPECTED_RETURN, out_b)
    assert out_a.read_bytes() == out_b.read_bytes()


# ─── No-mlflow structural check (analog of AC2.7) ──────────────────────


def test_module_does_not_import_mlflow() -> None:
    """``_research_diagnostics.py`` must remain backend-agnostic — no mlflow."""
    tree = ast.parse(MODULE_FILE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("mlflow"), (
                    f"_research_diagnostics.py must not `import {alias.name}`."
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith("mlflow"), (
                f"_research_diagnostics.py must not `from {module} import ...`"
            )
