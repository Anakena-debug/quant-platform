"""quantcore.leakage — automated point-in-time (lookahead) leakage detection.

A feature or label pipeline is *point-in-time clean* only if every output at time ``t`` is a
function of inputs at times ``<= t``. Lookahead bias — letting the future leak into the past —
is the single most common way a backtest lies, and the bias that repeatedly killed this
program's alpha studies. This module makes "no lookahead" a **checkable property** of any
transform rather than something humans eyeball.

Two complementary tests, both treating the transform as a black box ``data -> output`` over a
time-indexed Series/DataFrame (rows are time, ordered):

1. **truncation invariance** (:func:`truncation_test`) — the fundamental causality law:
   ``transform(data[:k])`` must equal ``transform(data)[:k]``. If appending future rows changes
   a past output, the transform looked ahead. Exact for causal transforms (rolling/expanding/
   cumulative), and it catches forward shifts and full-sample statistics alike.
2. **future perturbation** (:func:`perturbation_test`) — scramble every input row after a
   cutoff and assert the outputs at/before the cutoff are unchanged. Catches transforms that
   read future *values* (e.g. a centered window) without changing output length.

:func:`assert_no_lookahead` runs the checks and raises :class:`LeakageError` with the earliest
violation, so a pipeline's PIT-cleanliness can be pinned in a unit test.

    from quantcore.leakage import assert_no_lookahead
    assert_no_lookahead(lambda s: s.rolling(20, min_periods=1).mean(), prices)   # passes
    assert_no_lookahead(lambda s: (s - s.mean()) / s.std(), prices)              # raises LeakageError
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

Transform = Callable[[Any], Any]


class LeakageError(AssertionError):
    """Raised when a transform is shown to use future information (lookahead)."""

    def __init__(self, report: LeakageReport) -> None:
        super().__init__(report.detail)
        self.report = report


@dataclass(frozen=True, slots=True)
class LeakageReport:
    """The verdict of a lookahead scan over a transform."""

    method: str
    is_causal: bool
    n_checks: int  # number of cutoffs tested
    max_abs_diff: float  # largest past-output change observed (0.0 if clean)
    first_violation_cutoff: int | None
    first_violation_row: int | None
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _as_2d(obj: pd.Series | pd.DataFrame) -> np.ndarray:
    """View a transform output as a 2-D array (Series -> column vector), dtype preserved."""
    arr = obj.to_numpy()
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def _first_divergence(
    expected: pd.Series | pd.DataFrame,
    actual: pd.Series | pd.DataFrame,
    *,
    atol: float,
    rtol: float,
) -> tuple[int | None, float]:
    """First row where ``actual`` differs from ``expected``; plus a magnitude.

    Returns ``(None, 0.0)`` when they agree everywhere. Numeric output is compared with a NaN-aware
    tolerance (magnitude = max |diff|); categorical/object output (e.g. regime or sign labels) is
    compared by exact, missing-aware equality (magnitude = 1.0 to mark "labels differ"). A position
    where exactly one side is missing counts as a divergence — a structural lookahead.
    """
    a = _as_2d(expected)
    b = _as_2d(actual)
    if a.shape != b.shape:  # transform changed row/col count under truncation -> not causal
        return 0, float("inf")

    if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
        af = a.astype(float)
        bf = b.astype(float)
        both_nan = np.isnan(af) & np.isnan(bf)
        ok = np.isclose(af, bf, atol=atol, rtol=rtol, equal_nan=False) | both_nan
        if bool(ok.all()):
            return None, 0.0
        absdiff = np.abs(af - bf)
        absdiff[~np.isfinite(absdiff)] = 0.0  # NaN/inf diffs flagged by `ok`; contribute 0
        magnitude = float(absdiff.max())
    else:  # categorical / object labels: exact, missing-aware equality
        both_missing = pd.isna(a) & pd.isna(b)
        ok = (a == b) | both_missing
        if bool(ok.all()):
            return None, 0.0
        magnitude = 1.0

    bad_rows = np.where(~ok.all(axis=1))[0]
    return int(bad_rows[0]), magnitude


def _default_cutoffs(n: int) -> list[int]:
    """A spread of prefix lengths, biased toward the tail where full-sample leakage shows most."""
    candidates = {max(2, int(n * f)) for f in (0.25, 0.5, 0.75, 0.9)}
    candidates.add(n - 1)
    return sorted(c for c in candidates if 2 <= c < n)


def _resolve_cutoffs(data: pd.Series | pd.DataFrame, cutoffs: Sequence[int] | None) -> list[int]:
    n = len(data)
    if n < 4:
        raise ValueError(f"need at least 4 rows to scan for lookahead, got {n}")
    resolved = list(cutoffs) if cutoffs is not None else _default_cutoffs(n)
    resolved = sorted({c for c in resolved if 2 <= c < n})
    if not resolved:
        raise ValueError("no valid cutoffs in (2, len(data)) to test")
    return resolved


def truncation_test(
    transform: Transform,
    data: pd.Series | pd.DataFrame,
    *,
    cutoffs: Sequence[int] | None = None,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> LeakageReport:
    """Check ``transform(data[:k]) == transform(data)[:k]`` for several prefix lengths ``k``.

    This is the fundamental point-in-time law: a past output may not depend on whether future
    rows exist. Causal transforms satisfy it exactly; full-sample statistics and forward shifts
    do not.
    """
    ks = _resolve_cutoffs(data, cutoffs)
    full = transform(data)
    worst = 0.0
    for k in ks:
        prefix = transform(data.iloc[:k])
        row, diff = _first_divergence(full.iloc[:k], prefix, atol=atol, rtol=rtol)
        worst = max(worst, diff)
        if row is not None:
            return LeakageReport(
                method="truncation",
                is_causal=False,
                n_checks=len(ks),
                max_abs_diff=diff,
                first_violation_cutoff=k,
                first_violation_row=row,
                detail=(
                    f"lookahead: transform(data[:{k}]) disagrees with transform(data)[:{k}] "
                    f"at row {row} (Δ={diff:.3g}) — a past output changed when future rows were "
                    f"added, so it is not point-in-time."
                ),
            )
    return LeakageReport(
        method="truncation",
        is_causal=True,
        n_checks=len(ks),
        max_abs_diff=worst,
        first_violation_cutoff=None,
        first_violation_row=None,
        detail=f"no lookahead across {len(ks)} truncation cutoffs (max Δ={worst:.3g})",
    )


def perturbation_test(
    transform: Transform,
    data: pd.Series | pd.DataFrame,
    *,
    cutoffs: Sequence[int] | None = None,
    atol: float = 1e-8,
    rtol: float = 1e-5,
    seed: int = 0,
) -> LeakageReport:
    """Scramble every input row after a cutoff; assert outputs at/before the cutoff are unchanged.

    Complements :func:`truncation_test` by catching transforms that read future *values* while
    preserving output length/shape (e.g. a centered rolling window).
    """
    ks = _resolve_cutoffs(data, cutoffs)
    base = transform(data)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for k in ks:
        perturbed = data.copy()
        tail = perturbed.iloc[k:]
        # A gross, deterministic perturbation of the future — any leak into the head is obvious.
        noise = rng.standard_normal(tail.to_numpy().shape) * 1e3 + 1e4
        perturbed.iloc[k:] = noise if isinstance(perturbed, pd.DataFrame) else noise.ravel()
        after = transform(perturbed)
        row, diff = _first_divergence(base.iloc[:k], after.iloc[:k], atol=atol, rtol=rtol)
        worst = max(worst, diff)
        if row is not None:
            return LeakageReport(
                method="perturbation",
                is_causal=False,
                n_checks=len(ks),
                max_abs_diff=diff,
                first_violation_cutoff=k,
                first_violation_row=row,
                detail=(
                    f"lookahead: perturbing inputs after row {k} changed output at row {row} "
                    f"(Δ={diff:.3g}) — the transform reads future values."
                ),
            )
    return LeakageReport(
        method="perturbation",
        is_causal=True,
        n_checks=len(ks),
        max_abs_diff=worst,
        first_violation_cutoff=None,
        first_violation_row=None,
        detail=f"no lookahead across {len(ks)} perturbation cutoffs (max Δ={worst:.3g})",
    )


def assert_no_lookahead(
    transform: Transform,
    data: pd.Series | pd.DataFrame,
    *,
    methods: Sequence[str] = ("truncation", "perturbation"),
    cutoffs: Sequence[int] | None = None,
    atol: float = 1e-8,
    rtol: float = 1e-5,
    seed: int = 0,
) -> list[LeakageReport]:
    """Run the requested lookahead tests; raise :class:`LeakageError` on the first violation.

    Returns the (all-causal) reports when clean — usable as a one-line PIT guard in a test.
    """
    reports: list[LeakageReport] = []
    for method in methods:
        if method == "truncation":
            report = truncation_test(transform, data, cutoffs=cutoffs, atol=atol, rtol=rtol)
        elif method == "perturbation":
            report = perturbation_test(
                transform, data, cutoffs=cutoffs, atol=atol, rtol=rtol, seed=seed
            )
        else:
            raise ValueError(f"unknown leakage test method: {method!r}")
        if not report.is_causal:
            raise LeakageError(report)
        reports.append(report)
    return reports


def to_json(reports: Sequence[LeakageReport], *, indent: int | None = 2) -> str:
    """Serialize leakage reports to JSON (a list of report dicts)."""
    return json.dumps([r.to_dict() for r in reports], indent=indent)


__all__ = [
    "LeakageError",
    "LeakageReport",
    "assert_no_lookahead",
    "perturbation_test",
    "to_json",
    "truncation_test",
]
