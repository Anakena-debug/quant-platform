"""quantcore core-primitive benchmark — the documented perf baseline + regression guard.

Times the load-bearing quantcore math primitives on representative synthetic data so
perf work is data-driven (the "profile before optimizing" rule) and regressions are
catchable. Dev tooling — lives outside ``src/`` and is not packaged.

    uv run python quantcore/bench/bench_core.py            # human table
    uv run python quantcore/bench/bench_core.py --json     # machine/agent JSON
    uv run python quantcore/bench/bench_core.py --repeat 7

Each row reports the MIN wall-clock over ``--repeat`` runs (min = least noisy estimate
of intrinsic cost). Sizes are modest so the suite finishes in a few seconds; bump them
to stress a specific primitive.
"""

from __future__ import annotations

import argparse
import json
import timeit
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from quantcore.covariance import denoise_covariance, detone_covariance, ledoit_wolf_shrinkage
from quantcore.features.features import frac_diff_ffd
from quantcore.features.microstructure import kyle_lambda_rolling
from quantcore.features.psy_gsadf import sadf  # PSY-correct SADF (not the deprecated shim)
from quantcore.labels.labelling import cusum_filter, get_daily_vol

_RNG = np.random.default_rng(7)


@dataclass(frozen=True, slots=True)
class BenchRow:
    category: str
    name: str
    ms: float


def _returns(n_samples: int = 750, n_features: int = 120) -> np.ndarray:
    return _RNG.standard_normal((n_samples, n_features))


def _price_series(n: int = 2000) -> pd.Series:
    idx = pd.bdate_range("2015-01-02", periods=n)
    steps = _RNG.normal(0.0, 0.01, n)
    return pd.Series(100.0 * np.exp(np.cumsum(steps)), index=idx)


def _build_cases() -> list[tuple[str, str, Callable[[], object]]]:
    """(category, name, thunk) — each thunk runs one primitive on fixed synthetic data."""
    ret = _returns()
    cov = np.cov(ret, rowvar=False)
    close = _price_series()
    # PSY sadf is ~O(n^2) ADF regressions; keep the series short. It takes an ndarray.
    short_log = np.log(close.to_numpy()[:300])
    vols = pd.Series(_RNG.integers(1, 100, len(close)).astype(float), index=close.index)
    return [
        ("covariance", "ledoit_wolf_shrinkage[750x120]", lambda: ledoit_wolf_shrinkage(ret)),
        ("covariance", "denoise_covariance[750x120]", lambda: denoise_covariance(ret)),
        ("covariance", "detone_covariance[120x120]", lambda: detone_covariance(cov)),
        ("labels", "get_daily_vol[2000]", lambda: get_daily_vol(close)),
        ("labels", "cusum_filter[2000]", lambda: cusum_filter(close, threshold=0.02)),
        ("features", "frac_diff_ffd[2000]", lambda: frac_diff_ffd(close, d=0.4)),
        (
            "features",
            "kyle_lambda_rolling[2000,w100]",
            lambda: kyle_lambda_rolling(close, vols, 100),
        ),
        ("features", "sadf[300]", lambda: sadf(short_log)),
    ]


def run(repeat: int = 5) -> list[BenchRow]:
    rows: list[BenchRow] = []
    for category, name, thunk in _build_cases():
        thunk()  # warm up (import/JIT/caches) so the timing reflects steady state
        best = min(timeit.repeat(thunk, number=1, repeat=repeat))
        rows.append(BenchRow(category, name, round(best * 1e3, 3)))
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench_core", description="quantcore core-primitive benchmark")
    p.add_argument("--repeat", type=int, default=5, help="timing runs per primitive (min reported)")
    p.add_argument("--json", action="store_true", help="emit JSON")
    args = p.parse_args(argv)

    rows = run(repeat=args.repeat)
    if args.json:
        print(json.dumps([asdict(r) for r in rows], indent=2))
        return 0
    print(f"{'category':<12} {'primitive':<34} {'ms (min)':>10}")
    print("-" * 58)
    for r in sorted(rows, key=lambda r: -r.ms):
        print(f"{r.category:<12} {r.name:<34} {r.ms:>10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
