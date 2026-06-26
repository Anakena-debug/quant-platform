"""Throughput bench: native (Rust/PyO3) vs pure-Python OnlineRollingFlow.

Goes through the production entry point ``quantcore.features.online_rolling``.
Byte-parity is the HARD gate (the test suite owns it); the speedup printed here
is a SHOWCASE/soft metric — no threshold is enforced. Recorded in the s50 seal
report on the macOS production box.

Run:  uv run --all-extras python native/bench/bench_online_rolling.py
"""

from __future__ import annotations

import time

import numpy as np

from quantcore.features.online_rolling import (
    _NATIVE_AVAILABLE,
    _OnlineRollingFlowNative,
    _OnlineRollingFlowPy,
)

N = 1_000_000
REPS = 5
WARMUP = 1_000


def _bench(cls: type, data: list[float]) -> float:
    """Best-of-REPS wall-clock (s) for N sequential updates; lower is better."""
    warm = cls()
    for v in data[:WARMUP]:
        warm.update(v)
    best = float("inf")
    for _ in range(REPS):
        r = cls()
        t0 = time.perf_counter()
        for v in data:
            r.update(v)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    rng = np.random.default_rng(2026)
    data = rng.uniform(-1.0, 1.0, size=N).tolist()

    py_t = _bench(_OnlineRollingFlowPy, data)
    py_rate = N / py_t / 1e6
    print(f"pure-Python: {py_rate:7.3f} M upd/s  ({py_t:.3f}s for {N:,} updates)")

    if _NATIVE_AVAILABLE and _OnlineRollingFlowNative is not None:
        nat_t = _bench(_OnlineRollingFlowNative, data)
        nat_rate = N / nat_t / 1e6
        print(f"native:      {nat_rate:7.3f} M upd/s  ({nat_t:.3f}s for {N:,} updates)")
        print(f"speedup:     {py_t / nat_t:7.2f}x  (native vs pure-Python)")
    else:
        print("native:      NOT AVAILABLE — build with `maturin develop` (skipped)")


if __name__ == "__main__":
    main()
