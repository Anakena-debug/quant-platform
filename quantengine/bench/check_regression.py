"""S36b AC5 — programmatic regression gate.

Runs the throughput benchmark (the same ``run_full_benchmark`` the
S36 PR4 CLI uses), writes the result JSON to
``bench/output/throughput-s36b-baseline.json``, then compares against
the S36 baseline JSON. Exits non-zero if any of four constraints is
violated (per D5):

    p99_us            ≤ baseline.p99_us * budget        (default 0.75)
    bridge_cost_us_p99 ≤ baseline.bridge_cost_us_p99 * 1.10
    queue_depth_p99   == 0
    backpressure_drops_total == 0

The constraints are independently regressable — the bridge cost is
mostly platform / scheduler grain (D5), so a soft 10% headroom is
allowed there; the pipeline budget is the actual S36b lever, with a
25% improvement default expressed via ``--budget 0.75``.

Output is structured: pass/fail per constraint + the absolute numbers
+ ratios. The seal-report operator pastes this into the report's
"Throughput Benchmark Output" section.

Usage:

    cd quantengine
    uv run python bench/check_regression.py \\
        --baseline bench/output/throughput-20260522-171611.json \\
        --budget 0.75
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Final

# AC5 invokes this script as `python bench/check_regression.py ...` (not
# `python -m bench.check_regression`), so the `bench` package isn't on
# sys.path by default. Insert the parent directory (quantengine/) so the
# `from bench.throughput import ...` line below resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Note: an earlier draft imported `cli` here to surface PR1's uvloop
# install side-effect during the bench. PR1 was reverted in PR5
# after the bench measured uvloop net-negative on macOS;
# the cli import is no longer needed. If a future sprint resurrects
# uvloop, re-add the cli import so the bench measures the runtime
# policy actually shipped to production.
from bench.throughput import ThroughputResult, run_full_benchmark  # noqa: E402

_DEFAULT_BUDGET: Final[float] = 0.75
_BRIDGE_HEADROOM: Final[float] = 1.10
_DEFAULT_OUTPUT_NAME: Final[str] = "throughput-s36b-baseline.json"


def _output_dir() -> Path:
    return Path(__file__).resolve().parent / "output"


def _format_check(label: str, ok: bool, actual: float, threshold: float, op: str) -> str:
    marker = "PASS" if ok else "FAIL"
    return f"  [{marker}] {label}: actual={actual:.3f} {op} threshold={threshold:.3f}"


def _check_constraints(
    new: ThroughputResult, baseline: dict[str, Any], budget: float
) -> tuple[bool, list[str]]:
    """Return (all_passed, lines_to_print)."""
    lines: list[str] = []
    all_passed = True

    # p99_us pipeline constraint
    baseline_p99 = float(baseline["p99_us"])
    threshold_p99 = baseline_p99 * budget
    p99_ok = new.p99_us <= threshold_p99
    lines.append(
        _format_check("p99_us ≤ baseline × budget", p99_ok, new.p99_us, threshold_p99, "≤")
    )
    all_passed = all_passed and p99_ok

    # bridge_cost_us_p99 regression bound
    baseline_bridge = float(baseline["bridge_cost_us_p99"])
    threshold_bridge = baseline_bridge * _BRIDGE_HEADROOM
    bridge_ok = new.bridge_cost_us_p99 <= threshold_bridge
    lines.append(
        _format_check(
            "bridge_cost_us_p99 ≤ baseline × 1.10",
            bridge_ok,
            new.bridge_cost_us_p99,
            threshold_bridge,
            "≤",
        )
    )
    all_passed = all_passed and bridge_ok

    # queue_depth_p99 must stay at 0 (SPSC queue must keep up)
    qd_ok = new.queue_depth_p99 == 0
    lines.append(
        _format_check("queue_depth_p99 == 0", qd_ok, float(new.queue_depth_p99), 0.0, "==")
    )
    all_passed = all_passed and qd_ok

    # backpressure_drops_total must be 0 at sustained rate
    drops_ok = new.backpressure_drops_total == 0
    lines.append(
        _format_check(
            "backpressure_drops_total == 0",
            drops_ok,
            float(new.backpressure_drops_total),
            0.0,
            "==",
        )
    )
    all_passed = all_passed and drops_ok

    return all_passed, lines


def _format_summary(new: ThroughputResult, baseline: dict[str, Any], budget: float) -> str:
    """One-block before/after summary suitable for the seal report."""
    baseline_p99 = float(baseline["p99_us"])
    baseline_bridge = float(baseline["bridge_cost_us_p99"])
    improvement = 100.0 * (1.0 - new.p99_us / baseline_p99) if baseline_p99 > 0 else 0.0
    return (
        "Before (S36 baseline) → After (S36b):\n"
        f"  p99_us:             {baseline_p99:>8.2f} → {new.p99_us:>8.2f}  "
        f"({improvement:+.1f}%; budget = ≥{(1.0 - budget) * 100:.0f}%)\n"
        f"  bridge_cost_us_p99: {baseline_bridge:>8.2f} → {new.bridge_cost_us_p99:>8.2f}\n"
        f"  queue_depth_p99:    {int(baseline['queue_depth_p99']):>8d} → {new.queue_depth_p99:>8d}\n"
        f"  backpressure_drops_total: {int(baseline['backpressure_drops_total'])} "
        f"→ {new.backpressure_drops_total}\n"
        f"  sustained_qps:      {float(baseline['sustained_qps']):>8.1f} → "
        f"{new.sustained_qps:>8.1f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S36b AC5 regression gate.")
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to the S36 baseline throughput JSON.",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=_DEFAULT_BUDGET,
        help=f"Improvement multiplier; new p99 must be ≤ baseline × budget. "
        f"Default {_DEFAULT_BUDGET} (25%% improvement).",
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=300_000,
        help="Bench event count (default 300_000).",
    )
    parser.add_argument(
        "--bridge-calls",
        type=int,
        default=10_000,
        help="Bridge microbench call count (default 10_000).",
    )
    parser.add_argument(
        "--target-qps",
        type=float,
        default=5000.0,
        help="Target input rate (default 5000.0).",
    )
    parser.add_argument(
        "--queue-maxsize",
        type=int,
        default=1000,
        help="Engine queue maxsize (default 1000).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=_DEFAULT_OUTPUT_NAME,
        help=f"Output JSON filename under bench/output/ (default {_DEFAULT_OUTPUT_NAME}).",
    )
    args = parser.parse_args(argv)

    if not args.baseline.exists():
        print(f"ERROR: baseline not found: {args.baseline}", file=sys.stderr)
        return 2

    print(
        f"Running post-S36b benchmark (n_events={args.n_events}, "
        f"target_qps={args.target_qps}) — this takes ~60s ...",
        file=sys.stderr,
    )
    result = run_full_benchmark(
        n_events=args.n_events,
        bridge_calls=args.bridge_calls,
        queue_maxsize=args.queue_maxsize,
        target_qps=args.target_qps,
    )

    out_dir = _output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name
    out_path.write_text(json.dumps(result.to_dict(), indent=2))
    print(f"Wrote post-S36b benchmark JSON to {out_path}", file=sys.stderr)

    with args.baseline.open("r") as f:
        baseline_data = json.load(f)

    all_passed, check_lines = _check_constraints(result, baseline_data, args.budget)

    print()
    print(_format_summary(result, baseline_data, args.budget))
    print()
    print("Constraint checks:")
    for line in check_lines:
        print(line)
    print()
    if all_passed:
        print("S36b regression gate: PASS")
        return 0
    print("S36b regression gate: FAIL — at least one constraint violated above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
