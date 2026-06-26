"""Smoke + structural tests for the S36 PR4 throughput benchmark.

What this test enforces (AC7):

  - The benchmark module is importable.
  - ``run_full_benchmark`` completes on a small N without error.
  - ``ThroughputResult`` exposes all four named D14 metrics
    (``p99_us``, ``sustained_qps``, ``queue_depth_p99``,
    ``bridge_cost_us_p99``) per amended D6.
  - All four metrics are finite + non-negative.
  - The bridge microbench p99 is positive (round-trip time can never
    be zero).
  - ``to_dict()`` serializes cleanly to JSON (CLI artifact path).

What this test does NOT enforce:

  - The D14 30µs p99 budget itself. That's a manual review item at
    seal time; CI hardware variability + small-N noise would make
    a budget gate flaky here. The seal report classifies the
    full-N (300k) result against D6's margin table.
  - The single-instrument deviation note. That's documented in the
    seal report (manual review item per amended D3).
"""

from __future__ import annotations

import json
import math

import pytest

from bench.throughput import ThroughputResult, run_full_benchmark


@pytest.fixture(scope="module")
def small_result() -> ThroughputResult:
    """Run a small benchmark once; share across structural tests."""
    return run_full_benchmark(
        n_events=500,
        bridge_calls=200,
        queue_maxsize=1000,
        target_qps=2000.0,  # smaller target keeps test runtime under ~1s
    )


class TestBenchmarkExecution:
    def test_returns_throughput_result(self, small_result: ThroughputResult) -> None:
        assert isinstance(small_result, ThroughputResult)

    def test_n_samples_matches_request(self, small_result: ThroughputResult) -> None:
        assert small_result.n_samples == 500

    def test_wall_time_positive(self, small_result: ThroughputResult) -> None:
        assert small_result.wall_time_s > 0
        assert math.isfinite(small_result.wall_time_s)


class TestAC6NamedMetrics:
    """Per amended D6 (commit 92a9a1f), four metrics are required.

    Test method names embed the metric name so a future regression
    that drops the field surfaces in the test report immediately.
    """

    def test_p99_us_present_and_finite(self, small_result: ThroughputResult) -> None:
        assert small_result.p99_us >= 0
        assert math.isfinite(small_result.p99_us)

    def test_sustained_qps_present_and_finite(self, small_result: ThroughputResult) -> None:
        assert small_result.sustained_qps >= 0
        assert math.isfinite(small_result.sustained_qps)

    def test_queue_depth_p99_present_and_nonnegative(self, small_result: ThroughputResult) -> None:
        assert small_result.queue_depth_p99 >= 0

    def test_bridge_cost_us_p99_present_finite_positive(
        self, small_result: ThroughputResult
    ) -> None:
        # The bridge microbench measures a thread-to-loop round trip;
        # the p99 must be strictly positive (cannot be zero).
        assert small_result.bridge_cost_us_p99 > 0
        assert math.isfinite(small_result.bridge_cost_us_p99)


class TestPercentileMonotonicity:
    def test_latency_percentiles_are_monotonic(self, small_result: ThroughputResult) -> None:
        # p50 ≤ p95 ≤ p99 ≤ p999 ≤ max — never violated by definition,
        # but a guard against accidental mis-indexing of the quantile
        # array in throughput.py.
        assert (
            small_result.p50_us
            <= small_result.p95_us
            <= small_result.p99_us
            <= small_result.p999_us
            <= small_result.max_us
        )

    def test_bridge_percentiles_are_monotonic(self, small_result: ThroughputResult) -> None:
        assert (
            small_result.bridge_cost_us_p50
            <= small_result.bridge_cost_us_p99
            <= small_result.bridge_cost_us_max
        )


class TestJsonSerializable:
    def test_to_dict_round_trips_through_json(self, small_result: ThroughputResult) -> None:
        payload = small_result.to_dict()
        text = json.dumps(payload)
        parsed = json.loads(text)
        # Spot-check a few keys; full structural equivalence comes
        # from the dataclass.
        assert parsed["n_samples"] == small_result.n_samples
        assert parsed["p99_us"] == small_result.p99_us
        assert parsed["bridge_cost_us_p99"] == small_result.bridge_cost_us_p99
        assert parsed["workload"] == "single_instrument_full_drain"
