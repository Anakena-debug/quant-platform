"""Cross-sectional benchmark harnesses for S19+.

Currently exposes ``run_cov_benchmark`` only — a purged-k-fold covariance-
estimator comparison harness across {sample, LW, RMT} × {MV-as-GMV, NCO}.
The conformal branch axis is fully deferred from S19 per F-RP-007; see
``cov_benchmark`` module docstring for the deferral discipline.
"""

from quantstrat.benchmarks.cov_benchmark import run_cov_benchmark

__all__ = ["run_cov_benchmark"]
