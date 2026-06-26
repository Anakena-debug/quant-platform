"""Concrete Strategy subclasses.

Each module here defines a ``Strategy`` (as declared by
``quantengine.strategies.base.Strategy``) that wraps a frozen ``quantcore`` package
(pipelines + primary + meta + conformal calibrator) and exposes
``predict(market) -> AlphaSignal``.
"""

from quantstrat.strategies.panel_weights import PanelWeightsStrategy

__all__ = ["PanelWeightsStrategy"]
