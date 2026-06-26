"""quantcore — AFML research library for quantitative finance.

Information-driven bars, triple-barrier labels, purged / combinatorial CV, sample
weights, covariance denoising (Ledoit-Wolf / RMT), feature importance, bet sizing
(Kelly), conformal prediction, and deflated-Sharpe / PBO validation.

Public API
----------
Subpackages: ``bars``, ``labels``, ``cv``, ``weights``, ``covariance``,
``importance``, ``sizing``, ``validation``, ``features``, ``preprocessing``,
``uncertainty``, ``models``, ``data``, ``signals``.

Headline symbols are re-exported at the top level — *lazily*, so ``import
quantcore`` stays cheap and heavy deps (scipy / numba / statsmodels) load only on
first use::

    from quantcore import kelly_fraction, sharpe_ratio, PurgedKFold, apply_triple_barrier

``quantcore.__all__`` enumerates the curated surface (introspectable by tooling and
agents); anything not listed is reachable via its subpackage, e.g.
``quantcore.bars.dollar_imbalance_bars``. Precise static types for the lazy surface
are provided by the companion ``__init__.pyi`` stub (kept in sync with ``_EXPORTS`` /
``_SUBPACKAGES`` by ``tests/test_init_stub_surface.py``).
"""

from __future__ import annotations

import importlib

__version__ = "0.1.0"

# Curated top-level surface: name -> providing submodule. Resolution is lazy (see
# __getattr__): nothing is imported until first access, so the package front door
# carries no import cost. Niche / colliding symbols (e.g. per-module
# SCHEMA_VERSION) stay on their subpackage by design.
_EXPORTS: dict[str, str] = {
    # sizing
    "kelly_fraction": "quantcore.sizing",
    "bet_size_sigmoid": "quantcore.sizing",
    "constrained_bet_size": "quantcore.sizing",
    # labels
    "apply_triple_barrier": "quantcore.labels",
    "TripleBarrierConfig": "quantcore.labels",
    "MetaLabeler": "quantcore.labels",
    "cusum_filter": "quantcore.labels",
    "get_daily_vol": "quantcore.labels",
    "get_events": "quantcore.labels",
    # cv
    "PurgedKFold": "quantcore.cv",
    "CombinatorialPurgedKFold": "quantcore.cv",
    "cv_score_purged": "quantcore.cv",
    "ml_get_train_times": "quantcore.cv",
    # weights
    "get_sample_weights": "quantcore.weights",
    "get_sample_uniqueness": "quantcore.weights",
    "seq_bootstrap": "quantcore.weights",
    "block_bootstrap": "quantcore.weights",
    "BootstrapConfig": "quantcore.weights",
    # covariance
    "ledoit_wolf_shrinkage": "quantcore.covariance",
    "denoise_covariance": "quantcore.covariance",
    "detone_covariance": "quantcore.covariance",
    "marchenko_pastur_threshold": "quantcore.covariance",
    # importance
    "feature_importance_mdi": "quantcore.importance",
    "feature_importance_mda": "quantcore.importance",
    "feature_importance_sfi": "quantcore.importance",
    "importance_gate": "quantcore.importance",
    # validation (empty subpackage __init__ -> resolve from the stats module)
    "sharpe_ratio": "quantcore.validation.stats",
    "sharpe_ratio_stats": "quantcore.validation.stats",
    "probabilistic_sharpe_ratio": "quantcore.validation.stats",
    "deflated_sharpe_ratio": "quantcore.validation.stats",
    "probability_of_backtest_overfitting": "quantcore.validation.stats",
    "min_backtest_length": "quantcore.validation.stats",
    "haircut_sharpe": "quantcore.validation.stats",
    # models
    "FrozenFlowRegimeArtifact": "quantcore.models",
    "HorizonSpec": "quantcore.models",
    "IncompatibleArtifactError": "quantcore.models",
    # data contracts
    "Bar": "quantcore.data",
    "BarKind": "quantcore.data",
    "Side": "quantcore.data",
    "TradeEvent": "quantcore.data",
    "BookSnapshot": "quantcore.data",
    "OrderEvent": "quantcore.data",
    # bars (headline constructors; the full builder set is on quantcore.bars)
    "dollar_bars": "quantcore.bars",
    "volume_bars": "quantcore.bars",
    "tick_bars": "quantcore.bars",
    "aggregate_to_ohlcv": "quantcore.bars",
}

# Subpackages reachable as lazy attributes (so `import quantcore; quantcore.bars`
# works and they appear in __all__ for discovery).
_SUBPACKAGES: tuple[str, ...] = (
    "bars",
    "labels",
    "cv",
    "weights",
    "covariance",
    "importance",
    "sizing",
    "validation",
    "features",
    "preprocessing",
    "uncertainty",
    "models",
    "data",
    "signals",
    "catalog",
    "screening",
    "factory",
    "factors",
    "discoveries",
    "provenance",
    "leakage",
)

__all__ = sorted([*_EXPORTS, *_SUBPACKAGES])


def __getattr__(name: str) -> object:
    """PEP 562 lazy attribute resolution for the curated top-level surface."""
    module = _EXPORTS.get(name)
    if module is not None:
        return getattr(importlib.import_module(module), name)
    if name in _SUBPACKAGES:
        return importlib.import_module(f"quantcore.{name}")
    raise AttributeError(f"module 'quantcore' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
