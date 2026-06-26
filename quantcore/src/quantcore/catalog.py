"""quantcore.catalog — a discoverable registry of feature / factor building blocks.

The catalog is the discoverability backbone: a single, introspectable, JSON-
serializable registry of the feature-engineering and signal primitives quantcore
provides, with structured metadata (category, required inputs, references). It is
*decoupled* — every spec references its implementation by import path and resolves
lazily via :meth:`FactorSpec.resolve`, so importing the catalog pulls in nothing
heavy and no feature module needs to know the catalog exists::

    from quantcore.catalog import list_factors, search, get, to_json

    list_factors(category="microstructure")     # browse by category
    search("spread")                            # substring over name/summary/tags
    fn = get("amihud_illiquidity").resolve()    # import the real callable on demand
    to_json()                                   # the whole catalog as JSON (tooling/agents)

New primitives register via :func:`register`; alpha factors defined elsewhere
(e.g. research pipelines) can register into this same surface without modifying
quantcore.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class FactorSpec:
    """Metadata for one feature/factor primitive (implementation resolved lazily).

    Attributes
    ----------
    name      : unique registry key == the callable's name in ``module``.
    category  : grouping, e.g. ``"microstructure"`` / ``"entropy"``.
    module    : import path providing the callable, e.g. ``"quantcore.features.microstructure"``.
    summary   : one-line human description.
    inputs    : required data series/columns (descriptive, not enforced).
    tags      : free-form search tags.
    reference : literature citation, e.g. ``"AFML Ch.5"``.
    """

    name: str
    category: str
    module: str
    summary: str
    inputs: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    reference: str = ""

    def resolve(self) -> Callable[..., object]:
        """Import and return the underlying callable (lazy; no import cost until called)."""
        fn: Callable[..., object] = getattr(importlib.import_module(self.module), self.name)
        return fn

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_REGISTRY: dict[str, FactorSpec] = {}


def register(spec: FactorSpec, *, overwrite: bool = False) -> FactorSpec:
    """Add ``spec`` to the catalog. Raises on duplicate name unless ``overwrite``."""
    if spec.name in _REGISTRY and not overwrite:
        raise ValueError(
            f"factor {spec.name!r} already registered (pass overwrite=True to replace)"
        )
    _REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> FactorSpec:
    """Return the spec named ``name`` (raises KeyError with a hint if absent)."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"no factor {name!r} in catalog; see quantcore.catalog.list_factors()"
        ) from None


def list_factors(*, category: str | None = None, tag: str | None = None) -> list[FactorSpec]:
    """All specs (sorted by category, name), optionally filtered by category and/or tag."""
    out = sorted(_REGISTRY.values(), key=lambda s: (s.category, s.name))
    if category is not None:
        out = [s for s in out if s.category == category]
    if tag is not None:
        out = [s for s in out if tag in s.tags]
    return out


def search(query: str) -> list[FactorSpec]:
    """Case-insensitive substring search over name, summary, and tags."""
    q = query.lower()
    return [
        s
        for s in list_factors()
        if q in s.name.lower() or q in s.summary.lower() or any(q in t.lower() for t in s.tags)
    ]


def categories() -> list[str]:
    """Sorted list of distinct categories present in the catalog."""
    return sorted({s.category for s in _REGISTRY.values()})


def to_json(*, indent: int | None = 2) -> str:
    """Serialize the whole catalog to JSON (a list of spec dicts)."""
    return json.dumps([s.to_dict() for s in list_factors()], indent=indent)


_MICRO = "quantcore.features.microstructure"
_ENTROPY = "quantcore.features.entropy"
_BREAKS = "quantcore.features.structural_breaks"
_FFD = "quantcore.features.features"
_TOB = "quantcore.features.top_of_book"
_FACTORS = "quantcore.factors"

_BUILTINS: tuple[FactorSpec, ...] = (
    # --- microstructure ---
    FactorSpec(
        "tick_rule",
        "microstructure",
        _MICRO,
        "Lee-Ready tick-rule trade-sign classification",
        ("price",),
        ("signing", "flow"),
    ),
    FactorSpec(
        "bulk_volume_classification",
        "microstructure",
        _MICRO,
        "Bulk-volume classification of buy/sell volume (BVC)",
        ("close", "volume"),
        ("flow",),
    ),
    FactorSpec(
        "roll_spread",
        "microstructure",
        _MICRO,
        "Roll effective bid-ask spread from serial covariance",
        ("close",),
        ("spread", "liquidity"),
        "Roll 1984",
    ),
    FactorSpec(
        "corwin_schultz_spread",
        "microstructure",
        _MICRO,
        "Corwin-Schultz high-low bid-ask spread estimator",
        ("high", "low"),
        ("spread", "liquidity"),
        "Corwin-Schultz 2012",
    ),
    FactorSpec(
        "parkinson_volatility",
        "microstructure",
        _MICRO,
        "Parkinson high-low range volatility estimator",
        ("high", "low"),
        ("volatility",),
        "Parkinson 1980",
    ),
    FactorSpec(
        "kyle_lambda",
        "microstructure",
        _MICRO,
        "Kyle's lambda price-impact coefficient",
        ("price", "signed_volume"),
        ("impact", "liquidity"),
        "Kyle 1985",
    ),
    FactorSpec(
        "amihud_illiquidity",
        "microstructure",
        _MICRO,
        "Amihud illiquidity (|return| per dollar volume)",
        ("returns", "dollar_volume"),
        ("liquidity", "impact"),
        "Amihud 2002",
    ),
    FactorSpec(
        "vpin",
        "microstructure",
        _MICRO,
        "Volume-synchronized probability of informed trading (VPIN)",
        ("buy_volume", "sell_volume"),
        ("toxicity", "flow"),
        "Easley-Lopez de Prado-O'Hara 2012",
    ),
    FactorSpec(
        "microstructure_features",
        "microstructure",
        _MICRO,
        "Bundled microstructure feature set over OHLCV bars",
        ("ohlcv",),
        ("bundle",),
    ),
    # --- entropy ---
    FactorSpec(
        "shannon_entropy",
        "entropy",
        _ENTROPY,
        "Shannon entropy of an encoded symbol sequence",
        ("encoded",),
        ("entropy", "information"),
    ),
    FactorSpec(
        "normalized_entropy",
        "entropy",
        _ENTROPY,
        "Shannon entropy normalized to [0,1]",
        ("encoded",),
        ("entropy",),
    ),
    FactorSpec(
        "lempel_ziv_entropy",
        "entropy",
        _ENTROPY,
        "Lempel-Ziv complexity entropy-rate estimate",
        ("encoded",),
        ("entropy", "complexity"),
        "AFML Ch.18",
    ),
    FactorSpec(
        "kontoyiannis_entropy",
        "entropy",
        _ENTROPY,
        "Kontoyiannis entropy-rate estimator",
        ("encoded",),
        ("entropy",),
        "Kontoyiannis 1998",
    ),
    FactorSpec(
        "entropy_regime",
        "entropy",
        _ENTROPY,
        "Rolling-entropy regime indicator",
        ("returns",),
        ("entropy", "regime"),
    ),
    FactorSpec(
        "mutual_information",
        "entropy",
        _ENTROPY,
        "Mutual information between two series",
        ("x", "y"),
        ("information", "dependence"),
    ),
    FactorSpec(
        "entropy_features",
        "entropy",
        _ENTROPY,
        "Bundled entropy feature set",
        ("returns",),
        ("bundle",),
    ),
    # --- structural breaks ---
    FactorSpec(
        "adf_test",
        "structural_breaks",
        _BREAKS,
        "Augmented Dickey-Fuller unit-root test",
        ("series",),
        ("stationarity",),
    ),
    FactorSpec(
        "sadf",
        "structural_breaks",
        _BREAKS,
        "Supremum ADF explosiveness / bubble statistic",
        ("log_price",),
        ("bubble", "regime"),
        "Phillips-Shi-Yu 2015",
    ),
    FactorSpec(
        "gsadf",
        "structural_breaks",
        _BREAKS,
        "Generalized supremum ADF (rolling SADF)",
        ("log_price",),
        ("bubble", "regime"),
        "Phillips-Shi-Yu 2015",
    ),
    FactorSpec(
        "chow_test",
        "structural_breaks",
        _BREAKS,
        "Chow structural-break test at a known break",
        ("series",),
        ("break",),
        "Chow 1960",
    ),
    FactorSpec(
        "cusum_test",
        "structural_breaks",
        _BREAKS,
        "CUSUM test for structural breaks",
        ("series",),
        ("break",),
    ),
    FactorSpec(
        "structural_break_analysis",
        "structural_breaks",
        _BREAKS,
        "Bundled structural-break analysis (SADF/GSADF + date stamps)",
        ("log_price",),
        ("bundle",),
    ),
    # --- fractional differentiation ---
    FactorSpec(
        "frac_diff_ffd",
        "fractional_diff",
        _FFD,
        "Fixed-width-window fractional differentiation",
        ("series",),
        ("memory", "stationarity"),
        "AFML Ch.5",
    ),
    FactorSpec(
        "find_optimal_d",
        "fractional_diff",
        _FFD,
        "Minimum d achieving stationarity under FFD",
        ("series",),
        ("memory", "stationarity"),
        "AFML Ch.5",
    ),
    FactorSpec(
        "get_weights_ffd",
        "fractional_diff",
        _FFD,
        "FFD weight vector for a given differentiation order d",
        ("d",),
        ("memory",),
        "AFML Ch.5",
    ),
    # --- order flow / top-of-book ---
    FactorSpec(
        "top_of_book_features",
        "flow",
        _TOB,
        "Top-of-book (BBO) microstructure features",
        ("bbo",),
        ("microstructure", "flow"),
    ),
    FactorSpec(
        "signed_flow_features", "flow", _TOB, "Signed order-flow features", ("trades",), ("flow",)
    ),
    FactorSpec(
        "bar_flow_ratios", "flow", _TOB, "Per-bar order-flow ratio features", ("bar",), ("flow",)
    ),
    # --- cross-sectional factors (panel -> [dates x assets], ready for quantcore.factory) ---
    FactorSpec(
        "cross_sectional_momentum",
        "cross_sectional",
        _FACTORS,
        "Jegadeesh-Titman trailing-return momentum (skips the recent month)",
        ("close",),
        ("momentum", "trend"),
        "Jegadeesh-Titman 1993",
    ),
    FactorSpec(
        "cross_sectional_reversal",
        "cross_sectional",
        _FACTORS,
        "Short-term reversal (negative trailing return)",
        ("close",),
        ("reversal", "mean_reversion"),
    ),
    FactorSpec(
        "cross_sectional_volatility",
        "cross_sectional",
        _FACTORS,
        "Trailing realized volatility (low-volatility-anomaly characteristic)",
        ("close",),
        ("volatility", "risk"),
    ),
    FactorSpec(
        "cross_sectional_illiquidity",
        "cross_sectional",
        _FACTORS,
        "Amihud rolling illiquidity (|return| per dollar volume)",
        ("close", "volume"),
        ("liquidity", "impact"),
        "Amihud 2002",
    ),
)

for _spec in _BUILTINS:
    register(_spec)


__all__ = [
    "FactorSpec",
    "categories",
    "get",
    "list_factors",
    "register",
    "search",
    "to_json",
]
