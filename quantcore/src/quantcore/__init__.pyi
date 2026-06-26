"""Static type stub for quantcore's lazy top-level surface.

The package front door resolves its curated symbols through a PEP 562 ``__getattr__``
(see ``__init__.py``), which static checkers can only see as ``object``. This stub
re-declares that exact surface — the ``_EXPORTS`` symbols and ``_SUBPACKAGES`` — with
their real types, so ``from quantcore import kelly_fraction`` is precisely typed and
introspectable by checkers, IDEs, and agents while imports stay lazy at runtime.

Kept byte-for-byte in sync with the runtime ``_EXPORTS`` / ``_SUBPACKAGES`` by
``tests/test_init_stub_surface.py`` — add a symbol there and to the runtime dict in the
same change, or that test fails.
"""

# Subpackages reachable as lazy attributes (mirror of _SUBPACKAGES).
from . import (
    bars as bars,
    catalog as catalog,
    covariance as covariance,
    cv as cv,
    data as data,
    discoveries as discoveries,
    factors as factors,
    factory as factory,
    features as features,
    importance as importance,
    labels as labels,
    leakage as leakage,
    models as models,
    preprocessing as preprocessing,
    provenance as provenance,
    screening as screening,
    signals as signals,
    sizing as sizing,
    uncertainty as uncertainty,
    validation as validation,
    weights as weights,
)

# Curated top-level symbols (mirror of _EXPORTS), typed via their providing submodule.
from .bars import (
    aggregate_to_ohlcv as aggregate_to_ohlcv,
    dollar_bars as dollar_bars,
    tick_bars as tick_bars,
    volume_bars as volume_bars,
)
from .covariance import (
    denoise_covariance as denoise_covariance,
    detone_covariance as detone_covariance,
    ledoit_wolf_shrinkage as ledoit_wolf_shrinkage,
    marchenko_pastur_threshold as marchenko_pastur_threshold,
)
from .cv import (
    CombinatorialPurgedKFold as CombinatorialPurgedKFold,
    PurgedKFold as PurgedKFold,
    cv_score_purged as cv_score_purged,
    ml_get_train_times as ml_get_train_times,
)
from .data import (
    Bar as Bar,
    BarKind as BarKind,
    BookSnapshot as BookSnapshot,
    OrderEvent as OrderEvent,
    Side as Side,
    TradeEvent as TradeEvent,
)
from .importance import (
    feature_importance_mda as feature_importance_mda,
    feature_importance_mdi as feature_importance_mdi,
    feature_importance_sfi as feature_importance_sfi,
    importance_gate as importance_gate,
)
from .labels import (
    MetaLabeler as MetaLabeler,
    TripleBarrierConfig as TripleBarrierConfig,
    apply_triple_barrier as apply_triple_barrier,
    cusum_filter as cusum_filter,
    get_daily_vol as get_daily_vol,
    get_events as get_events,
)
from .models import (
    FrozenFlowRegimeArtifact as FrozenFlowRegimeArtifact,
    HorizonSpec as HorizonSpec,
    IncompatibleArtifactError as IncompatibleArtifactError,
)
from .sizing import (
    bet_size_sigmoid as bet_size_sigmoid,
    constrained_bet_size as constrained_bet_size,
    kelly_fraction as kelly_fraction,
)
from .validation.stats import (
    deflated_sharpe_ratio as deflated_sharpe_ratio,
    haircut_sharpe as haircut_sharpe,
    min_backtest_length as min_backtest_length,
    probabilistic_sharpe_ratio as probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting as probability_of_backtest_overfitting,
    sharpe_ratio as sharpe_ratio,
    sharpe_ratio_stats as sharpe_ratio_stats,
)
from .weights import (
    BootstrapConfig as BootstrapConfig,
    block_bootstrap as block_bootstrap,
    get_sample_uniqueness as get_sample_uniqueness,
    get_sample_weights as get_sample_weights,
    seq_bootstrap as seq_bootstrap,
)

__version__: str
__all__: list[str]
