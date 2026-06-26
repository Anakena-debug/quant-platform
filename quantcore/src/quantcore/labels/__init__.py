"""quantcore.labels — AFML Ch. 3 labeling primitives.

Public entry points re-exported so callers can write
``from quantcore.labels import MetaLabeler`` without the inner module
names. Includes:
  - Triple-barrier labeling (``labelling.py``): ``TripleBarrierConfig``,
    ``get_daily_vol``, ``cusum_filter``, ``get_events``,
    ``apply_triple_barrier``.
  - Meta-labeling (``meta.py``, AFML §3.5): ``MetaLabeler``,
    ``EconomicRationaleNotProvided``.
"""

from quantcore.labels.labelling import (
    TripleBarrierConfig,
    apply_triple_barrier,
    cusum_filter,
    get_daily_vol,
    get_events,
)
from quantcore.labels.meta import (
    EconomicRationaleNotProvided,
    MetaLabeler,
)

__all__ = [
    "EconomicRationaleNotProvided",
    "MetaLabeler",
    "TripleBarrierConfig",
    "apply_triple_barrier",
    "cusum_filter",
    "get_daily_vol",
    "get_events",
]
