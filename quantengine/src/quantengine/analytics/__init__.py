"""Post-run analytics over the ledger.

Pure read-only computations over a completed (or in-progress) ``Ledger``
and ``PortfolioState``. Does not mutate state or submit orders.
"""

from quantengine.analytics.shortfall import (
    ImplementationShortfall,
    compute_shortfall,
)

__all__ = [
    "ImplementationShortfall",
    "compute_shortfall",
]
