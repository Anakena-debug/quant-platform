"""Corporate-actions handler.

Applies NYSE / NASDAQ corporate actions to ``PortfolioState`` in the
same event-sourced pattern as fills: pure reducer, immutable state,
audit-logged to the ``Ledger``.

See :mod:`quantengine.corp_actions.handler`.
"""

from quantengine.corp_actions.handler import (
    CashDividend,
    CorpAction,
    CorpActionHandler,
    StockSplit,
)

__all__ = [
    "CashDividend",
    "CorpAction",
    "CorpActionHandler",
    "StockSplit",
]
