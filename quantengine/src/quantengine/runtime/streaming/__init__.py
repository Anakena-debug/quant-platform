"""quantengine.runtime.streaming — live-deployable async runtime (S35).

Public surface re-exported from this package's modules so callers can
write ``from quantengine.runtime.streaming import StreamingEngine``
without the inner module name.

Sibling of ``quantengine.runtime.daily_cycle`` (the batch path); both
orchestrate the same ``AbstractBroker`` / ``RiskGate`` substrate but
streaming consumes tick events through an async loop while daily_cycle
materialises one ``MarketSnapshot`` per day.

Per the S33 L1-deployment boundary (§5.D5): this package does NOT
import from ``quantcore.book``. Per the quantengine quantcore-
independence pattern (see ``quantengine.contracts.signal`` rationale),
this package does NOT import from ``quantcore`` at all — concrete
quantcore classes (``TradeEvent``, ``Bar``, ``BarBuilder`` subclasses,
``OnlineCUSUMFilter``, ``OnlineEWMAVolatility``) satisfy the
structural Protocols defined here at runtime via duck typing.
"""

from quantengine.runtime.streaming._demo import (
    DemoBroker,
    PriceLookup,
    SyntheticTradeFeed,
)
from quantengine.runtime.streaming.engine import (
    EngineConfig,
    EngineMetrics,
    ShutdownMode,
    StreamingEngine,
)
from quantengine.runtime.streaming.protocols import (
    AsyncBrokerProtocol,
    BarBuilderProtocol,
    BarLike,
    BrokerTimeoutError,
    Clock,
    CUSUMEvent,
    DataFeedProtocol,
    EventClock,
    OnlineCUSUMFilterProtocol,
    OnlineEWMAVolatilityProtocol,
    StreamContext,
    StreamingStrategy,
    SyncBrokerFacade,
    TradeEventLike,
    WallClock,
)
from quantengine.runtime.streaming.safe_broker import (
    JournalRecord,
    PriceProvider,
    SafeBroker,
    SafeBrokerJournalError,
    StateProvider,
)
from quantengine.runtime.streaming.wrappers import (
    ThreadSafeBrokerWrapper,
    WrapperTimeouts,
)

__all__ = [
    "AsyncBrokerProtocol",
    "BarBuilderProtocol",
    "BarLike",
    "BrokerTimeoutError",
    "CUSUMEvent",
    "Clock",
    "DataFeedProtocol",
    "DemoBroker",
    "EngineConfig",
    "EngineMetrics",
    "EventClock",
    "JournalRecord",
    "OnlineCUSUMFilterProtocol",
    "OnlineEWMAVolatilityProtocol",
    "PriceLookup",
    "PriceProvider",
    "SafeBroker",
    "SafeBrokerJournalError",
    "ShutdownMode",
    "StateProvider",
    "StreamContext",
    "StreamingEngine",
    "StreamingStrategy",
    "SyncBrokerFacade",
    "SyntheticTradeFeed",
    "ThreadSafeBrokerWrapper",
    "TradeEventLike",
    "WallClock",
    "WrapperTimeouts",
]
