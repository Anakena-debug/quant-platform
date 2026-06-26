"""Audit journal: SHA-256 hash-chained integrity fingerprint over the ledger.

See :mod:`quantengine.audit.journal` for derivation and API.
"""

from quantengine.audit.journal import (
    GENESIS,
    GENESIS_TAG,
    ChainResult,
    canonical_event_bytes,
    chain_digest,
    iter_chain,
    verify_chain,
)

__all__ = [
    "GENESIS",
    "GENESIS_TAG",
    "ChainResult",
    "canonical_event_bytes",
    "chain_digest",
    "iter_chain",
    "verify_chain",
]
