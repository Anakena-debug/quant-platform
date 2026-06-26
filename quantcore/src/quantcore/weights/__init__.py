"""quantcore.weights — sample-weighting primitives (AFML Ch. 4).

Public entry points re-exported so callers can write
`from quantcore.weights import get_sample_weights` without the inner
module name. Includes:
  - AFML-4.10 sample weights + concurrency + uniqueness + sequential
    bootstrap (P0.3 corrected implementation in `bootstrap.py`).
  - Block / circular-block bootstrap + Patton-Politis-White 2009
    optimal block-length selector (P2.2, in `block_bootstrap.py`).
"""

from quantcore.weights.block_bootstrap import (
    block_bootstrap,
    politis_white_block_length,
)
from quantcore.weights.bootstrap import (
    BootstrapConfig,
    get_num_concurrent_events,
    get_sample_uniqueness,
    get_sample_weights,
    seq_bootstrap,
)

__all__ = [
    "BootstrapConfig",
    "block_bootstrap",
    "get_num_concurrent_events",
    "get_sample_uniqueness",
    "get_sample_weights",
    "politis_white_block_length",
    "seq_bootstrap",
]
