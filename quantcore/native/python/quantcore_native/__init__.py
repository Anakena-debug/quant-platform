"""Native acceleration kernels for quantcore (Rust/PyO3).

The compiled extension is the submodule ``_quantcore_native``; this package
re-exports its public names so ``from quantcore_native import OnlineRollingFlow``
works. Pure-Python references live in ``quantcore`` / ``alpha_research`` and
remain the fallback when this package is not installed.
"""

from ._quantcore_native import (
    OnlineRollingFlow,
    bar_realized_moments_native,
    deseasonalize_expanding_native,
)

__all__ = [
    "OnlineRollingFlow",
    "bar_realized_moments_native",
    "deseasonalize_expanding_native",
]
