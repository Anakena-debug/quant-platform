"""quantcore.sizing — bet sizing primitives (AFML Ch. 10).

Public entry points re-exported from `sizing.py` so callers can write
`from quantcore.sizing import kelly_fraction` without the inner module name.
"""

from quantcore.sizing.sizing import (
    bet_size_sigmoid,
    constrained_bet_size,
    kelly_fraction,
)

__all__ = [
    "bet_size_sigmoid",
    "constrained_bet_size",
    "kelly_fraction",
]
