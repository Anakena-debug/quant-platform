"""
Optional Numba support for AFML.

Numba provides ~10x speedup for numerical loops but is optional.
If not installed, functions fall back to pure Python (slower but functional).
"""

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    # Fallback: identity decorators
    def njit(*args, **kwargs):
        """No-op njit decorator when Numba not available."""

        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
