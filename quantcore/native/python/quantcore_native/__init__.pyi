"""Type stub for the native quantcore kernels (PyO3 has no inline types)."""

class OnlineRollingFlow:
    def __init__(self) -> None: ...
    def reset(self) -> None: ...
    def update(self, value: float) -> dict[str, float]: ...

def bar_realized_moments_native(
    prices: list[float],
    close_indices: list[int],
    min_obs: int,
) -> tuple[
    list[float],  # rv
    list[float],  # rs_plus
    list[float],  # rs_minus
    list[float],  # sjv
    list[float],  # bv
    list[float],  # rj
    list[float],  # r_skew
    list[float],  # rq
    list[int],  # m_obs
]: ...
def deseasonalize_expanding_native(
    values: list[float],
    buckets: list[int],
    n_buckets: int,
) -> list[float]: ...  # multiplier

__all__: list[str]
