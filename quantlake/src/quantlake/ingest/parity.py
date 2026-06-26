"""Cross-vendor parity (s81 REQ4) — convention check BEFORE any disagreement rate.

Comparing an ADJUSTED LSEG close to a RAW Databento close yields a meaningless "disagreement rate".
:func:`cross_vendor_parity` REFUSES to run unless the caller has verified the two sides share an
adjustment convention (``conventions_match=True``) or supplied an explicit documented transform. Then it
reports the disagreement rate + the worst offenders as a data-quality artifact (gate on review, not a
threshold).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class ConventionError(ValueError):
    """A parity comparison was attempted without a verified like-for-like adjustment convention."""


@dataclass(frozen=True)
class ParityResult:
    n_compared: int
    disagreement_rate: float  # fraction of joined rows with |left/right - 1| > tol
    tol: float
    worst: pd.DataFrame  # top offenders (for the data-quality artifact)


def cross_vendor_parity(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: list[str],
    left_col: str,
    right_col: str,
    conventions_match: bool,
    tol: float = 0.01,
    top: int = 10,
) -> ParityResult:
    """Disagreement rate between two vendors' price columns on the join keys ``on``.

    ``conventions_match`` MUST be True — the caller asserts (after verifying) that ``left_col`` and
    ``right_col`` are like-for-like (e.g. both RAW/unadjusted). Otherwise raises :class:`ConventionError`.
    """
    if not conventions_match:
        raise ConventionError(
            "refusing adjusted-vs-raw comparison: verify both closes share an adjustment convention "
            "(or supply a documented like-for-like transform) before computing a disagreement rate"
        )
    merged = left.merge(right, on=on, suffixes=("_l", "_r"))
    lc = merged[left_col] if left_col in merged.columns else merged[f"{left_col}_l"]
    rc = merged[right_col] if right_col in merged.columns else merged[f"{right_col}_r"]
    rel = (lc.astype(float) / rc.astype(float) - 1.0).abs()
    merged = merged.assign(_rel=rel)
    valid = merged[rel.notna() & (rc.astype(float) != 0.0)]
    rate = float((valid["_rel"] > tol).mean()) if len(valid) else float("nan")
    worst = valid.nlargest(top, columns="_rel")  # pyright: ignore[reportArgumentType, reportCallIssue]  # pandas stub df[mask]
    return ParityResult(n_compared=len(valid), disagreement_rate=rate, tol=tol, worst=worst)  # pyright: ignore[reportArgumentType]


__all__ = ["ConventionError", "ParityResult", "cross_vendor_parity"]
