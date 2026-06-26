"""quantcore.importance — feature importance (AFML Ch. 8, MLAM Ch. 6).

Public entry points re-exported from `importance.py` so callers can write
`from quantcore.importance import feature_importance_mdi` without the
inner module name.

Open defect: MDI cardinality bias. OOB-corrected variant per
AFML §8.2 Snippet 8.2 is PENDING_TRIAGE.
"""

from quantcore.importance.importance import (
    feature_importance_mda,
    feature_importance_mdi,
    feature_importance_sfi,
    importance_gate,
)

__all__ = [
    "feature_importance_mda",
    "feature_importance_mdi",
    "feature_importance_sfi",
    "importance_gate",
]
