"""quantcore.cv — Purged cross-validation (AFML Ch. 7).

Public entry points re-exported from `purged_kfold.py` so callers can
write `from quantcore.cv import PurgedKFold` without the inner module
name.
"""

from quantcore.cv.purged_kfold import (
    CombinatorialPurgedKFold,
    PurgedKFold,
    cv_score_purged,
    ml_get_train_times,
)

__all__ = [
    "CombinatorialPurgedKFold",
    "PurgedKFold",
    "cv_score_purged",
    "ml_get_train_times",
]
