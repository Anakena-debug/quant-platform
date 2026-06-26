"""
P0.2 regression pin: legacy ``quantcore.validation.validation`` module stays deleted.

Context
-------
The broken legacy ``PurgedKFold`` (and its colocated ``WalkForwardCV``) lived in
``quantcore/validation/validation.py``. The legacy ``PurgedKFold`` applied a
one-sided purge rule ``keep = t1[train] < test_start`` which drops *every*
training sample whose label ends after the test-fold start — i.e. all
post-test training data is silently discarded. The correct AFML §7.3 rule is a
two-sided interval-overlap purge

    purge(train) = { i : not (t1_i < t0_test_min or t0_i > t1_test_max) }

plus an embargo. See the
``quantcore/validation/purged_kfold.py`` docstring for the full derivation.

The broken class had **zero callers** across ``quantcore/``, ``quantengine/``,
``quantdata/`` and the legacy ``quantcore/examples/`` trees at P0.2 execution
time (verified 2026-04-18). ``WalkForwardCV`` likewise had zero external
callers and was deleted alongside without a shim.

What this test pins
-------------------
1. ``quantcore.validation.validation`` module is no longer importable.
2. ``PurgedKFold`` and ``WalkForwardCV`` cannot be imported from that path.
3. The replacement ``quantcore.cv.purged_kfold.PurgedKFold`` is still present
   (post-S1.1b reorg; prior location was ``quantcore.validation.purged_kfold``).

If these tests fail, somebody silently resurrected the deleted module. Do not
relax the assertions; re-verify why the broken class is back and remove it.

Import convention
-----------------
Imports use the ``quantcore.*`` namespace (S1.1b namespaced form).
"""

from __future__ import annotations

import importlib
import importlib.util
import sys

import pytest


# -----------------------------------------------------------------------------
# Group 1 — module-level removal
# -----------------------------------------------------------------------------
def test_legacy_validation_module_not_findable() -> None:
    """``quantcore.validation.validation`` module spec must not resolve.

    Uses ``find_spec`` so the test works regardless of import caching.
    """
    # Evict any cached copy from a prior test run; otherwise a stale entry in
    # sys.modules could mask a regression.
    sys.modules.pop("quantcore.validation.validation", None)
    spec = importlib.util.find_spec("quantcore.validation.validation")
    assert spec is None, (
        "quantcore/validation/validation.py was resurrected. "
        "This module contained a broken PurgedKFold (drops post-test train "
        "samples). Use quantcore.cv.purged_kfold.PurgedKFold."
    )


def test_legacy_validation_import_raises() -> None:
    """``import quantcore.validation.validation`` must raise ``ModuleNotFoundError``."""
    sys.modules.pop("quantcore.validation.validation", None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("quantcore.validation.validation")


# -----------------------------------------------------------------------------
# Group 2 — symbol-level removal via the deleted path
# -----------------------------------------------------------------------------
def test_legacy_purged_kfold_symbol_import_raises() -> None:
    """``from quantcore.validation.validation import PurgedKFold`` must fail.

    Expects ``ImportError`` — ``ModuleNotFoundError`` is a subclass, so this
    catches both the "module gone" and hypothetical "module present but
    symbol removed" cases.
    """
    sys.modules.pop("quantcore.validation.validation", None)
    with pytest.raises(ImportError):
        from quantcore.validation.validation import PurgedKFold  # noqa: F401


def test_legacy_walk_forward_cv_symbol_import_raises() -> None:
    """``from quantcore.validation.validation import WalkForwardCV`` must fail.

    ``WalkForwardCV`` was colocated with the broken PurgedKFold and had zero
    external callers at deletion time; it was removed together rather than
    migrated to a new module (which would have been a design exercise, not
    cleanup).
    """
    sys.modules.pop("quantcore.validation.validation", None)
    with pytest.raises(ImportError):
        from quantcore.validation.validation import WalkForwardCV  # noqa: F401


# -----------------------------------------------------------------------------
# Group 3 — replacement class still importable (sanity)
# -----------------------------------------------------------------------------
def test_correct_purged_kfold_still_importable() -> None:
    """``quantcore.cv.purged_kfold.PurgedKFold`` must remain present.

    This is the canonical AFML §7.3 implementation with two-sided interval
    purging + embargo. If this import breaks while the tests in Group 1/2
    still pass, the deletion cleanup went too far.

    Path updated by S1.1b reorg (prior: ``quantcore.validation.purged_kfold``).
    """
    from quantcore.cv.purged_kfold import PurgedKFold

    assert PurgedKFold is not None
    # Smoke check: the class must still be a sklearn-compatible splitter.
    # We do not depend on a specific sklearn base class here to avoid
    # coupling this test to the replacement module's private base-class
    # choice; simple attribute presence is sufficient as a regression pin.
    assert hasattr(PurgedKFold, "split"), "replacement PurgedKFold missing .split()"
    assert hasattr(PurgedKFold, "get_n_splits"), "replacement PurgedKFold missing .get_n_splits()"
