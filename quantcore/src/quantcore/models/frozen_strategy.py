"""Frozen flow-regime strategy artifact — disk contract for a deployable model.

Why this exists
---------------
``quantengine`` may never fit, train, or CV anything (ARCHITECTURE.md invariant
1) — it only loads a pre-fitted package and calls ``predict_proba``. The
validated ``flow_only`` TXN alpha, however, is only ever trained per-fold inside
the ``alpha_R`` walk-forward and discarded. This module is the serialization
boundary that lets ``alpha_R`` (research, writer) hand a single FROZEN model
package to ``quantengine`` (execution, reader) — without either importing the
other. It lives in ``quantcore`` because that is the only package both already
depend on (mirroring ``quantengine.data.signal.SignalArtifact``, whose
manifest-style layout this deliberately echoes).

What is frozen
--------------
The deployable is dual-horizon: one ``GradientBoostingClassifier`` per spread
regime (``tight`` = H100, ``moderate`` = H50), each with its calibrated entry
threshold and its per-class ``mu`` (mean realized return per label class, used to
turn ``predict_proba`` into an expected return ``er = Σ_c proba[c]·mu[c]``).

Directory layout
----------------
    <dir>/
      manifest.json        # always JSON (jq/grep-readable for ops triage)
      model_h100.joblib    # tight regime estimator
      model_h50.joblib     # moderate regime estimator

The manifest pins everything needed to score deterministically AND to refuse a
stale/incompatible package: feature order, per-horizon specs, label/bar config,
train-window provenance, and per-file ``sha256`` + library versions.

Loading discipline
------------------
``joblib`` and ``sklearn`` are imported lazily inside ``read`` so this module
(and the dataclasses) import cleanly in an environment without the sklearn extra
— only the actual model-load path needs it. ``read`` validates the package and
raises rather than returning a silently-wrong estimator.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
ARTIFACT_KIND = "frozen_flow_regime_strategy"
N_FEATURES = 15
RAW_FLOW_COLS: tuple[str, ...] = ("signed_vol_imb", "signed_dollar_imb", "signed_tick_imb")

# Aggressor-side sign convention the training features were built under.
# "dbn-aggressor" = the corrected s83-F11 convention ('B'/bid = buy aggressor
# = +1, 'A'/ask = sell aggressor = -1). Artifacts written before s83 carry no
# key and load as "pre-s83-inverted": still loadable (research comparison),
# but parity tests against post-s83 pipelines must treat them as stale.
SIDE_CONVENTION = "dbn-aggressor"
SIDE_CONVENTION_PRE_S83 = "pre-s83-inverted"


class IncompatibleArtifactError(RuntimeError):
    """Raised when a frozen artifact cannot be safely loaded.

    Covers schema/kind mismatch, feature-order or class drift, payload
    sha256 mismatch (corruption/tampering), and a sklearn-version mismatch
    that would make the pickled estimator unsafe to unpickle. The
    version check can be downgraded to a warning via
    ``read(..., allow_version_mismatch=True)`` for deliberate ops overrides.
    """


@dataclass(frozen=True)
class HorizonSpec:
    """One spread-regime horizon: its model file, gate bounds, and scoring scalars.

    ``mu`` maps label class (-1/0/1, as int-keyed) to the mean realized return
    of that class on the training window. ``classes`` is the estimator's
    ``classes_`` order, persisted so ``predict_proba`` columns map back to
    {-1,0,1} deterministically regardless of sklearn's internal ordering.
    """

    name: str  # "tight" | "moderate"
    horizon: int  # H (100 | 50)
    model_file: str  # relative filename within the artifact dir
    spread_lo_bps: float  # exclusive lower bound of the regime's rel-spread gate
    spread_hi_bps: float  # inclusive upper bound
    entry_th: float
    exit_th: float
    flip_th: float
    mu: dict[int, float]  # {-1: .., 0: .., 1: ..}
    classes: list[int]  # estimator.classes_ order

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        # JSON object keys are strings; store mu with str keys, restore on read.
        d["mu"] = {str(k): float(v) for k, v in self.mu.items()}
        d["classes"] = [int(c) for c in self.classes]
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> HorizonSpec:
        return cls(
            name=str(d["name"]),
            horizon=int(d["horizon"]),
            model_file=str(d["model_file"]),
            spread_lo_bps=float(d["spread_lo_bps"]),
            spread_hi_bps=float(d["spread_hi_bps"]),
            entry_th=float(d["entry_th"]),
            exit_th=float(d["exit_th"]),
            flip_th=float(d["flip_th"]),
            mu={int(k): float(v) for k, v in d["mu"].items()},
            classes=[int(c) for c in d["classes"]],
        )


@dataclass(frozen=True)
class FrozenFlowRegimeArtifact:
    """The full frozen-strategy manifest (everything except the model blobs).

    Construct it, then call :meth:`write` with the fitted estimators. The reader
    side gets it back from :meth:`read` together with the loaded estimators.
    """

    ticker: str
    feature_order: list[str]  # exact training column order (length N_FEATURES)
    raw_flow_cols: tuple[str, ...]
    horizons: list[HorizonSpec]
    regime_priority: list[str]  # e.g. ["tight", "moderate"]
    cost_bps: float
    label_config: dict[str, Any]
    bar_config: dict[str, Any]
    train_window: dict[str, Any]
    provenance: dict[str, Any]
    schema_version: int = SCHEMA_VERSION
    artifact_kind: str = ARTIFACT_KIND
    n_features: int = N_FEATURES
    side_convention: str = SIDE_CONVENTION

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def write(self, dir_path: Path | str, models: dict[str, Any]) -> Path:
        """Serialize the manifest + one joblib model blob per horizon.

        ``models`` maps each ``HorizonSpec.name`` to its fitted estimator.
        ``provenance['model_sha256']`` is (re)computed here per file from the
        bytes actually written, so the manifest's hashes always match the
        payloads on disk. Returns the artifact directory path.
        """
        import joblib  # lazy: writing needs joblib but importing this module must not

        out = Path(dir_path)
        out.mkdir(parents=True, exist_ok=True)

        names = {h.name for h in self.horizons}
        missing = names - set(models)
        if missing:
            raise ValueError(f"models missing for horizons: {sorted(missing)}")

        sha: dict[str, str] = {}
        for h in self.horizons:
            payload = out / h.model_file
            joblib.dump(models[h.name], payload)
            sha[h.model_file] = _sha256_file(payload)

        manifest = self._manifest_dict()
        # Stamp the recomputed payload hashes into provenance (authoritative).
        manifest["provenance"] = {**manifest["provenance"], "model_sha256": sha}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return out

    def _manifest_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_kind": self.artifact_kind,
            "ticker": self.ticker,
            "feature_order": list(self.feature_order),
            "n_features": self.n_features,
            "raw_flow_cols": list(self.raw_flow_cols),
            "horizons": [h.to_json() for h in self.horizons],
            "regime_priority": list(self.regime_priority),
            "cost_bps": float(self.cost_bps),
            "label_config": self.label_config,
            "bar_config": self.bar_config,
            "train_window": self.train_window,
            "provenance": self.provenance,
            "side_convention": self.side_convention,
        }

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    @classmethod
    def read(
        cls,
        dir_path: Path | str,
        *,
        allow_version_mismatch: bool = False,
    ) -> tuple[FrozenFlowRegimeArtifact, dict[str, Any]]:
        """Load + validate a frozen artifact; return (artifact, models).

        Validates (raising :class:`IncompatibleArtifactError` on any failure):
          - schema_version / artifact_kind,
          - feature_order length == n_features == N_FEATURES, raw_flow_cols exact,
          - per horizon: payload exists, sha256 matches provenance, estimator
            ``classes_`` matches the persisted ``classes``, ``n_features_in_``
            matches n_features,
          - sklearn version matches provenance (override → warning).
        """
        import warnings

        import joblib  # lazy

        d = Path(dir_path)
        manifest = cls.read_manifest(d)

        if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
            raise IncompatibleArtifactError(
                f"schema_version {manifest.get('schema_version')!r} != {SCHEMA_VERSION}"
            )
        if manifest.get("artifact_kind") != ARTIFACT_KIND:
            raise IncompatibleArtifactError(
                f"artifact_kind {manifest.get('artifact_kind')!r} != {ARTIFACT_KIND!r}"
            )

        feature_order = [str(c) for c in manifest["feature_order"]]
        n_features = int(manifest["n_features"])
        if len(feature_order) != n_features or n_features != N_FEATURES:
            raise IncompatibleArtifactError(
                f"feature_order length {len(feature_order)} / n_features {n_features} "
                f"!= expected {N_FEATURES}"
            )
        raw_flow_cols = tuple(str(c) for c in manifest["raw_flow_cols"])
        if raw_flow_cols != RAW_FLOW_COLS:
            raise IncompatibleArtifactError(f"raw_flow_cols {raw_flow_cols!r} != {RAW_FLOW_COLS!r}")

        provenance = manifest.get("provenance", {})
        expected_sha: dict[str, str] = provenance.get("model_sha256", {})
        art_sklearn = provenance.get("sklearn_version")
        try:
            import sklearn

            cur_sklearn = sklearn.__version__
        except ImportError:  # pragma: no cover
            cur_sklearn = None
        if art_sklearn is not None and cur_sklearn is not None and art_sklearn != cur_sklearn:
            msg = (
                f"artifact built with sklearn {art_sklearn}; current env has "
                f"{cur_sklearn}. Pickled estimators may not load correctly."
            )
            if allow_version_mismatch:
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
            else:
                raise IncompatibleArtifactError(msg)

        horizons = [HorizonSpec.from_json(h) for h in manifest["horizons"]]
        models: dict[str, Any] = {}
        for h in horizons:
            payload = d / h.model_file
            if not payload.exists():
                raise IncompatibleArtifactError(f"model payload missing: {payload}")
            want = expected_sha.get(h.model_file)
            got = _sha256_file(payload)
            if want is not None and got != want:
                raise IncompatibleArtifactError(
                    f"sha256 mismatch for {h.model_file}: manifest {want} != file {got}"
                )
            est = joblib.load(payload)
            est_classes = [int(c) for c in getattr(est, "classes_", [])]
            if est_classes != h.classes:
                raise IncompatibleArtifactError(
                    f"{h.name}: estimator.classes_ {est_classes} != manifest {h.classes}"
                )
            n_in = int(getattr(est, "n_features_in_", -1))
            if n_in != n_features:
                raise IncompatibleArtifactError(
                    f"{h.name}: estimator.n_features_in_ {n_in} != {n_features}"
                )
            models[h.name] = est

        artifact = cls(
            ticker=str(manifest["ticker"]),
            feature_order=feature_order,
            raw_flow_cols=raw_flow_cols,
            horizons=horizons,
            regime_priority=[str(r) for r in manifest["regime_priority"]],
            cost_bps=float(manifest["cost_bps"]),
            label_config=dict(manifest.get("label_config", {})),
            bar_config=dict(manifest.get("bar_config", {})),
            train_window=dict(manifest.get("train_window", {})),
            provenance=dict(provenance),
            side_convention=str(manifest.get("side_convention", SIDE_CONVENTION_PRE_S83)),
        )
        return artifact, models

    @staticmethod
    def read_manifest(dir_path: Path | str) -> dict[str, Any]:
        """Return the manifest dict (small, always JSON). Cheap; no sklearn."""
        p = Path(dir_path) / "manifest.json"
        if not p.exists():
            raise FileNotFoundError(f"manifest.json not found at {dir_path}")
        return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


__all__ = [
    "ARTIFACT_KIND",
    "N_FEATURES",
    "RAW_FLOW_COLS",
    "SCHEMA_VERSION",
    "FrozenFlowRegimeArtifact",
    "HorizonSpec",
    "IncompatibleArtifactError",
]
