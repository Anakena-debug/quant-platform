"""Feature importance stability across walk-forward folds.

Computes MDI + PFI per fold, then reports:
- per-fold rankings
- fold-to-fold Spearman rank correlation
- top-k Jaccard overlap
- selection frequency

Usage:

    uv run python scripts/diagnose_feature_importance.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

from run_amzn_combined import run_walk_forward

OUTPUT_DIR = Path("diagnostics/feature_importance")
TOP_K = 10
COST_BPS = 5.0


def _mdi_importance(model: object, feature_names: list[str]) -> pd.Series:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return pd.Series(dtype=float)
    return pd.Series(imp, index=feature_names, name="mdi").sort_values(ascending=False)


def _pfi_importance(model: object, X: pd.DataFrame, y: pd.Series) -> pd.Series:
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=10,
        random_state=42,
        scoring="balanced_accuracy",
    )
    return pd.Series(
        result.importances_mean,
        index=X.columns,
        name="pfi",
    ).sort_values(ascending=False)


def _topk_set(importance: pd.Series, k: int) -> set[str]:
    return set(importance.sort_values(ascending=False).head(k).index)


def compute_stability(
    fold_importances: dict[int, pd.Series],
    top_k: int = TOP_K,
) -> dict[str, pd.DataFrame]:
    fold_ids = sorted(fold_importances)
    all_features = sorted(set().union(*[s.index for s in fold_importances.values()]))

    ranked = {
        fid: fold_importances[fid]
        .reindex(all_features)
        .fillna(0.0)
        .rank(ascending=False, method="average")
        for fid in fold_ids
    }
    top_sets = {fid: _topk_set(fold_importances[fid], top_k) for fid in fold_ids}

    spearman_mat = pd.DataFrame(index=fold_ids, columns=fold_ids, dtype=float)
    jaccard_mat = pd.DataFrame(index=fold_ids, columns=fold_ids, dtype=float)

    for i in fold_ids:
        for j in fold_ids:
            rho, _ = spearmanr(ranked[i].values, ranked[j].values, nan_policy="omit")
            spearman_mat.loc[i, j] = rho
            union = top_sets[i] | top_sets[j]
            inter = top_sets[i] & top_sets[j]
            jaccard_mat.loc[i, j] = len(inter) / len(union) if union else np.nan

    rows = []
    for feature in all_features:
        scores = [
            fold_importances[fid].reindex(all_features).fillna(0.0).loc[feature] for fid in fold_ids
        ]
        ranks = [ranked[fid].loc[feature] for fid in fold_ids]
        selected = [feature in top_sets[fid] for fid in fold_ids]
        rows.append(
            {
                "feature": feature,
                "mean_importance": float(np.mean(scores)),
                "std_importance": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "mean_rank": float(np.mean(ranks)),
                "std_rank": float(np.std(ranks, ddof=1)) if len(ranks) > 1 else 0.0,
                f"top_{top_k}_freq": float(np.mean(selected)),
            }
        )

    summary = (
        pd.DataFrame(rows)
        .sort_values(
            [f"top_{top_k}_freq", "mean_rank", "mean_importance"], ascending=[False, True, False]
        )
        .reset_index(drop=True)
    )

    return {"spearman": spearman_mat, "jaccard": jaccard_mat, "summary": summary}


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Feature importance stability — AMZN combined EV-gated\n", flush=True)

    t0 = time.monotonic()
    folds = run_walk_forward(cost_bps=COST_BPS)
    if not folds:
        print("No folds produced.", file=sys.stderr)
        return 1

    print(f"\n{len(folds)} folds. Computing MDI + PFI per fold...\n", flush=True)

    mdi_per_fold: dict[int, pd.Series] = {}
    pfi_per_fold: dict[int, pd.Series] = {}

    for f in folds:
        print(
            f"  Fold {f.fold_id}: {f.X_test.shape[0]} test samples, {f.X_test.shape[1]} features",
            flush=True,
        )

        mdi = _mdi_importance(f.model, list(f.X_test.columns))
        pfi = _pfi_importance(f.model, f.X_test, f.y_test)

        mdi_per_fold[f.fold_id] = mdi
        pfi_per_fold[f.fold_id] = pfi

        print(f"    MDI top-3: {list(mdi.head(3).index)}", flush=True)
        print(f"    PFI top-3: {list(pfi.head(3).index)}", flush=True)

        mdi.to_csv(OUTPUT_DIR / f"fold_{f.fold_id}_mdi.csv")
        pfi.to_csv(OUTPUT_DIR / f"fold_{f.fold_id}_pfi.csv")

    print("\nComputing stability metrics (MDI)...", flush=True)
    mdi_stability = compute_stability(mdi_per_fold, TOP_K)

    print("Computing stability metrics (PFI)...", flush=True)
    pfi_stability = compute_stability(pfi_per_fold, TOP_K)

    elapsed = time.monotonic() - t0

    print(f"\n{'=' * 70}", flush=True)
    print(f"  FEATURE IMPORTANCE STABILITY ({elapsed:.0f}s)", flush=True)
    print(f"{'=' * 70}\n", flush=True)

    for method, stability in [("MDI", mdi_stability), ("PFI", pfi_stability)]:
        sp = stability["spearman"]
        jc = stability["jaccard"]
        sm = stability["summary"]

        off_diag_sp = sp.values[np.triu_indices_from(sp.values, k=1)]
        off_diag_jc = jc.values[np.triu_indices_from(jc.values, k=1)]

        print(f"  --- {method} ---", flush=True)
        print(
            f"  Mean fold Spearman rank corr: {np.mean(off_diag_sp):.3f} ± {np.std(off_diag_sp):.3f}",
            flush=True,
        )
        print(
            f"  Mean top-{TOP_K} Jaccard:         {np.mean(off_diag_jc):.3f} ± {np.std(off_diag_jc):.3f}",
            flush=True,
        )
        print(flush=True)

        print("  Top features by selection frequency:", flush=True)
        print(
            f"  {'Feature':<35s} {'MeanImp':>8s} {'StdImp':>8s} {'MeanRank':>8s} {'StdRank':>8s} {'Freq':>6s}",
            flush=True,
        )
        print(f"  {'─' * 78}", flush=True)
        for _, r in sm.head(15).iterrows():
            print(
                f"  {r['feature']:<35s} {r['mean_importance']:>8.4f} {r['std_importance']:>8.4f} "
                f"{r['mean_rank']:>8.1f} {r['std_rank']:>8.1f} {r[f'top_{TOP_K}_freq']:>6.0%}",
                flush=True,
            )
        print(flush=True)

        sp.to_csv(OUTPUT_DIR / f"{method.lower()}_spearman.csv")
        jc.to_csv(OUTPUT_DIR / f"{method.lower()}_jaccard.csv")
        sm.to_csv(OUTPUT_DIR / f"{method.lower()}_summary.csv", index=False)

    print(f"  Saved diagnostics to {OUTPUT_DIR}/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
