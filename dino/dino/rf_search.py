from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold, cross_val_predict

from fishtools.brdu.features import features_array_from_frame
from fishtools.brdu.model import load_barrage_training_table


@dataclass(frozen=True)
class SearchConfig:
    barrage_dir: Path
    seed: int
    n_iter: int
    n_splits: int
    out_csv: Path


def _parse_args() -> SearchConfig:
    p = argparse.ArgumentParser(description="Quick RF hyperparameter search for BrdU/EdU AUC.")
    p.add_argument("--barrage-dir", type=Path, default=Path("output/brdu_edu_barrage"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-iter", type=int, default=30, help="Number of sampled hyperparameter configs.")
    p.add_argument("--n-splits", type=int, default=5, help="Stratified CV folds (clamped by minority class).")
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV path. Default: <barrage-dir>/rf_hyperparam_search.csv",
    )
    args = p.parse_args()
    barrage_dir = Path(args.barrage_dir)
    out_csv = Path(args.out_csv) if args.out_csv is not None else (barrage_dir / "rf_hyperparam_search.csv")
    return SearchConfig(
        barrage_dir=barrage_dir,
        seed=int(args.seed),
        n_iter=int(args.n_iter),
        n_splits=int(args.n_splits),
        out_csv=out_csv,
    )


def _cv_auc_for_target(
    *,
    train: pd.DataFrame,
    target: str,
    params: dict[str, object],
    seed: int,
    n_splits: int,
) -> tuple[float, int]:
    y = train[target].to_numpy(dtype=int)
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        raise ValueError(f"Target={target}: only one class present.")
    n_splits_eff = min(int(n_splits), int(np.min(counts)))
    if n_splits_eff < 2:
        raise ValueError(
            f"Target={target}: not enough minority samples for CV (min_class={int(np.min(counts))})."
        )

    X = features_array_from_frame(train, target=target)
    cv = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)
    clf = RandomForestClassifier(
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        **params,
    )
    prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
    score = np.asarray(prob[:, 1], dtype=np.float64)
    return float(roc_auc_score(y, score)), n_splits_eff


def main() -> None:
    cfg = _parse_args()
    train = load_barrage_training_table(barrage_dir=cfg.barrage_dir)
    logger.info(f"Loaded training table: n={len(train)} from {cfg.barrage_dir}")

    space: dict[str, list[object]] = {
        "n_estimators": [200, 400, 800],
        "max_depth": [None, 6, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", 0.5, 1.0],
    }
    sampler = ParameterSampler(space, n_iter=cfg.n_iter, random_state=cfg.seed)

    rows: list[dict[str, object]] = []
    for i, params in enumerate(sampler, start=1):
        logger.info(f"[{i}/{cfg.n_iter}] params={params}")
        brdu_auc, brdu_splits = _cv_auc_for_target(
            train=train, target="brdu", params=params, seed=cfg.seed, n_splits=cfg.n_splits
        )
        edu_auc, edu_splits = _cv_auc_for_target(
            train=train, target="edu", params=params, seed=cfg.seed, n_splits=cfg.n_splits
        )
        rows.append(
            {
                **params,
                "seed": cfg.seed,
                "brdu_auc": brdu_auc,
                "edu_auc": edu_auc,
                "mean_auc": 0.5 * (brdu_auc + edu_auc),
                "brdu_splits": brdu_splits,
                "edu_splits": edu_splits,
            }
        )

    out = pd.DataFrame(rows).sort_values("mean_auc", ascending=False).reset_index(drop=True)
    cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.out_csv, index=False)
    logger.info(f"Wrote: {cfg.out_csv}")

    top = out.head(10)
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
