from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from fishtools.brdu.features import (
    META_REQUIRED_COLS,
    ensure_log_means,
    feature_names as _feature_names,
    features_array_from_frame as _features_array_from_frame,
)


@dataclass(frozen=True)
class SweepConfig:
    index_csv: Path
    labels_csv: Path
    out_dir: Path
    seed: int
    n_splits: int


def _load_training_table(cfg: SweepConfig) -> pd.DataFrame:
    meta = pd.read_csv(cfg.index_csv)
    labels = pd.read_csv(cfg.labels_csv)

    missing = sorted(set(META_REQUIRED_COLS) - set(meta.columns))
    if missing:
        raise ValueError(f"index.csv missing required columns: {missing}")
    if "cell" not in meta.columns:
        raise ValueError("index.csv missing required column: cell")
    if "cell" not in labels.columns:
        raise ValueError("labels.csv missing required column: cell")
    for col in ("brdu", "edu"):
        if col not in labels.columns:
            raise ValueError(f"labels.csv missing required column: {col}")

    meta["cell"] = meta["cell"].astype(str)
    labels["cell"] = labels["cell"].astype(str)
    labels["brdu"] = labels["brdu"].astype(int)
    labels["edu"] = labels["edu"].astype(int)

    df = labels.merge(meta, on="cell", how="inner")
    ensure_log_means(df)
    return df


def _models(seed: int) -> dict[str, object]:
    return {
        "logreg": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        solver="liblinear", class_weight="balanced", random_state=seed, max_iter=2000
                    ),
                ),
            ]
        ),
        "svm_linear": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("clf", SVC(kernel="linear", probability=True, class_weight="balanced", random_state=seed)),
            ]
        ),
        "svm_rbf": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
            ]
        ),
        "rf": RandomForestClassifier(
            n_estimators=400,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=600,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "gboost": GradientBoostingClassifier(random_state=seed),
    }


def _predict_scores(model: object, X: np.ndarray, y: np.ndarray, cv: StratifiedKFold) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
        return np.asarray(prob[:, 1], dtype=np.float64)
    if hasattr(model, "decision_function"):
        score = cross_val_predict(model, X, y, cv=cv, method="decision_function")
        return np.asarray(score, dtype=np.float64)
    raise TypeError(f"Model does not support predict_proba or decision_function: {type(model)}")


def _feature_importance_table(model: object, feature_names: list[str]) -> pd.DataFrame | None:
    clf = model
    if isinstance(model, Pipeline):
        clf = model.named_steps["clf"]

    if hasattr(clf, "feature_importances_"):
        vals = np.asarray(getattr(clf, "feature_importances_"), dtype=np.float64)
        return pd.DataFrame({"feature": feature_names, "importance": vals}).sort_values(
            "importance", ascending=False
        )

    if hasattr(clf, "coef_"):
        coef = np.asarray(getattr(clf, "coef_"))[0].astype(np.float64)
        return (
            pd.DataFrame({"feature": feature_names, "coef": coef})
            .assign(abs_coef=np.abs(coef))
            .sort_values("abs_coef", ascending=False)
        )

    return None


def _plot_rocs(results: dict[str, dict[str, object]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), dpi=160)
    for ax, target in zip(axs, ("brdu", "edu"), strict=True):
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        for name, r in results[target].items():
            fpr, tpr, _thr = roc_curve(r["y_true"], r["score"])
            ax.plot(fpr, tpr, linewidth=2, label=f"{name} AUC={r['auc']:.3f}")
        ax.set_title(target)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _parse_args() -> SweepConfig:
    p = argparse.ArgumentParser(description="Evaluate simple BrdU/EdU models and plot ROC curves.")
    p.add_argument("--index-csv", type=Path, default=Path("output/brdu_edu_barrage/index.csv"))
    p.add_argument("--labels-csv", type=Path, default=Path("output/brdu_edu_barrage/labels.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("output/brdu_edu_barrage"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-splits", type=int, default=5)
    args = p.parse_args()
    return SweepConfig(
        index_csv=Path(args.index_csv),
        labels_csv=Path(args.labels_csv),
        out_dir=Path(args.out_dir),
        seed=int(args.seed),
        n_splits=int(args.n_splits),
    )


def main() -> None:
    cfg = _parse_args()
    df = _load_training_table(cfg)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loaded labeled set: n={len(df)} from {cfg.labels_csv}")
    results: dict[str, dict[str, dict[str, object]]] = {"brdu": {}, "edu": {}}
    auc_rows: list[dict[str, object]] = []

    for target in ("brdu", "edu"):
        y = df[target].to_numpy(dtype=int)
        class_counts = dict(zip(*np.unique(y, return_counts=True), strict=True))
        logger.info(f"Target={target}: class_counts={class_counts}")
        if len(class_counts) < 2:
            logger.warning(f"Skipping target={target}: only one class present.")
            continue

        min_class = int(min(class_counts.values()))
        n_splits = min(cfg.n_splits, min_class)
        if n_splits < 2:
            logger.warning(
                f"Skipping target={target}: not enough minority samples for CV (min_class={min_class})."
            )
            continue

        X = _features_array_from_frame(df, target=target)
        feature_names = _feature_names(target)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)

        for name, model in _models(cfg.seed).items():
            score = _predict_scores(model, X, y, cv)
            auc = float(roc_auc_score(y, score))
            results[target][name] = {"y_true": y, "score": score, "auc": auc}
            auc_rows.append({"target": target, "model": name, "auc": auc, "n_splits": n_splits})

            model.fit(X, y)
            imp = _feature_importance_table(model, feature_names)
            if imp is not None:
                out = cfg.out_dir / f"feature_importance_{target}_{name}.csv"
                imp.to_csv(out, index=False)

    auc_df = pd.DataFrame(auc_rows).sort_values(["target", "auc"], ascending=[True, False])
    auc_out = cfg.out_dir / "model_auc.csv"
    auc_df.to_csv(auc_out, index=False)
    logger.info(f"Wrote AUC table: {auc_out}")

    roc_out = cfg.out_dir / "model_roc_curves.png"
    _plot_rocs(results, roc_out)
    logger.info(f"Wrote ROC plot: {roc_out}")

    print(auc_df.to_string(index=False))


if __name__ == "__main__":
    main()
