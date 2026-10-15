"""Persisted per-marker names, each derived from the marker key."""

from __future__ import annotations

CLASSIFIER_METRICS: tuple[str, ...] = (
    "best_c",
    "cv_balanced_accuracy",
    "cv_folds",
    "validation_balanced_accuracy",
)
BARRAGE_STATS: tuple[str, ...] = ("mean", "min", "max", "median", "std")


def probability_column(key: str) -> str:
    return f"p_{key}"


def prediction_column(key: str) -> str:
    return f"pred_{key}"


def oof_probability_column(key: str) -> str:
    return f"oof_p_{key}"


def saved_column(key: str) -> str:
    return f"saved_{key}"


def miscall_column(key: str) -> str:
    return f"{key}_miscall"


def rhu2_probability_column(key: str) -> str:
    return f"{key}_prob"


def image_path_column(key: str) -> str:
    return f"{key}_image_path"


def cutoff_key(key: str) -> str:
    return f"{key}_cutoff"


def model_key(key: str) -> str:
    return f"{key}_model"


def scaler_key(key: str) -> str:
    return f"{key}_feature_scaler"


def metric_key(key: str, metric: str) -> str:
    return f"{key}_{metric}"


def positive_count_key(key: str) -> str:
    return f"{key}_positive"


def stat_column(key: str, stat: str) -> str:
    return f"{key}_{stat}"


def intensity_column(key: str, stat: str = "mean") -> str:
    return f"intensity_{key}_{stat}"


def derived_names(key: str) -> tuple[str, ...]:
    """Every persisted name that a key produces, including the key itself."""
    return (
        key,
        probability_column(key),
        prediction_column(key),
        oof_probability_column(key),
        saved_column(key),
        miscall_column(key),
        rhu2_probability_column(key),
        image_path_column(key),
        cutoff_key(key),
        model_key(key),
        scaler_key(key),
        positive_count_key(key),
        *(metric_key(key, metric) for metric in CLASSIFIER_METRICS),
        *(stat_column(key, stat) for stat in BARRAGE_STATS),
        intensity_column(key),
    )
