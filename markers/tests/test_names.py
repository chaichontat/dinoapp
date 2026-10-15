import pytest

from dino_markers import names
from dino_markers import (
    BARRAGE_STATS,
    CLASSIFIER_METRICS,
    cutoff_key,
    image_path_column,
    intensity_column,
    metric_key,
    miscall_column,
    model_key,
    oof_probability_column,
    positive_count_key,
    prediction_column,
    probability_column,
    rhu2_probability_column,
    saved_column,
    scaler_key,
    stat_column,
)


def test_constants():
    assert CLASSIFIER_METRICS == ("best_c", "cv_balanced_accuracy", "cv_folds", "validation_balanced_accuracy")
    assert BARRAGE_STATS == ("mean", "min", "max", "median", "std")


@pytest.mark.parametrize("key", ["brdu", "edu", "pcna", "m7"])
def test_per_marker_names(key):
    assert probability_column(key) == f"p_{key}"
    assert prediction_column(key) == f"pred_{key}"
    assert oof_probability_column(key) == f"oof_p_{key}"
    assert saved_column(key) == f"saved_{key}"
    assert miscall_column(key) == f"{key}_miscall"
    assert rhu2_probability_column(key) == f"{key}_prob"
    assert image_path_column(key) == f"{key}_image_path"
    assert cutoff_key(key) == f"{key}_cutoff"
    assert model_key(key) == f"{key}_model"
    assert scaler_key(key) == f"{key}_feature_scaler"
    assert positive_count_key(key) == f"{key}_positive"
    for metric in CLASSIFIER_METRICS:
        assert metric_key(key, metric) == f"{key}_{metric}"
    for stat in BARRAGE_STATS:
        assert stat_column(key, stat) == f"{key}_{stat}"
    assert intensity_column(key) == f"intensity_{key}_mean"
    assert intensity_column(key, "std") == f"intensity_{key}_std"


def test_default_contract_reproduces_todays_names():
    assert [probability_column(k) for k in ("brdu", "edu")] == ["p_brdu", "p_edu"]
    assert [prediction_column(k) for k in ("brdu", "edu")] == ["pred_brdu", "pred_edu"]
    assert [cutoff_key(k) for k in ("brdu", "edu")] == ["brdu_cutoff", "edu_cutoff"]
    assert [rhu2_probability_column(k) for k in ("brdu", "edu")] == ["brdu_prob", "edu_prob"]
    assert [image_path_column(k) for k in ("pi", "edu", "brdu")] == ["pi_image_path", "edu_image_path", "brdu_image_path"]
    assert [model_key(k) for k in ("brdu", "edu")] == ["brdu_model", "edu_model"]
    assert [scaler_key(k) for k in ("brdu", "edu")] == ["brdu_feature_scaler", "edu_feature_scaler"]
    assert [intensity_column(k) for k in ("brdu", "edu")] == ["intensity_brdu_mean", "intensity_edu_mean"]
    assert [stat_column("pi", s) for s in BARRAGE_STATS] == ["pi_mean", "pi_min", "pi_max", "pi_median", "pi_std"]
    assert metric_key("brdu", "best_c") == "brdu_best_c"
    assert positive_count_key("edu") == "edu_positive"
    assert miscall_column("edu") == "edu_miscall"
    assert saved_column("brdu") == "saved_brdu"
    assert oof_probability_column("edu") == "oof_p_edu"


def test_names_module_has_every_function():
    assert names.probability_column is probability_column
