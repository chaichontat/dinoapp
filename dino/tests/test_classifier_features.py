from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dino.artifacts import file_sha256
from dino.classifier_features import load_classifier_context, load_context

RATIO_FEATURES = ("log1p_cell_mean", "log_cell_over_local_mean", "log1p_dapi_mean",
                  "robust_z_log_marker_over_dapi_k100")


def _props(tmp_path, *, brdu, dapi):
    cells = len(brdu)
    rng = np.random.default_rng(0)
    pd.DataFrame({
        "label": np.arange(1, cells + 1, dtype=np.uint32),
        "area": np.full(cells, 50, dtype=np.int64),
        "centroid_z": np.zeros(cells),
        "centroid_y": np.zeros(cells),
        "centroid_x": np.arange(cells, dtype=np.float64),
        "intensity_brdu_mean": brdu,
        "intensity_edu_mean": rng.uniform(10, 500, cells),
    }).to_parquet(tmp_path / "props.parquet", index=False)
    pd.DataFrame({
        "label": np.arange(1, cells + 1, dtype=np.uint32),
        "area": np.full(cells, 50, dtype=np.int64),
        "intensity_dapi_mean": dapi,
    }).to_parquet(tmp_path / "dapi.parquet", index=False)
    return tmp_path / "props.parquet", tmp_path / "dapi.parquet"


def test_ratio_context_appends_robust_local_z_of_marker_over_dapi(tmp_path):
    rng = np.random.default_rng(1)
    brdu, dapi = rng.uniform(10, 500, 150), rng.uniform(100, 300, 150)
    props, dapi_props = _props(tmp_path, brdu=brdu, dapi=dapi)
    context = load_context(RATIO_FEATURES, props, dapi_props, scale_zyx_um=np.ones(3))

    features = context.features_for_labels(np.array([1, 75]), neighbors=16)

    ratio = np.log((brdu + 1) / (dapi + 1))
    assert features["brdu"].shape == (2, 4)
    for row, label in enumerate((1, 75)):
        index = label - 1
        distance = np.abs(np.arange(150) - index).astype(float)
        distance[index] = np.inf
        around = ratio[np.argsort(distance, kind="stable")[:100]]
        center = np.median(around)
        spread = 1.4826 * np.median(np.abs(around - center))
        assert features["brdu"][row, 2] == pytest.approx(np.log1p(dapi[index]))
        assert features["brdu"][row, 3] == pytest.approx((ratio[index] - center) / spread)


def test_ratio_context_rejects_zero_local_spread(tmp_path):
    props, dapi_props = _props(tmp_path, brdu=np.full(120, 50.0), dapi=np.full(120, 200.0))
    context = load_context(RATIO_FEATURES, props, dapi_props, scale_zyx_um=np.ones(3))
    with pytest.raises(ValueError, match="zero spread"):
        context.features_for_labels(np.array([10]), neighbors=16)


def test_classifier_context_loads_ratio_features_and_rejects_unknown_sets(tmp_path):
    rng = np.random.default_rng(2)
    props, dapi_props = _props(tmp_path, brdu=rng.uniform(10, 500, 120), dapi=rng.uniform(100, 300, 120))
    bundle = {"context_features": list(RATIO_FEATURES), "intensity_scale_zyx_um": [1.0, 1.0, 1.0],
              "intensity_props": str(props), "intensity_props_sha256": file_sha256(props),
              "dapi_props": str(dapi_props), "dapi_props_sha256": file_sha256(dapi_props)}
    features = load_classifier_context(bundle).features_for_labels(np.array([5]), neighbors=16)
    assert features["edu"].shape == (1, 4)
    with pytest.raises(ValueError, match="Unsupported"):
        load_classifier_context({**bundle, "context_features": [*RATIO_FEATURES[:3], "robust_z_k50"]})


@pytest.mark.parametrize("left,right", [(None, {"model_sha256": "a"}),
                                         ({"model_sha256": "a"}, None),
                                         ({"model_sha256": "a"}, {"model_sha256": "b"})])
def test_feature_contract_requires_matching_spatial_subtraction(left, right):
    from types import SimpleNamespace
    from dino.classifier_features import validate_feature_contract
    from dino.intensity_context import MODEL_SCHEMA
    base = {"schema": MODEL_SCHEMA, "context_features": ["log1p_cell_mean", "log_cell_over_local_mean"],
            **{f"{t}_feature_scaler": SimpleNamespace(n_features_in_=1026) for t in ("brdu", "edu")}}
    validate_feature_contract({**base, "spatial_subtraction": left}, {**base, "spatial_subtraction": left})
    with pytest.raises(ValueError, match="spatial_subtraction"):
        validate_feature_contract({**base, "spatial_subtraction": left}, {**base, "spatial_subtraction": right})
