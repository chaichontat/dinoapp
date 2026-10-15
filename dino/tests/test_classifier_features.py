from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dino.artifacts import file_sha256
from dino.classifier_features import (
    EMBEDDING_WIDTH,
    assemble_classifier_features,
    load_classifier_context,
    load_context,
    validate_feature_contract,
)
from dino_markers import (
    DEFAULT_CONTRACT,
    ContractMismatchError,
    MarkerContract,
    MissingContractError,
    UnsupportedArtifactError,
)

CUSTOM = MarkerContract("dapi", ("edu", "brdu", "pcna"))

RATIO_FEATURES = ("log1p_cell_mean", "log_cell_over_local_mean", "log1p_dapi_mean",
                  "robust_z_log_marker_over_dapi_k100")


def _props(tmp_path, *, brdu, dapi, pcna=None):
    cells = len(brdu)
    rng = np.random.default_rng(0)
    extra = {} if pcna is None else {"intensity_pcna_mean": pcna}
    pd.DataFrame({
        "label": np.arange(1, cells + 1, dtype=np.uint32),
        "area": np.full(cells, 50, dtype=np.int64),
        "centroid_z": np.zeros(cells),
        "centroid_y": np.zeros(cells),
        "centroid_x": np.arange(cells, dtype=np.float64),
        "intensity_brdu_mean": brdu,
        "intensity_edu_mean": rng.uniform(10, 500, cells),
        **extra,
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
    context = load_context(RATIO_FEATURES, props, dapi_props, scale_zyx_um=np.ones(3), markers=DEFAULT_CONTRACT.markers)

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
    context = load_context(RATIO_FEATURES, props, dapi_props, scale_zyx_um=np.ones(3), markers=DEFAULT_CONTRACT.markers)
    with pytest.raises(ValueError, match="zero spread"):
        context.features_for_labels(np.array([10]), neighbors=16)


def test_classifier_context_loads_ratio_features_and_rejects_unknown_sets(tmp_path):
    rng = np.random.default_rng(2)
    props, dapi_props = _props(tmp_path, brdu=rng.uniform(10, 500, 120), dapi=rng.uniform(100, 300, 120))
    bundle = {"schema": "dino-knn-intensity-v1", "context_features": list(RATIO_FEATURES), "intensity_scale_zyx_um": [1.0, 1.0, 1.0],
              "intensity_props": str(props), "intensity_props_sha256": file_sha256(props),
              "dapi_props": str(dapi_props), "dapi_props_sha256": file_sha256(dapi_props)}
    features = load_classifier_context(bundle).features_for_labels(np.array([5]), neighbors=16)
    assert features["edu"].shape == (1, 4)
    with pytest.raises(ValueError, match="Unsupported"):
        load_classifier_context({**bundle, "context_features": [*RATIO_FEATURES[:3], "robust_z_k50"]})


def _classifier(contract=DEFAULT_CONTRACT, *, mode="marker", width=None, spatial_subtraction=None):
    context = ["log1p_cell_mean", "log_cell_over_local_mean"]
    markers = contract.markers
    if width is None:
        width = (EMBEDDING_WIDTH + len(context)) * (len(markers) if mode == "joint" else 1)
    bundle = {
        "schema": "dino-knn-intensity-v2",
        "marker_contract": contract.to_json(),
        "classifier_feature_mode": mode,
        "context_features": context,
        "spatial_subtraction": spatial_subtraction,
        **{f"{marker}_feature_scaler": SimpleNamespace(n_features_in_=width) for marker in markers},
    }
    return bundle


@pytest.mark.parametrize("left,right", [(None, {"model_sha256": "a"}),
                                         ({"model_sha256": "a"}, None),
                                         ({"model_sha256": "a"}, {"model_sha256": "b"})])
def test_feature_contract_requires_matching_spatial_subtraction(left, right):
    base = _classifier()
    validate_feature_contract({**base, "spatial_subtraction": left}, {**base, "spatial_subtraction": left})
    with pytest.raises(ValueError, match="spatial_subtraction"):
        validate_feature_contract({**base, "spatial_subtraction": left}, {**base, "spatial_subtraction": right})


def test_feature_contract_returns_the_bundle_contract():
    assert validate_feature_contract(_classifier(), _classifier()) == DEFAULT_CONTRACT
    legacy = {**_classifier(), "schema": "dino-knn-intensity-v1"}
    del legacy["marker_contract"]
    assert validate_feature_contract(legacy, legacy) == DEFAULT_CONTRACT
    assert validate_feature_contract(legacy, _classifier()) == DEFAULT_CONTRACT
    assert validate_feature_contract(_classifier(CUSTOM), _classifier(CUSTOM)) == CUSTOM


def test_feature_contract_rejects_a_same_length_reorder_of_the_embedding_bundle():
    reordered = MarkerContract("pi", ("edu", "brdu"))
    with pytest.raises(ContractMismatchError, match="marker order is part of the contract"):
        validate_feature_contract(_classifier(), _classifier(reordered))
    with pytest.raises(ContractMismatchError, match="persisted embeddings"):
        validate_feature_contract(_classifier(reordered), _classifier())


@pytest.mark.parametrize("schema", ["dino-zarr-model-v6", "dino-knn-intensity-v3", None])
def test_feature_contract_requires_intensity_model_schemas(schema):
    bundle = {**_classifier(), "schema": schema}
    with pytest.raises(ValueError, match="model and embedding schema"):
        validate_feature_contract(bundle, _classifier())
    with pytest.raises(ValueError, match="model and embedding schema"):
        validate_feature_contract(_classifier(), bundle)


def test_feature_contract_names_a_current_bundle_without_its_contract():
    bundle = _classifier()
    del bundle["marker_contract"]
    with pytest.raises(MissingContractError, match="classifier bundle"):
        validate_feature_contract(bundle, _classifier())
    with pytest.raises(MissingContractError, match="embedding bundle"):
        validate_feature_contract(_classifier(), bundle)


@pytest.mark.parametrize("mode, width, accepted", [
    ("marker", 1026, True), ("joint", 3078, True), ("marker", 3078, False), ("joint", 1026, False),
])
def test_three_marker_feature_widths_per_mode(mode, width, accepted):
    bundle = _classifier(CUSTOM, mode=mode, width=width)
    if accepted:
        assert validate_feature_contract(bundle, _classifier(CUSTOM)) == CUSTOM
    else:
        with pytest.raises(ValueError, match="Classifier feature dimensions"):
            validate_feature_contract(bundle, _classifier(CUSTOM))


def test_three_marker_widths_are_read_from_each_markers_scaler():
    bundle = _classifier(CUSTOM)
    del bundle["pcna_feature_scaler"]
    with pytest.raises(KeyError, match="pcna_feature_scaler"):
        validate_feature_contract(bundle, _classifier(CUSTOM))


def test_joint_assembly_follows_the_marker_order_it_is_given():
    embeddings = {"brdu": np.full((2, 2), 1.0), "edu": np.full((2, 2), 2.0)}
    contexts = {"brdu": np.full((2, 1), 3.0), "edu": np.full((2, 1), 4.0)}
    for markers, expected in ((("brdu", "edu"), [1, 1, 2, 2, 3, 4]), (("edu", "brdu"), [2, 2, 1, 1, 4, 3])):
        np.testing.assert_array_equal(
            assemble_classifier_features(embeddings, contexts, target="brdu", mode="joint", markers=markers),
            [expected] * 2,
        )


def test_three_marker_assembly_and_unknown_target():
    markers = CUSTOM.markers
    embeddings = {marker: np.full((1, 2), float(index)) for index, marker in enumerate(markers)}
    contexts = {marker: np.full((1, 1), 10.0 + index) for index, marker in enumerate(markers)}

    np.testing.assert_array_equal(
        assemble_classifier_features(embeddings, contexts, target="pcna", mode="joint", markers=markers),
        [[0, 0, 1, 1, 2, 2, 10, 11, 12]],
    )
    np.testing.assert_array_equal(
        assemble_classifier_features(embeddings, contexts, target="brdu", mode="marker", markers=markers),
        [[1, 1, 11]],
    )
    with pytest.raises(ValueError, match="Unknown classifier target: dapi"):
        assemble_classifier_features(embeddings, contexts, target="dapi", mode="marker", markers=markers)
    with pytest.raises(ValueError, match="Unknown classifier feature mode"):
        assemble_classifier_features(embeddings, contexts, target="edu", mode="other", markers=markers)


def test_dapi_and_ratio_contexts_cover_every_contract_marker(tmp_path):
    rng = np.random.default_rng(4)
    brdu, dapi, pcna = rng.uniform(10, 500, 150), rng.uniform(100, 300, 150), rng.uniform(10, 500, 150)
    props, dapi_props = _props(tmp_path, brdu=brdu, dapi=dapi, pcna=pcna)

    context = load_context(RATIO_FEATURES, props, dapi_props, scale_zyx_um=np.ones(3), markers=CUSTOM.markers)
    features = context.features_for_labels(np.array([1, 75]), neighbors=16)

    assert list(features) == ["edu", "brdu", "pcna"]
    ratio = np.log((pcna + 1) / (dapi + 1))
    index = 74
    distance = np.abs(np.arange(150) - index).astype(float)
    distance[index] = np.inf
    around = ratio[np.argsort(distance, kind="stable")[:100]]
    center = np.median(around)
    spread = 1.4826 * np.median(np.abs(around - center))
    assert features["pcna"].shape == (2, 4)
    assert features["pcna"][1, 2] == pytest.approx(np.log1p(dapi[index]))
    assert features["pcna"][1, 3] == pytest.approx((ratio[index] - center) / spread)


def test_classifier_context_takes_its_markers_from_the_bundle_contract(tmp_path):
    rng = np.random.default_rng(5)
    props, dapi_props = _props(
        tmp_path, brdu=rng.uniform(10, 500, 120), dapi=rng.uniform(100, 300, 120), pcna=rng.uniform(10, 500, 120)
    )
    bundle = {"schema": "dino-knn-intensity-v2", "marker_contract": CUSTOM.to_json(),
              "context_features": list(RATIO_FEATURES), "intensity_scale_zyx_um": [1.0, 1.0, 1.0],
              "intensity_props": str(props), "intensity_props_sha256": file_sha256(props),
              "dapi_props": str(dapi_props), "dapi_props_sha256": file_sha256(dapi_props)}

    features = load_classifier_context(bundle).features_for_labels(np.array([5]), neighbors=16)

    assert list(features) == ["edu", "brdu", "pcna"]
    base = {**bundle, "context_features": ["log1p_cell_mean", "log_cell_over_local_mean"]}
    assert list(load_classifier_context(base).features_for_labels(np.array([5]), neighbors=16)) == list(CUSTOM.markers)
    with pytest.raises(UnsupportedArtifactError, match="classifier bundle"):
        load_classifier_context({k: v for k, v in bundle.items() if k != "schema"})
