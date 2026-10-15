import copy

import pytest

from dino_markers import (
    BARRAGE,
    CONTRACT_FIELD,
    CORTEX_RESCORE,
    DEFAULT_CONTRACT,
    FEATURE_CACHE,
    INFERENCE,
    INTENSITY_MODEL,
    JOINT_MODEL,
    MODEL_KINDS,
    PREPARED_SOURCE,
    SPATIAL_REVIEW,
    ZARR_MODEL,
    ArtifactKind,
    MarkerConfigError,
    MarkerContract,
    MissingContractError,
    UnsupportedArtifactError,
    bundle_contract,
    derive_bundle,
)

DEFAULT_JSON = {"version": 1, "reference": "pi", "markers": ["brdu", "edu"]}
REORDERED = MarkerContract("pi", ("edu", "brdu"))
CUSTOM = MarkerContract("dapi", ("edu", "brdu", "pcna"))

KINDS = [
    (PREPARED_SOURCE, "prepared source completion", "artifact",
     "dino-zarr-source-complete-v1", "dino-zarr-source-complete-v2"),
    (BARRAGE, "barrage record", "artifact", "dino-zarr-barrage-v3", "dino-zarr-barrage-v4"),
    (FEATURE_CACHE, "feature cache", "schema", "dino-zarr-features-v2", "dino-zarr-features-v3"),
    (ZARR_MODEL, "model bundle", "schema", "dino-zarr-model-v6", "dino-zarr-model-v7"),
    (INTENSITY_MODEL, "intensity-context model bundle", "schema",
     "dino-knn-intensity-v1", "dino-knn-intensity-v2"),
    (JOINT_MODEL, "joint model bundle", "schema", "dino-joint-dataset-v1", "dino-joint-dataset-v2"),
    (INFERENCE, "inference record", "artifact", "dino-zarr-infer-v2", "dino-zarr-infer-v3"),
    (CORTEX_RESCORE, "cortex rescore record", "artifact", "dino-cortex-rescore-v1", "dino-cortex-rescore-v2"),
    (SPATIAL_REVIEW, "RHU2 spatial review manifest", "artifact",
     "rhu2-spatial-review-v1", "rhu2-spatial-review-v2"),
]
IDS = [kind[0].current for kind in KINDS]


def test_kind_constants_match_the_spec():
    for kind, name, key, legacy, current in KINDS:
        assert (kind.name, kind.version_key, kind.legacy, kind.current) == (name, key, legacy, current)
    assert BARRAGE.legacy_fields == {"crop_panels": [["pi", "PI"], ["edu", "EdU"], ["brdu", "BrdU"]]}
    for kind in (PREPARED_SOURCE, FEATURE_CACHE, ZARR_MODEL, INTENSITY_MODEL, JOINT_MODEL, INFERENCE,
                 CORTEX_RESCORE, SPATIAL_REVIEW):
        assert dict(kind.legacy_fields) == {}
    assert MODEL_KINDS == (ZARR_MODEL, INTENSITY_MODEL, JOINT_MODEL)


@pytest.mark.parametrize(("kind", "name", "key", "legacy", "current"), KINDS, ids=IDS)
def test_legacy_read(kind, name, key, legacy, current):
    record = {key: legacy, "other": 1}
    assert kind.recognizes(record)
    assert kind.contract(record, where="f.json") == DEFAULT_CONTRACT


@pytest.mark.parametrize(("kind", "name", "key", "legacy", "current"), KINDS, ids=IDS)
def test_current_read(kind, name, key, legacy, current):
    record = {key: current, CONTRACT_FIELD: CUSTOM.to_json()}
    assert kind.recognizes(record)
    assert kind.contract(record, where="f.json") == CUSTOM
    record[CONTRACT_FIELD] = DEFAULT_JSON
    assert kind.contract(record, where="f.json") == DEFAULT_CONTRACT


@pytest.mark.parametrize(("kind", "name", "key", "legacy", "current"), KINDS, ids=IDS)
def test_current_without_the_field_raises_naming_the_file(kind, name, key, legacy, current):
    with pytest.raises(MissingContractError) as info:
        kind.contract({key: current}, where="/data/complete.json")
    assert str(info.value) == f"/data/complete.json: {name} {key}={current!r} has no marker_contract"
    with pytest.raises(MissingContractError):
        kind.normalize({key: current}, where="/data/complete.json")


@pytest.mark.parametrize(("kind", "name", "key", "legacy", "current"), KINDS, ids=IDS)
def test_legacy_with_the_field_raises(kind, name, key, legacy, current):
    with pytest.raises(MarkerConfigError) as info:
        kind.contract({key: legacy, CONTRACT_FIELD: DEFAULT_JSON}, where="f.json")
    assert str(info.value) == f"f.json: {legacy} records must not carry {CONTRACT_FIELD}"


@pytest.mark.parametrize(("kind", "name", "key", "legacy", "current"), KINDS, ids=IDS)
def test_unknown_missing_and_null_versions(kind, name, key, legacy, current):
    with pytest.raises(UnsupportedArtifactError) as info:
        kind.contract({key: "other-v9"}, where="f.json")
    assert str(info.value) == (
        f"f.json: unsupported {name} {key}='other-v9'; expected {legacy!r} or {current!r}"
    )
    for record in ({}, {key: None}, {"other": legacy}):
        assert not kind.recognizes(record)
        with pytest.raises(UnsupportedArtifactError, match=rf"^f\.json: unsupported {name} {key}=None"):
            kind.contract(record, where="f.json")
    assert not kind.recognizes({key: "other-v9"})


@pytest.mark.parametrize(("kind", "name", "key", "legacy", "current"), KINDS, ids=IDS)
def test_stamp_for_default_and_reordered(kind, name, key, legacy, current):
    stamped = kind.stamp({"a": 1, "b": [2]}, DEFAULT_CONTRACT)
    assert stamped == {key: current, "a": 1, "b": [2], CONTRACT_FIELD: DEFAULT_JSON}
    assert list(stamped) == [key, "a", "b", CONTRACT_FIELD]
    reordered = kind.stamp({"a": 1}, REORDERED)
    assert reordered[CONTRACT_FIELD] == {"version": 1, "reference": "pi", "markers": ["edu", "brdu"]}
    assert kind.contract(reordered, where="x") == REORDERED
    assert kind.recognizes(reordered)


@pytest.mark.parametrize(("kind", "name", "key", "legacy", "current"), KINDS, ids=IDS)
def test_stamp_refuses_version_and_contract_fields(kind, name, key, legacy, current):
    with pytest.raises(ValueError):
        kind.stamp({key: current}, DEFAULT_CONTRACT)
    with pytest.raises(ValueError):
        kind.stamp({CONTRACT_FIELD: DEFAULT_JSON}, DEFAULT_CONTRACT)


@pytest.mark.parametrize(("kind", "name", "key", "legacy", "current"), KINDS, ids=IDS)
def test_normalize_current_and_legacy(kind, name, key, legacy, current):
    record = kind.stamp({"a": 1}, CUSTOM)
    normalized = kind.normalize(record, where="x")
    assert normalized == record and normalized is not record
    old = {key: legacy, "a": 1}
    snapshot = copy.deepcopy(old)
    expected = {"a": 1, key: current, CONTRACT_FIELD: DEFAULT_JSON, **copy.deepcopy(dict(kind.legacy_fields))}
    assert kind.normalize(old, where="x") == expected
    assert old == snapshot
    assert kind.normalize(old, where="x") == kind.stamp({"a": 1, **kind.legacy_fields}, DEFAULT_CONTRACT)


def test_barrage_v3_normalizes_with_legacy_panels_and_default_contract():
    record = {"artifact": "dino-zarr-barrage-v3", "size": 16}
    assert BARRAGE.normalize(record, where="run.json") == {
        "artifact": "dino-zarr-barrage-v4",
        "size": 16,
        "crop_panels": [["pi", "PI"], ["edu", "EdU"], ["brdu", "BrdU"]],
        "marker_contract": DEFAULT_JSON,
    }
    normalized = BARRAGE.normalize(record, where="run.json")
    normalized["crop_panels"][0][0] = "changed"
    assert BARRAGE.legacy_fields["crop_panels"][0][0] == "pi"


def test_legacy_barrage_carrying_crop_panels_is_rejected():
    record = {"artifact": "dino-zarr-barrage-v3", "crop_panels": [["pi", "PI"]]}
    with pytest.raises(MarkerConfigError, match=r"dino-zarr-barrage-v3 records must not carry crop_panels"):
        BARRAGE.contract(record, where="run.json")
    with pytest.raises(MarkerConfigError):
        BARRAGE.normalize(record, where="run.json")


def test_current_barrage_keeps_its_panels():
    record = BARRAGE.stamp({"crop_panels": [["brdu", "BrdU"], ["pi", "PI"], ["edu", "EdU"]]}, DEFAULT_CONTRACT)
    assert BARRAGE.normalize(record, where="x")["crop_panels"][0] == ["brdu", "BrdU"]


def test_bundle_contract_for_every_model_kind():
    for kind in MODEL_KINDS:
        assert bundle_contract({"schema": kind.legacy}, where="m.joblib") == DEFAULT_CONTRACT
        assert bundle_contract(kind.stamp({}, CUSTOM), where="m.joblib") == CUSTOM
        with pytest.raises(MissingContractError, match="^m.joblib: "):
            bundle_contract({"schema": kind.current}, where="m.joblib")


@pytest.mark.parametrize("bundle", [{}, {"schema": None}, {"schema": "dino-cortex-rescore-v1"}, {"schema": "dino-zarr-model-v5"}])
def test_bundle_contract_rejects_unknown_schemas(bundle):
    with pytest.raises(UnsupportedArtifactError, match=r"^m.joblib: "):
        bundle_contract(bundle, where="m.joblib")
    with pytest.raises(UnsupportedArtifactError, match=r"^m.joblib: "):
        derive_bundle(bundle, {}, where="m.joblib")


def test_derive_bundle_upgrades_knn_v1_with_the_default_contract():
    bundle = {"schema": "dino-knn-intensity-v1", "brdu_model": "m", "keep": [1]}
    snapshot = copy.deepcopy(bundle)
    derived = derive_bundle(bundle, {"new": 5, "keep": [2]}, where="b")
    assert derived == {
        "schema": "dino-knn-intensity-v2",
        "brdu_model": "m",
        "keep": [2],
        "new": 5,
        "marker_contract": DEFAULT_JSON,
    }
    assert bundle == snapshot


def test_derive_bundle_keeps_a_v2_bundles_custom_contract():
    bundle = INTENSITY_MODEL.stamp({"x": 1}, CUSTOM)
    derived = derive_bundle(bundle, {"y": 2}, where="b")
    assert derived == {**bundle, "y": 2}
    assert derived["marker_contract"] == CUSTOM.to_json()
    assert derived is not bundle


def test_derive_bundle_stays_in_the_input_kind():
    assert derive_bundle({"schema": "dino-zarr-model-v6"}, {}, where="b")["schema"] == "dino-zarr-model-v7"
    assert derive_bundle({"schema": "dino-joint-dataset-v1"}, {}, where="b")["schema"] == "dino-joint-dataset-v2"


@pytest.mark.parametrize("update", ["schema", "marker_contract"])
def test_derive_bundle_refuses_schema_and_contract_updates(update):
    with pytest.raises(ValueError):
        derive_bundle({"schema": "dino-knn-intensity-v1"}, {update: 1}, where="b")


def test_artifact_kind_is_a_frozen_dataclass():
    kind = ArtifactKind("thing", "artifact", "t-v1", "t-v2")
    assert kind.legacy_fields == {}
    with pytest.raises(AttributeError):
        kind.name = "other"
