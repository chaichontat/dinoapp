"""Frozen BrdU/EdU joint-fit artifacts (fixtures/legacy_joint) keep loading under the marker contract."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from dino_markers import DEFAULT_CONTRACT, INTENSITY_MODEL, JOINT_MODEL, ZARR_MODEL, bundle_contract, derive_bundle

from dino.artifacts import read_embedding_artifact
from dino.classifier_features import assemble_classifier_features, load_classifier_context
from dino.joint_train import fit


FIXTURE = Path(__file__).parent / "fixtures" / "legacy_joint" / "root"
ROOT = Path("/tmp/dino-wf4c-legacy-joint")
DATASETS = ("first", "second")


@pytest.fixture(scope="session")
def legacy_root() -> Path:
    """Place the frozen tree at the absolute root its bundles record."""
    if not (ROOT / "output" / "model.joblib").exists():
        staging = Path(tempfile.mkdtemp(prefix=".legacy-joint-", dir=ROOT.parent))
        try:
            shutil.copytree(FIXTURE, staging / "root")
            try:
                os.rename(staging / "root", ROOT)
            except OSError:
                if not (ROOT / "output" / "model.joblib").is_file():
                    raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    return ROOT


def test_legacy_joint_bundle_loads_with_the_default_contract(legacy_root) -> None:
    bundle = joblib.load(legacy_root / "output" / "model.joblib")

    assert bundle["schema"] == "dino-joint-dataset-v1"
    assert JOINT_MODEL.recognizes(bundle)
    assert "marker_contract" not in bundle
    assert bundle_contract(bundle, where="joint bundle") == DEFAULT_CONTRACT
    assert bundle["datasets"] == list(DATASETS)
    assert {key for key in bundle if key.startswith(("brdu_", "edu_"))} == {
        f"{target}_{suffix}" for target in ("brdu", "edu") for suffix in ("feature_scaler", "model", "cutoff")
    }
    assert bundle["feature_layout"] == [
        "embedding",
        *(f"{feature}|{name}" for name in DATASETS for feature in bundle["context_features"]),
        *(f"intercept|{name}" for name in DATASETS),
    ]
    assert bundle["brdu_model"].n_features_in_ == 1024 + 2 * (3 + 1)


@pytest.mark.parametrize("name", DATASETS)
def test_legacy_deployed_bundle_loads_with_the_default_contract(legacy_root, name) -> None:
    bundle = joblib.load(legacy_root / "output" / "deploy" / name / "model.joblib")

    assert bundle["schema"] == "dino-knn-intensity-v1"
    assert INTENSITY_MODEL.recognizes(bundle)
    assert "marker_contract" not in bundle
    assert bundle_contract(bundle, where="deployed bundle") == DEFAULT_CONTRACT
    assert bundle["classifier_feature_mode"] == "marker"
    assert bundle["joint_model"]["schema"] == "dino-joint-dataset-v1"
    assert bundle["joint_model"]["dataset"] == name
    for target in ("brdu", "edu"):
        assert bundle[f"{target}_feature_scaler"].n_features_in_ == 1024 + 3
        assert bundle[f"{target}_model"].n_features_in_ == 1024 + 3
    derived = derive_bundle(bundle, {}, where="deployed bundle")
    assert derived["schema"] == "dino-knn-intensity-v2"
    assert derived["marker_contract"] == DEFAULT_CONTRACT.to_json()


@pytest.mark.parametrize("name", DATASETS)
def test_legacy_embedding_bundle_is_a_v6_model_bundle(legacy_root, name) -> None:
    bundle = joblib.load(legacy_root / "inputs" / name / "embedding-model.joblib")

    assert ZARR_MODEL.recognizes(bundle)
    assert bundle_contract(bundle, where="embedding bundle") == DEFAULT_CONTRACT


def _assert_same_estimator(left, right) -> None:
    assert type(left) is type(right)
    for attribute in ("mean_", "scale_", "var_", "coef_", "intercept_", "classes_"):
        if hasattr(right, attribute):
            np.testing.assert_array_equal(getattr(left, attribute), getattr(right, attribute))


def _assert_same_values(new: dict, frozen: dict, *, skip: set[str]) -> None:
    assert set(new) == set(frozen) | {"marker_contract"}
    assert new["marker_contract"] == DEFAULT_CONTRACT.to_json()
    for key in set(frozen) - skip:
        if key.endswith(("_model", "_feature_scaler")) and not isinstance(frozen[key], dict):
            _assert_same_estimator(new[key], frozen[key])
        else:
            assert new[key] == frozen[key], key


def test_refit_from_the_legacy_inputs_reproduces_the_frozen_fit(legacy_root, tmp_path) -> None:
    output = tmp_path / "refit"

    fit(legacy_root / "config.json", output, workers=1)

    frozen = legacy_root / "output"
    assert (output / "labels.csv").read_bytes() == (frozen / "labels.csv").read_bytes()
    pd.testing.assert_frame_equal(
        pd.read_parquet(output / "labeled_predictions.parquet"), pd.read_parquet(frozen / "labeled_predictions.parquet")
    )
    new_joint, frozen_joint = (joblib.load(path / "model.joblib") for path in (output, frozen))
    assert new_joint["schema"] == "dino-joint-dataset-v2"
    _assert_same_values(new_joint, frozen_joint, skip={"schema", "created_at_utc", "validation_split"})
    for name in DATASETS:
        new, old = (joblib.load(path / "deploy" / name / "model.joblib") for path in (output, frozen))
        assert new["schema"] == "dino-knn-intensity-v2"
        assert new["joint_model"]["schema"] == "dino-joint-dataset-v2"
        assert (output / "deploy" / name / "labels.csv").read_bytes() == (
            frozen / "deploy" / name / "labels.csv"
        ).read_bytes()
        _assert_same_values(new, old, skip={"schema", "created_at_utc", "joint_model"})


@pytest.mark.parametrize("name", DATASETS)
def test_legacy_deployed_bundle_scores_the_frozen_joint_predictions(legacy_root, name) -> None:
    bundle = joblib.load(legacy_root / "output" / "deploy" / name / "model.joblib")
    frozen = pd.read_parquet(legacy_root / "output" / "labeled_predictions.parquet")
    members = frozen[frozen["dataset"] == name]
    artifact = read_embedding_artifact(
        legacy_root / "inputs" / name / "embeddings" / "shard-000" / "embeddings" / "part-000000.pt"
    )
    embeddings = {marker: artifact[marker].float().numpy() for marker in DEFAULT_CONTRACT.markers}
    context = load_classifier_context(bundle).features_for_labels(
        members["label"].to_numpy(), neighbors=bundle["intensity_neighbors"]
    )
    for marker in DEFAULT_CONTRACT.markers:
        features = assemble_classifier_features(
            embeddings, context, target=marker, mode="marker", markers=DEFAULT_CONTRACT.markers
        )
        probability = bundle[f"{marker}_model"].predict_proba(bundle[f"{marker}_feature_scaler"].transform(features))
        np.testing.assert_allclose(probability[:, 1], members[f"p_{marker}"].to_numpy(), rtol=0, atol=1e-10)
