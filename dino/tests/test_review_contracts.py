import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dino_markers import DEFAULT_CONTRACT, ContractMismatchError, MarkerConfigError, MarkerContract

from dino.review_contracts import combined_uncertainty, load_review_predictions


def test_uncertainty_uses_saved_cutoffs_and_keeps_exact_cutoff_infinite():
    result = combined_uncertainty(
        {"brdu": np.array([.25, .75, 1.]), "edu": np.array([.5, 1., .75])},
        {"brdu": .25, "edu": .75},
    )
    np.testing.assert_array_equal(result, [np.inf, 4., np.inf])
    assert result.dtype == np.float64


def test_uncertainty_takes_the_closest_margin_over_three_markers():
    result = combined_uncertainty(
        {"edu": [.9, .5], "brdu": [.2, .5], "pcna": [.45, .1]},
        {"edu": .5, "brdu": .5, "pcna": .5},
    )
    np.testing.assert_allclose(result, [1 / .05, np.inf])


def test_uncertainty_for_one_marker_is_its_inverse_margin():
    np.testing.assert_allclose(combined_uncertainty({"edu": [.25]}, {"edu": .5}), [4.])


@pytest.mark.parametrize("probabilities,cutoffs", [
    ({"brdu": [.2], "edu": [.2]}, {"brdu": .5}),
    ({"brdu": [.2]}, {"brdu": .5, "edu": .5}),
    ({"brdu": [.2]}, {"edu": .5}),
    ({}, {}),
])
def test_uncertainty_requires_the_same_nonempty_marker_keys(probabilities, cutoffs):
    with pytest.raises(ValueError, match="same nonempty markers"):
        combined_uncertainty(probabilities, cutoffs)


@pytest.mark.parametrize("brdu,edu,cutoff", [
    ([], [], .5), ([.2], [.2, .4], .5), ([np.nan], [.2], .5),
    ([-.1], [.2], .5), ([.2], [.2], 0.), ([.2], [.2], 1.), ([.2], [.2], np.inf), ([.2], [.2], np.nan),
])
def test_uncertainty_rejects_invalid_probabilities_or_cutoffs(brdu, edu, cutoff):
    with pytest.raises(ValueError):
        combined_uncertainty({"brdu": np.asarray(brdu), "edu": np.asarray(edu)}, {"brdu": cutoff, "edu": .5})


def test_uncertainty_rejects_two_dimensional_probabilities():
    with pytest.raises(ValueError, match="matching vectors"):
        combined_uncertainty({"brdu": np.full((2, 2), .2)}, {"brdu": .5})


@pytest.fixture
def review(tmp_path):
    rows = pd.DataFrame({
        "cell": ["run|1", "run|2"], "label": [1, 2], "tranche": ["t", "t"],
        "split": ["training", "validation"],
        "p_brdu": [.9, .1], "pred_brdu": [1, 0], "p_edu": [.2, .8], "pred_edu": [0, 1],
    })
    predictions = tmp_path / "predictions.parquet"
    rows.to_parquet(predictions, index=False)
    manifest = {
        "artifact": "dino-labeled-review-v1", "model_sha256": "m", "predictions": str(predictions),
        "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
        "cells": 2, "training_cells": 1, "validation_cells": 1, "brdu_cutoff": .5, "edu_cutoff": .5,
    }
    path = tmp_path / "complete.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def _load(path: Path, contract: MarkerContract = DEFAULT_CONTRACT):
    return load_review_predictions(path, model_sha256="m", cell_prefix="run", contract=contract)


def test_review_predictions_load_under_the_default_contract(review):
    path, _ = review
    assert _load(path).cell.tolist() == ["run|1", "run|2"]


@pytest.mark.parametrize(
    "artifact",
    [{}, {"artifact": None}, {"artifact": "dino-labeled-review-v2"}],
    ids=["missing", "null", "unknown"],
)
def test_review_predictions_reject_a_missing_null_or_unknown_version(review, artifact):
    path, manifest = review
    path.write_text(json.dumps({key: value for key, value in manifest.items() if key != "artifact"} | artifact))
    with pytest.raises(ValueError, match="differs from the selected classifier"):
        _load(path)


def test_review_predictions_reject_a_manifest_that_carries_a_contract(review):
    path, manifest = review
    path.write_text(json.dumps({**manifest, "marker_contract": DEFAULT_CONTRACT.to_json()}))
    with pytest.raises(MarkerConfigError, match="must not carry marker_contract"):
        _load(path)


@pytest.mark.parametrize("contract", [
    MarkerContract("pi", ("edu", "brdu")),
    MarkerContract("dapi", ("brdu", "edu")),
    MarkerContract("pi", ("brdu", "edu", "pcna")),
])
def test_review_predictions_reject_a_non_default_requested_contract(review, contract):
    path, _ = review
    with pytest.raises(ContractMismatchError, match="marker order is part of the contract"):
        _load(path, contract)
