"""Validate selected review artifacts and compute classifier decision margins."""
from __future__ import annotations

import json
from collections.abc import Mapping
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dino_markers import (
    CONTRACT_FIELD,
    DEFAULT_CONTRACT,
    MarkerConfigError,
    MarkerContract,
    cutoff_key,
    prediction_column,
    probability_column,
    require_same_contract,
)
from numpy.typing import ArrayLike

from dino.artifacts import file_sha256


def combined_uncertainty(probabilities: Mapping[str, ArrayLike], cutoffs: Mapping[str, float]) -> np.ndarray:
    """Return the maximum inverse decision margin across the named classifiers."""
    if not probabilities or set(probabilities) != set(cutoffs):
        raise ValueError("Classifier probabilities and thresholds must name the same nonempty markers")
    vectors = {marker: np.asarray(values, dtype=np.float64) for marker, values in probabilities.items()}
    shapes = {vector.shape for vector in vectors.values()}
    if any(vector.ndim != 1 for vector in vectors.values()) or len(shapes) != 1 or shapes == {(0,)}:
        raise ValueError("Classifier probabilities must be nonempty matching vectors")
    if not all(np.isfinite(vector).all() for vector in vectors.values()):
        raise ValueError("Classifier probabilities must be finite")
    if any(np.any((vector < 0) | (vector > 1)) for vector in vectors.values()):
        raise ValueError("Classifier probabilities must lie in [0, 1]")
    if not all(np.isfinite(cutoffs[marker]) and 0 < cutoffs[marker] < 1 for marker in vectors):
        raise ValueError("Classifier thresholds must lie strictly within (0, 1)")
    with np.errstate(divide="ignore"):
        return 1.0 / reduce(np.minimum, (np.abs(vectors[marker] - cutoffs[marker]) for marker in vectors))


def load_prediction_pointer(
    pointer_path: Path,
) -> tuple[str, Path, Path, dict[str, object]]:
    """Resolve the audited model selected for the production review app."""
    pointer = json.loads(pointer_path.read_text())
    if pointer.get("artifact") != "dino-review-current-v1":
        raise ValueError(f"Invalid review-model pointer: {pointer_path}")
    model_dir = Path(str(pointer["model_dir"]))
    prediction_dir = Path(str(pointer["prediction_dir"]))
    if prediction_dir.resolve() != (model_dir / "cortex-pred/predictions").resolve():
        raise ValueError(
            "Review-model pointer has an inconsistent prediction directory"
        )
    manifest_path = prediction_dir.parent / "complete.json"
    contract = json.loads(manifest_path.read_text())
    if pointer.get("model_sha256") != contract.get("model_sha256") or int(
        pointer["cells"]
    ) != int(contract["cells"]):
        raise ValueError("Review-model pointer differs from its inference manifest")
    if "review_manifest" in pointer:
        review_path = Path(str(pointer["review_manifest"]))
        if review_path.resolve() != (model_dir / "labeled-review/complete.json").resolve():
            raise ValueError("Review pointer has an inconsistent labeled review directory")
        if file_sha256(review_path) != pointer["review_manifest_sha256"]:
            raise ValueError("Review pointer differs from its labeled review manifest")
        review_contract = json.loads(review_path.read_text())
        if review_contract["model_sha256"] != contract["model_sha256"]:
            raise ValueError("Labeled review belongs to a different classifier")
        contract["review_manifest"] = str(review_path)
    return model_dir.name, prediction_dir, manifest_path, contract


def load_review_predictions(
    manifest_path: Path, *, model_sha256: str, cell_prefix: str, contract: MarkerContract
) -> pd.DataFrame:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('artifact') != 'dino-labeled-review-v1' or manifest['model_sha256'] != model_sha256:
        raise ValueError('Labeled review artifact differs from the selected classifier')
    if CONTRACT_FIELD in manifest:
        raise MarkerConfigError(f"{manifest_path}: dino-labeled-review-v1 records must not carry {CONTRACT_FIELD}")
    require_same_contract(
        contract, DEFAULT_CONTRACT, expected_from="the review target", actual_from=str(manifest_path)
    )
    predictions = Path(manifest['predictions'])
    if predictions.parent.resolve() != manifest_path.parent.resolve():
        raise ValueError('Labeled review predictions are outside their artifact directory')
    if file_sha256(predictions) != manifest['predictions_sha256']:
        raise ValueError('Labeled review predictions differ from their recorded identity')
    frame = pq.read_table(predictions).to_pandas()
    required = {'cell', 'label', 'tranche', 'split'}
    for marker in contract.markers:
        required |= {probability_column(marker), prediction_column(marker)}
    if (required - set(frame.columns) or len(frame) != manifest['cells']
            or frame.cell.duplicated().any()):
        raise ValueError('Labeled review predictions have incomplete or duplicate identities')
    if (not frame.cell.equals(cell_prefix + '|' + frame.label.astype(str))
            or not frame.split.isin(['training', 'validation']).all()
            or int((frame.split == 'training').sum()) != manifest['training_cells']
            or int((frame.split == 'validation').sum()) != manifest['validation_cells']):
        raise ValueError('Labeled review identities or actual split counts differ from their manifest')
    for marker in contract.markers:
        p = frame[probability_column(marker)]
        cutoff = manifest[cutoff_key(marker)]
        called = frame[prediction_column(marker)]
        if (not np.isfinite(cutoff) or not 0 < cutoff < 1 or not np.isfinite(p).all()
                or not p.between(0, 1).all()
                or not called.isin([0, 1]).all()
                or not np.array_equal(called, p >= cutoff)):
            raise ValueError('Labeled review probabilities or cutoff calls are invalid')
    return frame
