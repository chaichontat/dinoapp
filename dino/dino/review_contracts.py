"""Validate selected review artifacts and compute classifier decision margins."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from dino.artifacts import file_sha256


def combined_uncertainty(
    p_brdu: np.ndarray,
    p_edu: np.ndarray,
    *,
    brdu_threshold: float,
    edu_threshold: float,
) -> np.ndarray:
    """Return the maximum inverse decision margin across the two classifiers."""
    p_brdu = np.asarray(p_brdu, dtype=np.float64)
    p_edu = np.asarray(p_edu, dtype=np.float64)
    if p_brdu.ndim != 1 or p_edu.shape != p_brdu.shape or not len(p_brdu):
        raise ValueError("Classifier probabilities must be nonempty matching vectors")
    if not np.isfinite(p_brdu).all() or not np.isfinite(p_edu).all():
        raise ValueError("Classifier probabilities must be finite")
    if np.any((p_brdu < 0) | (p_brdu > 1) | (p_edu < 0) | (p_edu > 1)):
        raise ValueError("Classifier probabilities must lie in [0, 1]")
    if not (
        np.isfinite(brdu_threshold)
        and np.isfinite(edu_threshold)
        and 0 < brdu_threshold < 1
        and 0 < edu_threshold < 1
    ):
        raise ValueError("Classifier thresholds must lie strictly within (0, 1)")
    margin = np.minimum(np.abs(p_brdu - brdu_threshold), np.abs(p_edu - edu_threshold))
    with np.errstate(divide="ignore"):
        return 1.0 / margin


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


def load_review_predictions(manifest_path: Path, *, model_sha256: str, cell_prefix: str) -> pd.DataFrame:
    manifest = json.loads(manifest_path.read_text())
    if manifest['artifact'] != 'dino-labeled-review-v1' or manifest['model_sha256'] != model_sha256:
        raise ValueError('Labeled review artifact differs from the selected classifier')
    predictions = Path(manifest['predictions'])
    if predictions.parent.resolve() != manifest_path.parent.resolve():
        raise ValueError('Labeled review predictions are outside their artifact directory')
    if file_sha256(predictions) != manifest['predictions_sha256']:
        raise ValueError('Labeled review predictions differ from their recorded identity')
    frame = pq.read_table(predictions).to_pandas()
    required = {'cell', 'label', 'tranche', 'split', 'p_brdu', 'pred_brdu', 'p_edu', 'pred_edu'}
    if (required - set(frame.columns) or len(frame) != manifest['cells']
            or frame.cell.duplicated().any()):
        raise ValueError('Labeled review predictions have incomplete or duplicate identities')
    if (not frame.cell.equals(cell_prefix + '|' + frame.label.astype(str))
            or not frame.split.isin(['training', 'validation']).all()
            or int((frame.split == 'training').sum()) != manifest['training_cells']
            or int((frame.split == 'validation').sum()) != manifest['validation_cells']):
        raise ValueError('Labeled review identities or actual split counts differ from their manifest')
    for target in ('brdu', 'edu'):
        p = frame[f'p_{target}']
        cutoff = manifest[f'{target}_cutoff']
        if (not np.isfinite(cutoff) or not 0 < cutoff < 1 or not np.isfinite(p).all()
                or not p.between(0, 1).all()
                or not frame[f'pred_{target}'].isin([0, 1]).all()
                or not np.array_equal(frame[f'pred_{target}'], p >= cutoff)):
            raise ValueError('Labeled review probabilities or cutoff calls are invalid')
    return frame
