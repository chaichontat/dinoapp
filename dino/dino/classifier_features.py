"""Assemble persisted classifier features and load their declared intensity context."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pyarrow.parquet as pq

from dino.artifacts import file_sha256
from dino.intensity_context import MODEL_SCHEMA, IntensityContext


TARGETS = ("brdu", "edu")


BASE_CONTEXT_FEATURES = ("log1p_cell_mean", "log_cell_over_local_mean")


DAPI_CONTEXT_FEATURES = (*BASE_CONTEXT_FEATURES, "log1p_dapi_mean")


# Background reference for the marker/DAPI ratio: the median and MAD over this many nearest
# cells tolerate the labeled minority of a neighborhood.
RATIO_NEIGHBORS = 100


DAPI_RATIO_CONTEXT_FEATURES = (
    *DAPI_CONTEXT_FEATURES,
    f"robust_z_log_marker_over_dapi_k{RATIO_NEIGHBORS}",
)


class Context(Protocol):
    points_zyx_um: np.ndarray

    def features_for_labels(
        self, labels: np.ndarray, *, neighbors: int
    ) -> dict[str, np.ndarray]: ...


def _load_ordered_dapi_means(dapi_props: Path, marker_props: Path) -> np.ndarray:
    """Load DAPI means after proving they use the marker table's cell identities."""
    dapi = pq.read_table(dapi_props, columns=["label", "area", "intensity_dapi_mean"])
    marker = pq.read_table(marker_props, columns=["label", "area"])
    if dapi.num_rows != marker.num_rows:
        raise ValueError(
            f"DAPI and marker row counts differ: {dapi.num_rows} versus {marker.num_rows}"
        )
    if any(dapi[column].null_count for column in dapi.column_names):
        raise ValueError("DAPI regionprops contain null values")
    for column in ("label", "area"):
        left = dapi[column].combine_chunks().to_numpy(zero_copy_only=False)
        right = marker[column].combine_chunks().to_numpy(zero_copy_only=False)
        if not np.array_equal(left, right):
            raise ValueError(f"DAPI and marker regionprops differ in {column}")
    labels = dapi["label"].combine_chunks().to_numpy(zero_copy_only=False)
    expected = np.arange(1, len(labels) + 1, dtype=labels.dtype)
    if not np.array_equal(labels, expected):
        raise ValueError("DAPI labels must be contiguous and ordered from one")
    means = np.asarray(
        dapi["intensity_dapi_mean"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    if np.any(means < 0) or not np.isfinite(means).all():
        raise ValueError("DAPI means must be finite and nonnegative")
    return means


@dataclass(frozen=True)
class DapiIntensityContext:
    """Add one shared DAPI scalar to each target's marker-specific context."""

    marker: IntensityContext
    dapi_means: np.ndarray

    @property
    def points_zyx_um(self) -> np.ndarray:
        return self.marker.points_zyx_um

    @classmethod
    def load(
        cls,
        marker_props: Path,
        dapi_props: Path,
        *,
        scale_zyx_um: np.ndarray,
    ) -> DapiIntensityContext:
        marker = IntensityContext.load(marker_props, scale_zyx_um=scale_zyx_um)
        dapi_means = _load_ordered_dapi_means(dapi_props, marker_props)
        if len(dapi_means) != len(marker.points_zyx_um):
            raise ValueError("DAPI and marker context cell counts differ")
        return cls(marker=marker, dapi_means=dapi_means)

    def features_for_labels(
        self, labels: np.ndarray, *, neighbors: int
    ) -> dict[str, np.ndarray]:
        marker_features = self.marker.features_for_labels(labels, neighbors=neighbors)
        indices = np.asarray(labels, dtype=np.int64) - 1
        dapi = np.log1p(self.dapi_means[indices])[:, None]
        return {
            target: np.column_stack((marker_features[target], dapi))
            for target in TARGETS
        }


@dataclass(frozen=True)
class DapiRatioIntensityContext:
    """Append each marker's robust local z-score of log(marker/DAPI) to the DAPI context."""

    dapi: DapiIntensityContext

    @property
    def points_zyx_um(self) -> np.ndarray:
        return self.dapi.points_zyx_um

    def features_for_labels(
        self, labels: np.ndarray, *, neighbors: int
    ) -> dict[str, np.ndarray]:
        base = self.dapi.features_for_labels(labels, neighbors=neighbors)
        indices = np.asarray(labels, dtype=np.int64) - 1
        local = self.dapi.marker.neighbor_indices_for_labels(labels, neighbors=RATIO_NEIGHBORS)
        dapi = self.dapi.dapi_means
        features = {}
        for target in TARGETS:
            means = self.dapi.marker.means[target]
            ratio = np.log((means[indices] + 1.0) / (dapi[indices] + 1.0))
            around = np.log((means[local] + 1.0) / (dapi[local] + 1.0))
            center = np.median(around, axis=1)
            spread = 1.4826 * np.median(np.abs(around - center[:, None]), axis=1)
            if np.any(spread <= 0):
                raise ValueError(f"Local {target}/DAPI ratio has zero spread around some cells")
            features[target] = np.column_stack((base[target], (ratio - center) / spread))
        return features


def load_context(
    features: tuple[str, ...],
    marker_props: Path,
    dapi_props: Path,
    *,
    scale_zyx_um: np.ndarray,
) -> DapiIntensityContext | DapiRatioIntensityContext:
    """Load the DAPI-based context that produces exactly the named features."""
    context = DapiIntensityContext.load(marker_props, dapi_props, scale_zyx_um=scale_zyx_um)
    if features == DAPI_CONTEXT_FEATURES:
        return context
    if features == DAPI_RATIO_CONTEXT_FEATURES:
        return DapiRatioIntensityContext(context)
    raise ValueError(f"Unsupported DAPI context features: {features}")


def load_classifier_context(bundle: dict[str, object]) -> Context:
    """Load exactly the scalar-feature context declared by a model bundle."""
    features = tuple(str(value) for value in bundle.get("context_features", ()))
    marker_props = Path(str(bundle["intensity_props"]))
    expected_marker_hash = bundle.get("intensity_props_sha256")
    if not isinstance(expected_marker_hash, str) or (
        file_sha256(marker_props) != expected_marker_hash
    ):
        raise ValueError(f"Marker regionprops SHA-256 differs: {marker_props}")
    scale = np.asarray(bundle["intensity_scale_zyx_um"], dtype=np.float64)
    if features == BASE_CONTEXT_FEATURES:
        return IntensityContext.load(marker_props, scale_zyx_um=scale)
    if features not in (DAPI_CONTEXT_FEATURES, DAPI_RATIO_CONTEXT_FEATURES):
        raise ValueError(f"Unsupported classifier context features: {features}")
    dapi_props = Path(str(bundle["dapi_props"]))
    expected_hash = bundle.get("dapi_props_sha256")
    if not isinstance(expected_hash, str) or file_sha256(dapi_props) != expected_hash:
        raise ValueError(f"DAPI regionprops SHA-256 differs: {dapi_props}")
    return load_context(features, marker_props, dapi_props, scale_zyx_um=scale)


def assemble_classifier_features(
    embeddings: dict[str, np.ndarray],
    context_features: dict[str, np.ndarray],
    *,
    target: str,
    mode: str,
) -> np.ndarray:
    """Assemble classifier columns in the persisted marker order."""
    if target not in TARGETS:
        raise ValueError(f"Unknown classifier target: {target}")
    if mode == "marker":
        arrays = (embeddings[target], context_features[target])
    elif mode == "joint":
        arrays = tuple(embeddings[name] for name in TARGETS) + tuple(
            context_features[name] for name in TARGETS
        )
    else:
        raise ValueError(f"Unknown classifier feature mode: {mode}")
    if any(values.ndim != 2 for values in arrays):
        raise ValueError("Classifier feature arrays must be two-dimensional")
    if len({len(values) for values in arrays}) != 1:
        raise ValueError("Classifier feature arrays must have the same row count")
    return np.column_stack(arrays)


FEATURE_KEYS = (
    "feature_extractor",
    "weights_sha256",
    "model_source_sha256",
    "crop_size",
    "resize_size",
    "crop_eval_size",
    "highpass_sigma_px",
    "mask_erosion_px",
    "input_channels",
    "input_channel_names_by_target",
    "feature_precision",
    "feature_execution",
    "pixel_correction",
    "spatial_subtraction",
)
FEATURE_MODES = ("marker", "joint")
EMBEDDING_WIDTH = 1024


def validate_feature_contract(bundle: dict[str, object], embedding_bundle: dict[str, object]) -> None:
    """Require the classifier to consume the persisted embeddings' exact feature contract."""
    if bundle.get("schema") != MODEL_SCHEMA or embedding_bundle.get("schema") != MODEL_SCHEMA:
        raise ValueError(f"Expected model and embedding schema {MODEL_SCHEMA}")
    mismatched = [key for key in FEATURE_KEYS if bundle.get(key) != embedding_bundle.get(key)]
    if mismatched:
        raise ValueError(f"Persisted embeddings use a different feature contract: {mismatched}")
    if bundle.get("embedding_pca") is not None:
        raise ValueError("PCA-projected classifiers are not supported by this scorer")
    mode = str(bundle.get("classifier_feature_mode", "marker"))
    if mode not in FEATURE_MODES:
        raise ValueError(f"Unknown classifier feature mode: {mode}")
    context_features = tuple(str(value) for value in bundle.get("context_features", ()))
    if context_features not in (BASE_CONTEXT_FEATURES, DAPI_CONTEXT_FEATURES, DAPI_RATIO_CONTEXT_FEATURES):
        raise ValueError(f"Unsupported classifier context features: {context_features}")
    if context_features != BASE_CONTEXT_FEATURES and not all(
        isinstance(bundle.get(key), str) for key in ("dapi_props", "dapi_props_sha256")
    ):
        raise ValueError("DAPI classifier context is missing its artifact contract")
    expected = (
        EMBEDDING_WIDTH + len(context_features)
        if mode == "marker"
        else len(TARGETS) * (EMBEDDING_WIDTH + len(context_features))
    )
    actual = {target: int(bundle[f"{target}_feature_scaler"].n_features_in_) for target in TARGETS}
    if set(actual.values()) != {expected}:
        raise ValueError(f"Classifier feature dimensions differ from the saved contract: {actual} versus {expected}")
