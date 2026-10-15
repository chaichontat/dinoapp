"""Native ventricular fields and edits to the existing barrage annotations."""

from __future__ import annotations

import base64
import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock

import igl
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import zarr
from dino_markers import (
    CONTRACT_FIELD,
    CORTEX_RESCORE,
    DEFAULT_CONTRACT,
    INFERENCE,
    Label,
    MarkerConfig,
    MarkerContract,
    SavedFilter,
    UnsupportedArtifactError,
    call_weights,
    cutoff_key,
    decode_call,
    default_config,
    encode_call,
    identity_color,
    matches_saved,
    miscall_column,
    prediction_column,
    probability_column,
    require_same_contract,
    resolve_directory,
    saved_column,
    uncalled_code,
)
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from skimage.segmentation import find_boundaries

from dinoapp.sources import AnnotationFiles as AnnotationFiles
from dinoapp.native_geometry import image_geometry, vertex_normals
from dino.zarr_source import CellCrop, CellCropReader, metadata_digest
from dino.review_contracts import (
    combined_uncertainty,
    load_prediction_pointer as load_prediction_pointer,
)
from dinoapp.review_predictions import load_review_predictions
from squisher.jpegxr_zarr import register_jpegxr_codec

ROOT = Path("/working/eduseg/20260808-prod2/full-run2")
REVIEW_ROOT = ROOT


@dataclass(frozen=True)
class ModelCalls:
    calls: np.ndarray
    uncertainty: np.ndarray
    uncertainty_log_vmax: float


def load_model_calls(
    frame: pd.DataFrame,
    *,
    dataset: str,
    contract: MarkerContract,
    cutoffs: Mapping[str, float],
    cell_prefix: str,
    uncertainty_log_vmax: float | None = None,
) -> ModelCalls:
    """Validate saved model predictions and encode them as call codes and uncertainty by label."""
    markers = contract.markers
    required = {"label", "cell", *markers, *(probability_column(marker) for marker in markers)}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{dataset} predictions are missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{dataset} predictions are empty")
    labels = frame["label"]
    if labels.duplicated().any() or (labels <= 0).any():
        raise ValueError(f"{dataset} prediction labels must be unique and positive")
    if not frame["cell"].equals(cell_prefix + "|" + labels.astype(str)):
        raise ValueError(f"{dataset} prediction cell identities do not match {cell_prefix} segmentation labels")
    calls = frame[list(markers)]
    if calls.isna().any().any() or not calls.isin([0, 1]).all().all():
        raise ValueError(f"{dataset} prediction calls must be complete and binary")
    probabilities = {marker: frame[probability_column(marker)].to_numpy(dtype=np.float64) for marker in markers}
    if not all(np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all() for p in probabilities.values()):
        raise ValueError(f"{dataset} prediction probabilities must be finite and within [0, 1]")

    codes = np.full(int(labels.max()) + 1, uncalled_code(contract), dtype=np.uint8)
    codes[labels.to_numpy()] = calls.to_numpy(dtype=np.uint8) @ np.asarray(call_weights(contract), dtype=np.uint8)
    scores = combined_uncertainty(probabilities, cutoffs)
    uncertainty = np.full(len(codes), np.nan, dtype=np.float32)
    uncertainty[labels.to_numpy()] = scores
    if uncertainty_log_vmax is None:
        finite = np.log1p(scores[np.isfinite(scores)])
        if not len(finite):
            raise ValueError(f"{dataset}: No finite model uncertainty scores are available")
        uncertainty_log_vmax = float(np.quantile(finite, 0.999))
    return ModelCalls(codes, uncertainty, uncertainty_log_vmax)


def prediction_contract(manifest: Mapping[str, object], *, where: str) -> MarkerContract:
    """Read the marker contract of an inference or rescore manifest; an unversioned one is the default."""
    if "artifact" not in manifest:
        if CONTRACT_FIELD in manifest:
            raise UnsupportedArtifactError(f"{where}: unversioned prediction manifest carries {CONTRACT_FIELD}")
        return DEFAULT_CONTRACT
    for kind in (INFERENCE, CORTEX_RESCORE):
        if kind.recognizes(manifest):
            return kind.contract(manifest, where=where)
    raise UnsupportedArtifactError(f"{where}: unsupported prediction manifest artifact {manifest['artifact']!r}")


def classifier_caption(model_id: str, config: MarkerConfig, cutoffs: Mapping[str, float]) -> str:
    cuts = " · ".join(f"{config.label(marker)} cutoff {cutoffs[marker]:.10f}" for marker in config.contract.markers)
    return f"Classifier {model_id} · {cuts}"


def code_colors(config: MarkerConfig, codes: np.ndarray) -> np.ndarray:
    """Return the identity color of each call code, computed once per distinct code."""
    codes = np.asarray(codes)
    colors = np.empty(codes.shape + (3,), dtype=np.float64)
    for code in np.unique(codes):
        colors[codes == code] = identity_color(config, decode_call(config.contract, int(code)))
    return colors


PALETTE = code_colors(default_config(), np.arange(5, dtype=np.uint8))


def adapter_config(annotation_directory: Path, raw_channels: Mapping[str, str], *, adapter: str) -> MarkerConfig:
    """Resolve the marker configuration an adapter's native channels can serve."""
    config = resolve_directory(annotation_directory)
    if set(raw_channels.values()) != set(config.contract.channel_keys):
        raise ValueError(
            f"{adapter} native channels {sorted(raw_channels.values())} do not match the marker "
            f"configuration of {annotation_directory}: {config.contract}"
        )
    return config


def sample_center(segments: np.ndarray, *, seed: int, position: int) -> np.ndarray:
    """Sample uniformly by physical arc length, reproducibly for seed and position."""
    if seed < 0 or position < 0:
        raise ValueError("Seed and position must be nonnegative")
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    if not len(lengths) or not np.isfinite(lengths).all() or lengths.sum() <= 0:
        raise ValueError("No finite boundary length is available for this Z level")
    rng = np.random.default_rng(np.random.SeedSequence([seed, position]))
    segment = segments[rng.choice(len(segments), p=lengths / lengths.sum())]
    return segment[0] + rng.random() * (segment[1] - segment[0])


def pick_cell(mask: np.ndarray, x: float, y: float) -> int | None:
    """Map normalized image coordinates to the exact native segmentation pixel."""
    if not (np.isfinite(x) and np.isfinite(y) and 0 <= x < 1 and 0 <= y < 1):
        return None
    label = int(mask[int(y * mask.shape[0]), int(x * mask.shape[1])])
    return label if label else None


def training_miscalls(
    predictions: pd.DataFrame,
    live_labels: Mapping[str, Label],
    *,
    contract: MarkerContract,
    split: str,
    target: str | None,
    identity: SavedFilter,
) -> pd.DataFrame:
    """Rank hard model disagreements within one current saved identity."""
    required = {"cell", "label", "tranche", "split"}
    for marker in contract.markers:
        required |= {probability_column(marker), prediction_column(marker)}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Labeled predictions are missing columns: {sorted(missing)}")
    if predictions["cell"].duplicated().any():
        raise ValueError("Labeled prediction cells must be unique")
    if split not in {"training", "validation", "all"}:
        raise ValueError(f"Unexpected split: {split}")
    if target is not None and target not in contract.markers:
        raise ValueError(f"Unexpected target: {target}")

    rows = training_gallery(predictions, live_labels, contract=contract, split=split, identity=identity)
    scores = {}
    for marker in contract.markers:
        saved = rows[saved_column(marker)].to_numpy()
        probability = rows[probability_column(marker)].to_numpy()
        rows[miscall_column(marker)] = rows[prediction_column(marker)].astype(bool) != rows[
            saved_column(marker)
        ].astype(bool)
        scores[marker] = np.where(saved == 1, 1 - probability, probability)
    disagreements = []
    for marker in (target,) if target is not None else contract.markers:
        keep = rows[miscall_column(marker)].to_numpy()
        marker_rows = rows.loc[keep].copy()
        marker_rows["audit_score"] = scores[marker][keep]
        marker_rows["audit_marker"] = marker
        disagreements.append(marker_rows)
    rows = pd.concat(disagreements, ignore_index=True)
    return rows.sort_values(
        ["audit_score", "cell", "audit_marker"], ascending=[False, True, True]
    ).reset_index(drop=True)


def training_gallery(
    predictions: pd.DataFrame,
    live_labels: Mapping[str, Label],
    *,
    contract: MarkerContract,
    split: str,
    identity: SavedFilter,
) -> pd.DataFrame:
    """Select labeled cells whose current saved call matches the saved-state filter."""
    required = {"cell", "tranche", "split"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Labeled predictions are missing columns: {sorted(missing)}")
    if predictions["cell"].duplicated().any():
        raise ValueError("Labeled prediction cells must be unique")
    if split not in {"training", "validation", "all"}:
        raise ValueError(f"Unexpected split: {split}")
    if (
        not isinstance(identity, tuple)
        or len(identity) != len(contract.markers)
        or any(value not in (0, 1, None) for value in identity)
    ):
        raise ValueError(
            f"Gallery identity must hold one 0, 1 or None entry per marker {list(contract.markers)}"
        )

    rows = predictions[predictions["cell"].isin(live_labels)].copy()
    if split != "all":
        rows = rows[rows["split"] == split].copy()
    for index, marker in enumerate(contract.markers):
        rows[saved_column(marker)] = [live_labels[cell][index] for cell in rows["cell"]]
    keep = np.array([matches_saved(live_labels[cell], identity) for cell in rows["cell"]], dtype=bool)
    return rows[keep].sort_values("cell").reset_index(drop=True)


def render_marker_crop(crop: CellCrop, marker: str, display_max: float, *, label: str) -> bytes:
    """Render one native marker crop with established contrast and cell outlines."""
    from PIL import Image

    channel = crop.channels[marker]
    if channel.shape != crop.mask.shape:
        raise ValueError(
            f"{label} channel/mask shape mismatch: {channel.shape} != {crop.mask.shape}"
        )
    if not np.isfinite(display_max) or display_max <= 0:
        raise ValueError(f"{label} display maximum must be finite and positive")
    scaled = np.clip(channel.astype(np.float64) / display_max, 0, 1)
    rgb = colormaps["magma"](scaled)[..., :3]
    boundaries = find_boundaries(crop.labels, mode="inner")
    rgb[boundaries] = 0.65 * rgb[boundaries] + 0.35
    target = find_boundaries(crop.mask, mode="inner")
    rgb[target] = (0, 1, 1)
    output = io.BytesIO()
    Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(output, format="PNG")
    return output.getvalue()


def prefetch_marker_crop(source, cell: str, marker: str) -> None:
    """Populate the marker crop cache for one gallery cell."""
    source.marker_crop(int(cell.rsplit("|", 1)[1]), marker)


def read_cell_geometry(path: Path, label: int, shape: tuple[int, ...]):
    rows = pq.read_table(
        path, columns=["label", "plane_z", "centroid_y", "centroid_x"],
        filters=[("label", "=", label)],
    ).to_pandas()
    if len(rows) != 1:
        raise ValueError(f"Expected one saved crop geometry for cell {label}")
    row = rows.iloc[0]
    coordinates = np.array([row.plane_z, row.centroid_y, row.centroid_x], dtype=float)
    if (not np.isfinite(coordinates).all() or not np.all((coordinates >= 0) & (coordinates < shape))
            or row.plane_z != int(row.plane_z)):
        raise ValueError(f"Invalid saved crop geometry for cell {label}")
    return row


@dataclass
class Field:
    images: dict[str, np.ndarray]
    mask: np.ndarray
    ids: np.ndarray
    inverse: np.ndarray
    borders: np.ndarray
    origin_xy_um: np.ndarray
    scale_xy_um: np.ndarray
    center_xy_um: np.ndarray
    z_um: float | None
    z_index: int
    seed: int | None
    position: int | None
    contours: list[np.ndarray]
    boundary_info: dict[int, dict[str, object]]

    @property
    def token(self) -> str:
        return f"{self.seed}:{self.position}:{self.z_index}:{self.mask.shape}"


class FieldSource:
    """Read fixed-Z native pixels and sample the full supported 90% wall contour."""

    display_name = "Prod2 native"
    dataset_name = "Prod2"
    raw_channels = {"561": "pi", "BrdU": "brdu", "EdU": "edu"}

    @classmethod
    def annotation_directories(cls, root: Path | None = None) -> tuple[Path, ...]:
        root = ROOT if root is None else root
        return tuple(root / name for name in ("barrage-cortex", "barrage-vz-svz", "barrage-pixel"))

    def __init__(self, root: Path = ROOT):
        self.root = root
        self.annotation_paths = tuple(directory / "labels.csv" for directory in self.annotation_directories(root))
        self.default_annotation_path = self.annotation_paths[-1]
        self.config = adapter_config(
            self.default_annotation_path.parent, self.raw_channels, adapter=self.dataset_name
        )
        source_path = root / "dino-v3/source.json"
        source = json.loads(source_path.read_text())
        self.source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        labels_meta = source["labels"]
        if (
            metadata_digest(Path(labels_meta["root"]), labels_meta["array"])
            != labels_meta["metadata_sha256"]
        ):
            raise ValueError(
                "Postprocessed segmentation differs from the saved DINO source"
            )
        native = json.loads((root / "vz-native/native-channels.json").read_text())
        self.default_z = native["planes"][0]["z_um"]
        self.sources = {name: Path(native["sources"][name]["path"]) for name in self.raw_channels}
        register_jpegxr_codec()
        self.shape, self.scale, self.origin = image_geometry(self.sources["561"], "0")
        self.arrays = {}
        for name, path in self.sources.items():
            shape, scale, origin = image_geometry(path, "0")
            if (
                tuple(shape) != tuple(self.shape)
                or not np.array_equal(scale, self.scale)
                or not np.array_equal(origin, self.origin)
            ):
                raise ValueError(f"Native image geometry differs for {name}")
            self.arrays[name] = zarr.open_group(str(path), mode="r")["0"]
        self.labels = zarr.open_array(labels_meta["root"], mode="r")
        if tuple(self.labels.shape) != tuple(self.shape):
            raise ValueError("Postprocessed labels and native microscopy grids differ")
        # One XY crop can overlap four label chunks; retain those plus channel data.
        self.crop_reader = CellCropReader(
            labels=self.labels,
            channels={key: self.arrays[name] for name, key in self.raw_channels.items()},
            offset_zyx=(0, 0, 0),
            cache_bytes=4
            * int(np.prod(self.labels.chunks))
            * self.labels.dtype.itemsize
            + 64 * 1024**2,
        )
        self.crop_lock = Lock()
        mesh = np.load(Path(native["surface"]))
        vertices, faces = mesh["vertices_xyz_um"], mesh["faces"]
        normals = vertex_normals(vertices, faces)
        self.boundaries = []
        for key in ["vz60-s30.npz", "vz90-vz60-s30.npz"]:
            path = root.parent / "full-run/svz-cut/trim" / key
            cut = np.load(path)
            keep = cut["wall_support"] & np.isfinite(cut["cutoff_um"])
            selected = faces[np.all(keep[faces], axis=1)]
            ids, inverse = np.unique(selected, return_inverse=True)
            self.boundaries.append(
                (
                    vertices[ids] - normals[ids] * cut["cutoff_um"][ids, None],
                    inverse.reshape(-1, 3),
                )
            )
        self.assignment_paths = (
            root / "vz-rates/vz30.parquet",
            root / "vz-rates/vz90-nested.parquet",
        )
        for path in self.assignment_paths:
            meta = json.loads(path.with_suffix(".json").read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest() != meta["output_sha256"]:
                raise ValueError(f"Boundary assignment changed: {path}")
        self.model_version, prediction_dir, prediction_manifest, prediction_record = (
            load_prediction_pointer(root / "cell-review/current.json")
        )
        self._require_contract(
            prediction_contract(prediction_record, where=str(prediction_manifest)),
            actual_from=str(prediction_manifest),
        )
        markers = self.config.contract.markers
        columns = ["label", "cell", *markers, *(probability_column(marker) for marker in markers)]
        frame = ds.dataset(prediction_dir, format="parquet").to_table(columns=columns).to_pandas()
        cutoffs = {marker: float(prediction_record[cutoff_key(marker)]) for marker in markers}
        self._adopt_model_calls(frame, cutoffs=cutoffs, cell_prefix="full-run2")
        self.prediction_manifest = str(prediction_manifest)
        self.classifier_caption = classifier_caption(
            str(prediction_record["model_sha256"])[:12], self.config, cutoffs
        )
        self.training_review_manifest = str(prediction_record["review_manifest"])
        self.training_predictions = load_review_predictions(
            Path(self.training_review_manifest), cell_prefix="full-run2",
            model_sha256=str(prediction_record["model_sha256"]), contract=self.config.contract,
        )
        self.contrast = {
            key: tuple(float(limit) for limit in native["contrast_p1_p99_8"][name])
            for name, key in self.raw_channels.items()
        }
        print(
            f"Ready: {len(frame):,} model calls; default Z={self.default_z:.3f} µm",
            flush=True,
        )

    def _require_contract(self, contract: MarkerContract, *, actual_from: str) -> None:
        require_same_contract(
            self.config.contract, contract,
            expected_from=f"the annotations in {self.default_annotation_path.parent}", actual_from=actual_from,
        )

    def _adopt_model_calls(self, frame: pd.DataFrame, *, cutoffs: Mapping[str, float], cell_prefix: str) -> None:
        model = load_model_calls(
            frame, dataset=self.dataset_name, contract=self.config.contract, cutoffs=cutoffs,
            cell_prefix=cell_prefix,
        )
        self.calls, self.uncertainty = model.calls, model.uncertainty
        self.uncertainty_log_vmax = model.uncertainty_log_vmax

    @staticmethod
    def cell_id(label: int) -> str:
        return f"full-run2|{label}"

    def cell_geometry(self, label: int):
        return read_cell_geometry(self.root / "props-cortex.parquet", label, self.shape)

    def field_at_cell(self, label: int, width_um: float) -> Field:
        row = self.cell_geometry(label)
        center = self.origin[:0:-1] + np.array([row.centroid_x, row.centroid_y]) * self.scale[:0:-1]
        return self.field_at(center, int(row.plane_z), width_um)

    def field_at(self, center, zi: int, width_um: float, *, seed=None, position=None) -> Field:
        if not np.isfinite(width_um) or width_um <= 0:
            raise ValueError("Field width must be positive")
        if not 0 <= zi < self.shape[0]:
            raise ValueError("Z is outside the native image")
        center = np.asarray(center, dtype=float)
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError("Field center must contain finite XY coordinates")
        scale_xy, origin_xy = self.scale[:0:-1], self.origin[:0:-1]
        lo = np.floor((center - width_um / 2 - origin_xy) / scale_xy).astype(int)
        hi = np.ceil((center + width_um / 2 - origin_xy) / scale_xy).astype(int)
        shape_xy = np.asarray(self.shape[:0:-1])
        lo, hi = np.maximum(lo, 0), np.minimum(hi, shape_xy)
        if np.any(hi <= lo):
            raise ValueError("Field viewport is outside the native image")
        return self._load_field(zi, lo, hi, center, seed, position)

    def _read_cell_crop(self, label: int):
        """Read the barrage's maximal-area plane and display contract."""
        from dino.crop_rendering import read_crop

        row = self.cell_geometry(label)
        contract = json.loads((self.root / "barrage-pixel/run.json").read_text())
        with self.crop_lock:
            crop = read_crop(self.crop_reader, row, size=contract["size"])
        if not crop.mask.any():
            raise ValueError(
                f"Cell {label} has no mask on saved plane {int(row.plane_z)}"
            )
        return crop, int(row.plane_z), contract

    # Cache selected-cell detail crops.
    @property
    def gallery_predictions(self):
        return self.training_predictions

    @lru_cache(maxsize=32)
    def cell_crop(self, label: int):
        """Use the barrage's maximal-area plane, crop reader, and display contract."""
        from dino.crop_rendering import render_crop

        crop, plane_z, contract = self._read_cell_crop(label)
        output = io.BytesIO()
        render_crop(
            crop, cell=self.cell_id(label), display_max=contract["display_max"], panels=self.config.panels
        ).save(output, format="PNG")
        return output.getvalue(), plane_z

    # Retain one 24-cell visible page plus its 24-cell prefetched successor for each of two markers.
    @lru_cache(maxsize=128)
    def marker_crop(self, label: int, marker: str):
        """Render only one marker for the dense correction gallery."""
        if marker not in self.config.contract.markers:
            raise ValueError(f"Unexpected crop marker: {marker}")
        crop, plane_z, contract = self._read_cell_crop(label)
        image = render_marker_crop(
            crop, marker, contract["display_max"][marker], label=self.config.label(marker)
        )
        return image, plane_z

    @lru_cache(maxsize=8)
    def contours_at(self, z_index: int) -> list[np.ndarray]:
        z_um = float(self.origin[0] + z_index * self.scale[0])
        result = []
        for vertices, faces in self.boundaries:
            points, edges, _ = igl.isolines(
                vertices, faces, vertices[:, 2], np.array([z_um])
            )
            result.append(points[edges, :2])
        return result

    def field(self, seed: int, position: int, z_um: float, width_um: float) -> Field:
        if not np.isfinite(width_um) or width_um <= 0:
            raise ValueError("Field width must be positive")
        zi = int(round((z_um - self.origin[0]) / self.scale[0]))
        if not 0 <= zi < self.shape[0]:
            raise ValueError("Z is outside the native image")
        contours = self.contours_at(zi)
        scale_xy = self.scale[:0:-1]
        origin_xy = self.origin[:0:-1]
        low = origin_xy + width_um / 2 + scale_xy
        high = (
            origin_xy
            + (np.array(self.shape[:0:-1]) - 1) * scale_xy
            - width_um / 2
            - scale_xy
        )
        segments = contours[1]
        valid = np.all((segments >= low) & (segments <= high), axis=(1, 2))
        center = sample_center(segments[valid], seed=seed, position=position)
        lo = np.floor((center - width_um / 2 - origin_xy) / scale_xy).astype(int)
        hi = np.ceil((center + width_um / 2 - origin_xy) / scale_xy).astype(int)
        return self._load_field(zi, lo, hi, center, seed, position)

    def _load_field(self, zi, lo, hi, center, seed, position) -> Field:
        scale_xy, origin_xy = self.scale[:0:-1], self.origin[:0:-1]
        contours = self.contours_at(zi)
        sl = (zi, slice(lo[1], hi[1]), slice(lo[0], hi[0]))
        images = {self.raw_channels[name]: np.asarray(v[sl]) for name, v in self.arrays.items()}
        mask = np.asarray(self.labels[sl])
        ids, inverse = np.unique(mask, return_inverse=True)
        boundary_info = {}
        wanted = [int(i) for i in ids if i]
        for fraction, path in zip([60, 90], self.assignment_paths, strict=True):
            assignments = pq.read_table(
                path, filters=[("label", "in", wanted)]
            ).to_pylist()
            for row in assignments:
                data = boundary_info.setdefault(row["label"], {"r_vz_um": row["r_vz"]})
                data[f"cutoff_{fraction}_um"] = row["cutoff_um"]
                data[f"included_{fraction}"] = row["near_ventricle"]
        return Field(
            images,
            mask,
            ids,
            inverse.reshape(mask.shape),
            find_boundaries(mask, mode="inner") & (mask > 0),
            origin_xy + lo * scale_xy,
            scale_xy,
            center,
            float(self.origin[0] + zi * self.scale[0]),
            zi,
            seed,
            position,
            contours,
            boundary_info,
        )

    def field_calls(self, field: Field, annotations: AnnotationFiles) -> np.ndarray:
        contract = self.config.contract
        calls = np.full(len(field.ids), uncalled_code(contract), dtype=np.uint8)
        valid = field.ids < len(self.calls)
        calls[valid] = self.calls[field.ids[valid]]
        for i, label in enumerate(field.ids):
            manual = annotations.values.get(self.cell_id(int(label)))
            if manual is not None:
                calls[i] = encode_call(contract, manual)
        return calls

    def field_uncertainty(self, field: Field) -> np.ndarray:
        values = np.full(len(field.ids), np.nan, dtype=np.float32)
        valid = field.ids < len(self.uncertainty)
        values[valid] = self.uncertainty[field.ids[valid]]
        return values


def render_field(
    field: Field,
    calls: np.ndarray,
    *,
    config: MarkerConfig,
    mode: str,
    selected: int | None,
    border_alpha: float,
    boundary_alpha: float,
    contrast: Mapping[str, tuple[float, float]],
    uncertainty: np.ndarray | None = None,
    uncertainty_log_vmax: float | None = None,
) -> str:
    """Render a pixel-aligned PNG; browser clicks map directly to field.mask."""
    normalized = {}
    for key in config.contract.channel_keys:
        lo, hi = contrast[key]
        if not hi > lo:
            raise ValueError(f"Invalid contrast limits for {config.label(key)}")
        normalized[key] = np.clip((field.images[key].astype(float) - lo) / (hi - lo), 0, 1)
    composite = np.zeros(field.mask.shape + (3,))
    for marker in config.markers:
        composite += normalized[marker.key][..., None] * marker.color
    composite = np.minimum(composite, 1.0)
    if mode in ("Cell identities", "Model uncertainty"):
        image = composite
    elif mode == config.reference.key:
        image = np.repeat(normalized[mode][..., None], 3, axis=2)
    elif mode in config.contract.markers:
        image = normalized[mode][..., None] * config.marker(mode).color
    else:
        raise ValueError(f"Unknown field view: {mode}")
    if mode == "Cell identities":
        color = code_colors(config, calls)[field.inverse]
        alpha = field.borders[..., None] * border_alpha
        image = image * (1 - alpha) + color * alpha
    elif mode == "Model uncertainty":
        if uncertainty is None or uncertainty.shape != (len(field.ids),):
            raise ValueError("Model uncertainty must have one value per field cell ID")
        if uncertainty_log_vmax is None or not (
            np.isfinite(uncertainty_log_vmax) and uncertainty_log_vmax > 0
        ):
            raise ValueError("Model uncertainty display maximum must be positive")
        score = uncertainty[field.inverse]
        has_score = (field.mask > 0) & ~np.isnan(score)
        scaled = np.zeros(field.mask.shape, dtype=np.float64)
        scaled[has_score] = (
            np.minimum(np.log1p(score[has_score]), uncertainty_log_vmax)
            / uncertainty_log_vmax
        )
        uncertainty_color = colormaps["viridis"](scaled)[..., :3]
        image[has_score] = uncertainty_color[has_score]
    if selected is not None:
        selected_edge = field.borders & (field.mask == selected)
        image[selected_edge] = [1, 0.8, 0]
    height, width = field.mask.shape
    fig = Figure(figsize=(width / 100, height / 100), dpi=100, frameon=False)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(
        image, extent=(0, width, height, 0), interpolation="nearest", aspect="equal"
    )
    if field.contours:
        colors, styles = (["#00D4FF"], ["-"]) if len(field.contours) == 1 else (["white", "#00D4FF"], ["--", "-"])
        for segments, color, style in zip(field.contours, colors, styles, strict=True):
            pixels = (segments - field.origin_xy_um) / field.scale_xy_um + 0.5
            ax.add_collection(
                LineCollection(
                    pixels,
                    colors=color,
                    linestyles=style,
                    linewidths=0.9,
                    alpha=boundary_alpha,
                )
            )
    # A 20 µm scale bar is in native pixels in every view.
    bar_px = 20 / field.scale_xy_um[0]
    pad = width * 0.04
    ax.plot(
        [width - pad - bar_px, width - pad],
        [height - pad, height - pad],
        color="white",
        lw=2,
    )
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.set_axis_off()
    buffer = io.BytesIO()
    canvas.print_png(buffer)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )
