"""Native ventricular fields and edits to the existing barrage annotations."""

from __future__ import annotations

import base64
import hashlib
import io
import json
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
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from skimage.segmentation import find_boundaries

from dinoapp.label_store import Label
from dinoapp.sources import AnnotationFiles as AnnotationFiles
from dinoapp.native_geometry import image_geometry, vertex_normals
from dino.zarr_source import CellCropReader, metadata_digest
from dino.review_contracts import (
    combined_uncertainty as combined_uncertainty,
    load_prediction_pointer as load_prediction_pointer,
)
from dinoapp.review_predictions import load_review_predictions
from squisher.jpegxr_zarr import register_jpegxr_codec

ROOT = Path("/working/eduseg/20260808-prod2/full-run2")
REVIEW_ROOT = ROOT
PALETTE = np.array(
    [[0.25, 0.25, 0.25], [0, 1, 0], [1, 0, 1], [1, 1, 1], [0.6, 0.6, 0.6]]
)
IDENTITIES = ("Neither", "EdU only", "BrdU only", "Dual positive", "No saved call")


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
    live_labels: dict[str, Label],
    *,
    split: str,
    target: str,
    identity: Label,
) -> pd.DataFrame:
    """Rank hard model disagreements within one current saved identity."""
    required = {
        "cell",
        "label",
        "tranche",
        "split",
        "p_brdu",
        "pred_brdu",
        "p_edu",
        "pred_edu",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Labeled predictions are missing columns: {sorted(missing)}")
    if predictions["cell"].duplicated().any():
        raise ValueError("Labeled prediction cells must be unique")
    if split not in {"training", "validation", "all"}:
        raise ValueError(f"Unexpected split: {split}")
    if target not in {"brdu", "edu", "either"}:
        raise ValueError(f"Unexpected target: {target}")

    rows = training_gallery(
        predictions, live_labels, split=split, identity=identity
    )
    rows["brdu_miscall"] = rows["pred_brdu"].astype(bool) != rows["saved_brdu"].astype(
        bool
    )
    rows["edu_miscall"] = rows["pred_edu"].astype(bool) != rows["saved_edu"].astype(
        bool
    )
    brdu_score = np.where(
        rows["saved_brdu"].to_numpy() == 1,
        1 - rows["p_brdu"].to_numpy(),
        rows["p_brdu"].to_numpy(),
    )
    edu_score = np.where(
        rows["saved_edu"].to_numpy() == 1,
        1 - rows["p_edu"].to_numpy(),
        rows["p_edu"].to_numpy(),
    )
    scores = {"brdu": brdu_score, "edu": edu_score}
    markers = (target,) if target != "either" else ("brdu", "edu")
    disagreements = []
    for marker in markers:
        keep = rows[f"{marker}_miscall"].to_numpy()
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
    live_labels: dict[str, Label],
    *,
    split: str,
    identity: Label,
) -> pd.DataFrame:
    """Select labeled cells by their current saved BrdU/EdU identity."""
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
        or len(identity) != 2
        or any(value not in (0, 1) for value in identity)
    ):
        raise ValueError("Gallery identity must be a binary (BrdU, EdU) pair")

    rows = predictions[predictions["cell"].isin(live_labels)].copy()
    if split != "all":
        rows = rows[rows["split"] == split].copy()
    rows["saved_brdu"] = [live_labels[cell][0] for cell in rows["cell"]]
    rows["saved_edu"] = [live_labels[cell][1] for cell in rows["cell"]]
    rows = rows[
        (rows["saved_brdu"] == identity[0]) & (rows["saved_edu"] == identity[1])
    ]
    return rows.sort_values("cell").reset_index(drop=True)


def _toggle_marker(identity: Label, index: int) -> Label:
    if (
        not isinstance(identity, tuple)
        or len(identity) != 2
        or any(value not in (0, 1) for value in identity)
    ):
        raise ValueError("Cell identity must be a binary (BrdU, EdU) pair")
    value = list(identity)
    value[index] = 1 - value[index]
    return value[0], value[1]


def toggle_edu(identity: Label) -> Label:
    """Flip only the EdU bit of a binary (BrdU, EdU) identity."""
    return _toggle_marker(identity, 1)


def toggle_brdu(identity: Label) -> Label:
    """Flip only the BrdU bit of a binary (BrdU, EdU) identity."""
    return _toggle_marker(identity, 0)


def _render_marker_crop(crop, marker: str, display_max: float) -> bytes:
    """Render one native marker crop with established contrast and cell outlines."""
    from PIL import Image

    display_name = {"brdu": "BrdU", "edu": "EdU"}[marker]
    channel = crop.channels[marker]
    if channel.shape != crop.mask.shape:
        raise ValueError(
            f"{display_name} channel/mask shape mismatch: "
            f"{channel.shape} != {crop.mask.shape}"
        )
    if not np.isfinite(display_max) or display_max <= 0:
        raise ValueError(f"{display_name} display maximum must be finite and positive")
    scaled = np.clip(channel.astype(np.float64) / display_max, 0, 1)
    rgb = colormaps["magma"](scaled)[..., :3]
    boundaries = find_boundaries(crop.labels, mode="inner")
    rgb[boundaries] = 0.65 * rgb[boundaries] + 0.35
    target = find_boundaries(crop.mask, mode="inner")
    rgb[target] = (0, 1, 1)
    output = io.BytesIO()
    Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(output, format="PNG")
    return output.getvalue()


def render_edu_crop(crop, display_max: float) -> bytes:
    """Render one native EdU crop with established contrast and cell outlines."""
    return _render_marker_crop(crop, "edu", display_max)


def render_brdu_crop(crop, display_max: float) -> bytes:
    """Render one native BrdU crop with established contrast and cell outlines."""
    return _render_marker_crop(crop, "brdu", display_max)


def prefetch_edu_crop(source, cell: str) -> None:
    """Populate the EdU crop cache for one gallery cell."""
    source.edu_cell_crop(int(cell.rsplit("|", 1)[1]))


def prefetch_brdu_crop(source, cell: str) -> None:
    """Populate the BrdU crop cache for one gallery cell."""
    source.brdu_cell_crop(int(cell.rsplit("|", 1)[1]))


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
    anatomy_channel: str = "561"

    @property
    def token(self) -> str:
        return f"{self.seed}:{self.position}:{self.z_index}:{self.mask.shape}"


class FieldSource:
    """Read fixed-Z native pixels and sample the full supported 90% wall contour."""

    display_name = "Prod2 native"

    def __init__(self, root: Path = ROOT):
        self.root = root
        self.annotation_paths = tuple(
            root / name / "labels.csv"
            for name in ("barrage-cortex", "barrage-vz-svz", "barrage-pixel")
        )
        self.default_annotation_path = self.annotation_paths[-1]
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
        self.sources = {k: Path(v["path"]) for k, v in native["sources"].items()}
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
            channels={
                "pi": self.arrays["561"],
                "edu": self.arrays["EdU"],
                "brdu": self.arrays["BrdU"],
            },
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
        self.model_version, prediction_dir, prediction_manifest, prediction_contract = (
            load_prediction_pointer(root / "cell-review/current.json")
        )
        table = ds.dataset(prediction_dir, format="parquet").to_table(
            columns=["label", "cell", "brdu", "edu", "p_brdu", "p_edu"]
        )
        frame = table.to_pandas()
        if frame.label.duplicated().any() or frame[["brdu", "edu"]].isna().any().any():
            raise ValueError(
                "Prediction identities must be unique with complete binary calls"
            )
        if not frame.cell.equals("full-run2|" + frame.label.astype(str)):
            raise ValueError(
                "Prediction cell identities do not match full-run2 segmentation labels"
            )
        if not frame[["brdu", "edu"]].isin([False, True]).all().all():
            raise ValueError("Prediction calls must be binary")
        self.calls = np.full(int(frame.label.max()) + 1, 4, dtype=np.uint8)
        self.calls[frame.label] = frame.edu.to_numpy(
            dtype=np.uint8
        ) + 2 * frame.brdu.to_numpy(dtype=np.uint8)
        uncertainty = combined_uncertainty(
            frame.p_brdu.to_numpy(),
            frame.p_edu.to_numpy(),
            brdu_threshold=float(prediction_contract["brdu_cutoff"]),
            edu_threshold=float(prediction_contract["edu_cutoff"]),
        )
        self.uncertainty = np.full(len(self.calls), np.nan, dtype=np.float32)
        self.uncertainty[frame.label] = uncertainty
        finite_log_uncertainty = np.log1p(uncertainty[np.isfinite(uncertainty)])
        if not len(finite_log_uncertainty):
            raise ValueError("No finite model uncertainty scores are available")
        self.uncertainty_log_vmax = float(np.quantile(finite_log_uncertainty, 0.999))
        self.prediction_manifest = str(prediction_manifest)
        self.classifier_caption = (
            f"Classifier {str(prediction_contract['model_sha256'])[:12]} · "
            f"BrdU cutoff {float(prediction_contract['brdu_cutoff']):.10f} · "
            f"EdU cutoff {float(prediction_contract['edu_cutoff']):.10f}"
        )
        self.training_review_manifest = str(prediction_contract["review_manifest"])
        self.training_predictions = load_review_predictions(
            Path(self.training_review_manifest), cell_prefix="full-run2",
            model_sha256=str(prediction_contract["model_sha256"]),
        )
        self.contrast = np.array(
            [native["contrast_p1_p99_8"][k] for k in ["561", "BrdU", "EdU"]]
        )
        print(
            f"Ready: {len(frame):,} model calls; default Z={self.default_z:.3f} µm",
            flush=True,
        )

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
            crop, cell=self.cell_id(label), display_max=contract["display_max"]
        ).save(output, format="PNG")
        return output.getvalue(), plane_z

    # Retain one 24-cell visible page plus its 24-cell prefetched successor.
    @lru_cache(maxsize=64)
    def edu_cell_crop(self, label: int):
        """Render only EdU for the dense correction gallery."""
        crop, plane_z, contract = self._read_cell_crop(label)
        return render_edu_crop(crop, contract["display_max"]["edu"]), plane_z

    @lru_cache(maxsize=64)
    def brdu_cell_crop(self, label: int):
        """Render only BrdU for the dense correction gallery."""
        crop, plane_z, contract = self._read_cell_crop(label)
        return render_brdu_crop(crop, contract["display_max"]["brdu"]), plane_z

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
        images = {k: np.asarray(v[sl]) for k, v in self.arrays.items()}
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
        calls = np.full(len(field.ids), 4, dtype=np.uint8)
        valid = field.ids < len(self.calls)
        calls[valid] = self.calls[field.ids[valid]]
        for i, label in enumerate(field.ids):
            manual = annotations.values.get(self.cell_id(int(label)))
            if manual is not None:
                calls[i] = manual[1] + 2 * manual[0]
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
    mode: str,
    selected: int | None,
    border_alpha: float,
    boundary_alpha: float,
    contrast: np.ndarray,
    uncertainty: np.ndarray | None = None,
    uncertainty_log_vmax: float | None = None,
) -> str:
    """Render a pixel-aligned PNG; browser clicks map directly to field.mask."""
    normalized = {}
    for i, name in enumerate([field.anatomy_channel, "BrdU", "EdU"]):
        lo, hi = contrast[i]
        if not hi > lo:
            raise ValueError(f"Invalid contrast limits for {name}")
        normalized[name] = np.clip(
            (field.images[name].astype(float) - lo) / (hi - lo), 0, 1
        )
    brdu, edu = normalized["BrdU"], normalized["EdU"]
    if mode == field.anatomy_channel:
        image = np.repeat(normalized[field.anatomy_channel][..., None], 3, axis=2)
    elif mode == "BrdU":
        image = np.stack((brdu, np.zeros_like(brdu), brdu), axis=-1)
    elif mode == "EdU":
        image = np.stack((np.zeros_like(edu), edu, np.zeros_like(edu)), axis=-1)
    else:
        image = np.stack((brdu, edu, brdu), axis=-1)
    if mode == "Cell identities":
        color = PALETTE[calls[field.inverse]]
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
