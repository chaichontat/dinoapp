"""Prod3 data adapter for the shared native ventricular-field reviewer."""

from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path
from threading import Lock

import igl
import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import zarr
from dino_markers import cutoff_key, probability_column
from skimage.segmentation import find_boundaries

from dinoapp.native import (
    AnnotationFiles as AnnotationFiles,
    Field,
    adapter_config,
    classifier_caption,
    load_prediction_pointer,
    prediction_contract,
    render_field as render_field,
    render_marker_crop,
    sample_center,
    read_cell_geometry,
    FieldSource as NativeFieldSource,
)
from dino.zarr_source import CellCropReader, metadata_digest
from dinoapp.display_pixels import DisplayPixels
from dinoapp.review_predictions import load_review_predictions
from squisher.jpegxr_zarr import register_jpegxr_codec
from dinoapp.native_geometry import mesh_vertex_normals as vertex_normals


ROOT = Path("/working/eduseg/prod3")
REVIEW_ROOT = ROOT / "dino-prod3-realigned-v1"


class FieldSource(NativeFieldSource):
    """Read prod3 native pixels, calls, uncertainty, and reviewed VZ/SVZ boundaries."""

    display_name = "Prod3 native"
    dataset_name = "Prod3"
    cell_prefix = "prod3-realigned-v1"
    raw_channels = {"561": "pi", "BrdU": "brdu", "EdU": "edu"}

    @classmethod
    def annotation_directories(cls, root: Path | None = None) -> tuple[Path, ...]:
        return (REVIEW_ROOT / "barrage-cortex-v1", REVIEW_ROOT / "barrage-vz-svz-v1")

    def __init__(self, root: Path = ROOT):
        self.root = root
        self.annotation_paths = tuple(directory / "labels.csv" for directory in self.annotation_directories(root))
        self.default_annotation_path = self.annotation_paths[-1]
        self.config = adapter_config(
            self.default_annotation_path.parent, self.raw_channels, adapter=self.dataset_name
        )
        source_path = REVIEW_ROOT / "source/source.json"
        source = json.loads(source_path.read_text())
        labels_meta = source["labels"]
        if (
            metadata_digest(Path(labels_meta["root"]), labels_meta["array"])
            != labels_meta["metadata_sha256"]
        ):
            raise ValueError("Postprocessed labels differ from the saved prod3 DINO source")

        self.sources = {
            "561": root
            / "235-2-561638514-3/fusion-561-preibisch-v1/fused.ch0.ome.zarr",
            **{
                name: Path(source["channels"][key]["root"])
                for name, key in self.raw_channels.items()
                if key in self.config.contract.markers
            },
        }
        grid = source["channels"][self.config.contract.markers[0]]
        geometry = grid["ome"]["coordinate_transformations"]
        self.scale = np.asarray(next(v["scale"] for v in geometry if v["type"] == "scale"))
        self.origin = np.asarray(
            next(v["translation"] for v in geometry if v["type"] == "translation")
        )
        self.shape = tuple(grid["shape"])
        self.default_z = float(
            self.origin[0] + round((679.05868151 - self.origin[0]) / self.scale[0]) * self.scale[0]
        )

        register_jpegxr_codec()
        self.arrays = {
            name: zarr.open_group(str(path), mode="r")["0"]
            for name, path in self.sources.items()
        }
        if any(tuple(array.shape) != self.shape for array in self.arrays.values()):
            raise ValueError("Prod3 native channel shapes differ")
        self.labels = zarr.open_array(labels_meta["root"], mode="r")
        if tuple(self.labels.shape) != self.shape:
            raise ValueError("Prod3 postprocessed labels and native microscopy grids differ")

        self.display_pixels = DisplayPixels(REVIEW_ROOT / "model/model.joblib", REVIEW_ROOT / "source")
        self._require_contract(self.display_pixels.contract, actual_from=str(self.display_pixels.model_path))
        self.display_pixels.validate_native_grid(
            labels=labels_meta["root"], shape=self.shape, scale=self.scale, origin=self.origin,
        )
        for name, key in self.raw_channels.items():
            if key in self.config.contract.markers:
                self.arrays[name] = self.display_pixels.source.channels[key]
        self.crop_reader = CellCropReader(
            labels=self.labels,
            channels={
                **{key: self.arrays[name] for name, key in self.raw_channels.items()},
                "dapi": self.display_pixels.source.channels["pi"],
            },
            offset_zyx=(0, 0, 0),
            cache_bytes=4
            * int(np.prod(self.labels.chunks))
            * self.labels.dtype.itemsize
            + 64 * 1024**2,
        )
        self.pixel_version = self.display_pixels.identity
        self.crop_lock = Lock()
        self.display_name = f"{self.display_name} · {self.display_pixels.description}"
        self.props_path = REVIEW_ROOT / "source/cells.parquet"
        self.crop_contract = json.loads(
            (REVIEW_ROOT / "barrage-cortex-v1/run.json").read_text()
        )
        self.crop_contract["display_max"]["pi"] = 28686.583613151608

        mesh = np.load(
            root / "surface-workflow-overhang-v2/vz-surface/surface.npz"
        )
        vertices = np.asarray(mesh["vertices_xyz_um"], dtype=np.float64)
        faces = np.asarray(mesh["faces"], dtype=np.int64)
        normals = vertex_normals(vertices, faces)
        boundary = np.load(
            root
            / "surface-workflow-confidence-v4/eomes-boundaries/boundaries.npz"
        )
        self.boundaries = []
        for name in ("inner", "outer"):
            cutoff = boundary[f"{name}_um"]
            keep = (
                boundary["wall_support"]
                & boundary[f"estimated_{name}"]
                & np.isfinite(cutoff)
            )
            selected = faces[np.all(keep[faces], axis=1)]
            ids, inverse = np.unique(selected, return_inverse=True)
            self.boundaries.append(
                (
                    vertices[ids] - normals[ids] * cutoff[ids, None],
                    inverse.reshape(-1, 3),
                )
            )
        self.layer_path = REVIEW_ROOT / "vz-layers/cells.parquet"

        _, prediction_dir, prediction_manifest, prediction_record = (
            load_prediction_pointer(REVIEW_ROOT / "cell-review/current.json")
        )
        self._require_contract(
            prediction_contract(prediction_record, where=str(prediction_manifest)),
            actual_from=str(prediction_manifest),
        )
        markers = self.config.contract.markers
        columns = ["label", "cell", *markers, *(probability_column(marker) for marker in markers)]
        frame = ds.dataset(prediction_dir, format="parquet").to_table(columns=columns).to_pandas()
        cutoffs = {marker: float(prediction_record[cutoff_key(marker)]) for marker in markers}
        self._adopt_model_calls(frame, cutoffs=cutoffs, cell_prefix=self.cell_prefix)
        self.model_version = str(prediction_record["model_sha256"])[:12]
        self.prediction_manifest = str(prediction_manifest)
        self.classifier_caption = classifier_caption(self.model_version, self.config, cutoffs)
        self.training_review_manifest = str(prediction_record["review_manifest"])
        self.training_predictions = load_review_predictions(
            Path(self.training_review_manifest), cell_prefix=self.cell_prefix,
            model_sha256=str(prediction_record["model_sha256"]), contract=self.config.contract,
        )
        display_max = self.crop_contract["display_max"]
        self.contrast = {
            self.config.reference.key: (787.3379356892904, 28686.583613151608),
            **{marker: (0.0, display_max[marker]) for marker in markers},
        }
        print(
            f"Ready: {len(frame):,} prod3 model calls; default Z={self.default_z:.3f} µm",
            flush=True,
        )

    def cell_id(self, label: int) -> str:
        return f"{self.cell_prefix}|{label}"

    def cell_geometry(self, label: int):
        return read_cell_geometry(self.props_path, label, self.shape)

    def _read_cell_crop(self, label: int):
        from dino.crop_rendering import read_crop

        row = self.cell_geometry(label)
        with self.crop_lock:
            crop = read_crop(self.crop_reader, row, size=self.crop_contract["size"])
            size = self.crop_contract["size"]
            origin = (int(row.plane_z), int(np.rint(row.centroid_y)) - size // 2,
                      int(np.rint(row.centroid_x)) - size // 2)
            crop = self.display_pixels.crop(crop, origin, reference="dapi")
        if not crop.mask.any():
            raise ValueError(f"Cell {label} has no mask on saved plane {int(row.plane_z)}")
        return crop, int(row.plane_z)

    @property
    def gallery_predictions(self):
        return self.training_predictions

    @lru_cache(maxsize=32)
    def cell_crop(self, label: int):
        from dino.crop_rendering import render_crop

        crop, plane_z = self._read_cell_crop(label)
        output = io.BytesIO()
        render_crop(
            crop,
            cell=self.cell_id(label),
            display_max=self.crop_contract["display_max"],
            panels=self.config.panels,
        ).save(output, format="PNG")
        return output.getvalue(), plane_z

    @lru_cache(maxsize=128)
    def marker_crop(self, label: int, marker: str):
        if marker not in self.config.contract.markers:
            raise ValueError(f"Unexpected crop marker: {marker}")
        crop, plane_z = self._read_cell_crop(label)
        image = render_marker_crop(
            crop, marker, self.crop_contract["display_max"][marker], label=self.config.label(marker)
        )
        return image, plane_z

    @lru_cache(maxsize=8)
    def contours_at(self, z_index: int) -> list[np.ndarray]:
        z_um = float(self.origin[0] + z_index * self.scale[0])
        result = []
        for vertices, faces in self.boundaries:
            points, edges, _ = igl.isolines(
                vertices, faces, vertices[:, 2], np.asarray([z_um])
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
            + (np.asarray(self.shape[:0:-1]) - 1) * scale_xy
            - width_um / 2
            - scale_xy
        )
        segments = contours[-1]
        valid = np.all((segments >= low) & (segments <= high), axis=(1, 2))
        center = sample_center(segments[valid], seed=seed, position=position)
        lo = np.floor((center - width_um / 2 - origin_xy) / scale_xy).astype(int)
        hi = np.ceil((center + width_um / 2 - origin_xy) / scale_xy).astype(int)
        return self._load_field(zi, lo, hi, center, seed, position)

    def _load_field(self, zi, lo, hi, center, seed, position) -> Field:
        scale_xy, origin_xy = self.scale[:0:-1], self.origin[:0:-1]
        contours = self.contours_at(zi)
        selection = (zi, slice(lo[1], hi[1]), slice(lo[0], hi[0]))
        images = {self.raw_channels[name]: np.asarray(array[selection]) for name, array in self.arrays.items()}
        images = self.display_pixels.field_images(images, selection)
        mask = np.asarray(self.labels[selection])
        ids, inverse = np.unique(mask, return_inverse=True)
        wanted = [int(label) for label in ids if label]
        boundary_info = self._field_boundary_info(wanted)
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

    def _field_boundary_info(self, labels: list[int]) -> dict[int, dict[str, object]]:
        result = {}
        if labels:
            for row in pq.read_table(
                self.layer_path,
                columns=["label", "r_vz_um", "inside_vz_svz", "layer"],
                filters=[("label", "in", labels)],
            ).to_pylist():
                result[row["label"]] = {
                    "r_vz_um": row["r_vz_um"],
                    "included_60": row["layer"] == "VZ",
                    "included_90": row["inside_vz_svz"],
                }
        return result

