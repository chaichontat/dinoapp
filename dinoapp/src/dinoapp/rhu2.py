"""Seeded ROI sampling and native fields for a prepared RHU2 barrage."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from dino_markers import (
    SPATIAL_REVIEW,
    MarkerConfig,
    probability_column,
    require_same_contract,
    resolve_directory,
    rhu2_probability_column,
)
from skimage.segmentation import find_boundaries
import zarr

from dinoapp.native import Field, FieldSource, load_model_calls
from dinoapp.sources import Crop
from dino.crop_rendering import render_crop
from dino.zarr_source import CellCrop


@dataclass(kw_only=True)
class RoiField(Field):
    roi_key: str

    @property
    def token(self) -> str:
        return f'{self.roi_key}:{super().token}:{self.origin_xy_um.tolist()}'


def sample_location(counts: tuple[int, ...], seed: int, position: int) -> tuple[int, int]:
    """Give each ROI equal probability, independent of its cell count."""
    if seed < 0 or position < 0 or not counts or any(n <= 0 for n in counts):
        raise ValueError('Sampling requires nonnegative seed/position and nonempty ROIs')
    rng = np.random.default_rng(np.random.SeedSequence([seed, position]))
    roi = int(rng.integers(len(counts)))
    return roi, int(rng.integers(counts[roi]))


class SpatialSource:
    def __init__(self, root: Path):
        self.root = root.resolve()
        manifest_path = self.root / 'spatial.json'
        self.manifest = json.loads(manifest_path.read_text())
        if not SPATIAL_REVIEW.recognizes(self.manifest):
            raise ValueError('Not an RHU2 spatial review manifest')
        self.config = resolve_directory(self.root)
        require_same_contract(
            self.config.contract, SPATIAL_REVIEW.contract(self.manifest, where=str(manifest_path)),
            expected_from=f'the annotations in {self.root}', actual_from=str(manifest_path),
        )
        markers = self.config.contract.markers
        if self.manifest['probability_columns'] != [rhu2_probability_column(marker) for marker in markers]:
            raise ValueError(
                f"{manifest_path}: probability_columns {self.manifest['probability_columns']} "
                f"disagree with {self.config.contract}"
            )
        if set(self.manifest['decision_thresholds']) != set(markers):
            raise ValueError(
                f"{manifest_path}: decision_thresholds name {sorted(self.manifest['decision_thresholds'])} "
                f"but {self.config.contract} requires {sorted(markers)}"
            )
        self.rois = tuple(sorted(self.manifest['rois'], key=lambda row: row['key']))
        if not self.rois or len({row['key'] for row in self.rois}) != len(self.rois):
            raise ValueError('Spatial ROI keys must be nonempty and unique')
        self.counts = tuple(int(row['cells']) for row in self.rois)
        self.branches = tuple(sorted(self.manifest['branches'], key=lambda row: row['key']))
        if (len({row['key'] for row in self.branches}) != len(self.branches)
                or any(row['cells'] < 0 for row in self.branches)
                or sum(row['cells'] for row in self.branches) != sum(self.counts)):
            raise ValueError('Invalid cortical branch catalog')
        sample_location(self.counts, 0, 0)
        self.source_id = f'rhu2:{self.root}:{manifest_path.stat().st_mtime_ns}'

    @lru_cache(maxsize=4)
    def roi_source(self, roi_index: int) -> RoiSource:
        return RoiSource(self.root, self.rois[roi_index], self.manifest, config=self.config)

    def resolve_cell(self, cell: str) -> tuple[RoiSource, int]:
        key, separator, value = cell.rpartition("|")
        keys = [row["key"] for row in self.rois]
        if not separator or key not in keys or not value.isascii() or not value.isdecimal():
            raise ValueError(f"Invalid RHU2 cell identity: {cell}")
        roi = self.roi_source(keys.index(key))
        label = int(value)
        if roi.cell_id(label) != cell or label not in roi.by_label.index:
            raise ValueError(f"Cell is outside the cortical cohort: {cell}")
        return roi, label

    def sample(self, seed: int, position: int) -> tuple[RoiSource, int]:
        roi_index, cell_index = sample_location(self.counts, seed, position)
        return self.roi_source(roi_index), cell_index


    @lru_cache(maxsize=1)
    def branch_annotation_counts(self, cells: tuple[str, ...]) -> tuple[int, ...]:
        """Count saved cortical cells without changing canonical branch indices."""
        if not cells:
            return (0,) * len(self.branches)
        path = (self.root / self.manifest['branch_table']).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Branch membership table must be inside the review directory')
        membership = pd.read_parquet(path, filters=[('cell', 'in', cells)])
        counts = membership.groupby(['roi_key', 'ccf_adjusted']).size()
        return tuple(int(counts.get((branch['roi_key'], branch['ccf_adjusted']), 0))
                     for branch in self.branches)

    def sample_branch(self, branch_index: int, position: int) -> tuple[RoiSource, int]:
        if not 0 <= branch_index < len(self.branches):
            raise ValueError('Invalid branch index')
        branch = self.branches[branch_index]
        if branch['cells'] <= 0:
            raise ValueError('Selected branch has no cells in the cortical cohort')
        roi_index = [row['key'] for row in self.rois].index(branch['roi_key'])
        roi = self.roi_source(roi_index)
        indices = np.flatnonzero(roi.rows.ccf_adjusted.eq(branch['ccf_adjusted']))
        if len(indices) != branch['cells']:
            raise ValueError('Branch membership count disagrees with catalog')
        _, selected = sample_location((len(indices),), branch_index, position)
        return roi, int(indices[selected])


class RoiSource(FieldSource):
    """One immutable ROI reader; labels remain qualified by dataset and ROI."""

    @classmethod
    def annotation_directories(cls, root: Path | None = None) -> tuple[Path, ...]:
        if root is None:
            raise ValueError('An RHU2 ROI source annotates its review directory; pass the review root')
        return (root,)

    def __init__(self, root: Path, metadata: dict, manifest: dict, *, config: MarkerConfig):
        self.root = root
        self.config = config
        contract = config.contract
        self.key = metadata['key']
        self.display_name = f"{metadata['dataset']} · ROI {metadata['roi']}"
        table_path = (root / metadata['table']).resolve()
        if not table_path.is_relative_to(root.resolve()):
            raise ValueError('Spatial table must be inside the review directory')
        self.rows = pd.read_parquet(table_path).sort_values('cell', kind='stable').reset_index(drop=True)
        if len(self.rows) != metadata['cells'] or self.rows.label.duplicated().any():
            raise ValueError(f'Invalid cell count or duplicate labels for {self.key}')
        if not self.rows.cell.equals(self.key + '|' + self.rows.label.astype(str)):
            raise ValueError(f'Cell identities disagree with {self.key}')
        if not (self.rows.label > 0).all():
            raise ValueError('Segmentation labels must be positive')
        branch_path = (root / manifest['branch_table']).resolve()
        if not branch_path.is_relative_to(root.resolve()):
            raise ValueError('Branch membership table must be inside the review directory')
        membership = pd.read_parquet(branch_path, filters=[('roi_key', '=', self.key)])
        membership = membership.sort_values('cell').reset_index(drop=True)
        if not membership.cell.equals(self.rows.cell) or membership.ccf_adjusted.isna().any():
            raise ValueError(f'Branch membership identities disagree with {self.key}')
        self.rows['ccf_adjusted'] = membership.ccf_adjusted.to_numpy()
        self.by_label = self.rows.set_index('label')
        self.fused = zarr.open_array(metadata['fused'], mode='r')
        self.labels = zarr.open_array(metadata['segmentation'], mode='r')
        self.shape = tuple(metadata['shape'])
        if self.labels.shape != self.shape or self.fused.shape[:3] != self.shape:
            raise ValueError(f'Native image geometry changed for {self.key}')
        keys = list(self.fused.attrs['key'])
        if keys != metadata['channel_keys'] or set(keys) != set(contract.channel_keys):
            raise ValueError(f'Native channel identities changed for {self.key}')
        self.channels = {name: keys.index(name) for name in keys}
        self.xy_scale = float(metadata['pixel_size_um'])
        if not np.isfinite(self.xy_scale) or self.xy_scale <= 0:
            raise ValueError('Physical XY pixel spacing must be positive')
        coordinates = self.rows[['z', 'y', 'x']].to_numpy()
        if not np.isfinite(coordinates).all() or not ((coordinates >= 0) & (coordinates < self.shape)).all():
            raise ValueError(f'Cell coordinates outside native image for {self.key}')
        self.annotation_paths = (root / 'labels.csv',)
        self.default_annotation_path = self.annotation_paths[0]
        self.thresholds = manifest['decision_thresholds']
        probabilities = {marker: self.rows[rhu2_probability_column(marker)] for marker in contract.markers}
        predictions = pd.DataFrame({
            'label': self.rows.label, 'cell': self.rows.cell,
            **{probability_column(marker): probabilities[marker] for marker in contract.markers},
            **{marker: probabilities[marker] >= self.thresholds[marker] for marker in contract.markers},
        })
        model = load_model_calls(
            predictions, dataset=self.key, contract=contract, cutoffs=self.thresholds, cell_prefix=self.key,
            uncertainty_log_vmax=float(manifest['uncertainty_log_vmax']),
        )
        self.calls, self.uncertainty = model.calls, model.uncertainty
        self.uncertainty_log_vmax = model.uncertainty_log_vmax
        self.display_max = float(manifest['display_max'])
        self.contrast = dict.fromkeys(contract.channel_keys, (0.0, self.display_max))

    def cell_id(self, label: int) -> str:
        return f'{self.key}|{label}'

    def probability_caption(self, label: int) -> str:
        if label not in self.by_label.index:
            return 'No saved call probabilities for this segmented cell.'
        row = self.by_label.loc[label]
        return ' · '.join(
            f'{self.config.label(marker)} probability {row[rhu2_probability_column(marker)]:.4f}'
            for marker in self.config.contract.markers
        )

    def _bounds(self, x: float, y: float, size: int) -> tuple[int, int]:
        if size <= 0 or size > min(self.shape[1:]):
            raise ValueError('Field width must fit within the ROI image')
        x0 = int(np.clip(round(x) - size // 2, 0, self.shape[2] - size))
        y0 = int(np.clip(round(y) - size // 2, 0, self.shape[1] - size))
        return x0, y0

    @lru_cache(maxsize=2)
    def sampled_field(self, cell_index: int, seed: int, position: int, width_um: float) -> RoiField:
        if not np.isfinite(width_um) or width_um <= 0:
            raise ValueError('Field width must be positive')
        row = self.rows.iloc[cell_index]
        size = int(round(width_um / self.xy_scale))
        x0, y0 = self._bounds(row.x, row.y, size)
        zi = min(int(round(row.z)), self.shape[0] - 1)
        return self._load_field(x0, y0, size, zi, seed, position)

    def field_at_cell(self, label: int, width_um: float) -> RoiField:
        if label not in self.by_label.index:
            raise ValueError(f"Cell is outside the cortical cohort: {self.cell_id(label)}")
        if not np.isfinite(width_um) or width_um <= 0:
            raise ValueError("Field width must be positive")
        row = self.by_label.loc[label]
        _, zi = self.cell_crop(label)
        size = int(round(width_um / self.xy_scale))
        x0, y0 = self._bounds(row.x, row.y, size)
        return self._load_field(x0, y0, size, zi, None, None)

    def _load_field(self, x0, y0, size, zi, seed, position) -> RoiField:
        selection = (zi, slice(y0, y0 + size), slice(x0, x0 + size))
        raw = np.asarray(self.fused[selection])
        mask = np.asarray(self.labels[selection])
        mask = np.where(np.isin(mask, self.rows.label.to_numpy()), mask, 0)
        ids, inverse = np.unique(mask, return_inverse=True)
        return RoiField(
            images={key: raw[..., self.channels[key]] for key in self.config.contract.channel_keys},
            mask=mask, ids=ids, inverse=inverse.reshape(mask.shape),
            borders=find_boundaries(mask, mode='inner') & (mask > 0),
            origin_xy_um=np.array([x0, y0]) * self.xy_scale,
            scale_xy_um=np.array([self.xy_scale, self.xy_scale]),
            center_xy_um=np.array([x0 + size / 2, y0 + size / 2]) * self.xy_scale,
            z_um=None, z_index=zi, seed=seed, position=position,
            contours=[], boundary_info={}, roi_key=self.key,
        )

    @lru_cache(maxsize=16)
    def cell_crop(self, label: int, display_max: tuple[float, ...] | None = None) -> tuple[bytes, int]:
        """Render the annotation crop with one display maximum per contract channel in the cache key."""
        keys = self.config.contract.channel_keys
        limits = (self.display_max,) * len(keys) if display_max is None else display_max
        if len(limits) != len(keys) or not np.isfinite(limits).all() or min(limits) <= 0:
            raise ValueError(f'Crop display maxima must contain {len(keys)} finite positive values')
        row = self.by_label.loc[label]
        size = 100
        x0, y0 = self._bounds(row.x, row.y, size)
        slices = (slice(None), slice(y0, y0 + size), slice(x0, x0 + size))
        labels = np.asarray(self.labels[slices])
        areas = (labels == label).sum(axis=(1, 2))
        if not areas.any():
            raise ValueError(f'No native mask at recorded centroid for {self.cell_id(label)}')
        zi = int(areas.argmax())
        plane = labels[zi]
        raw = np.asarray(self.fused[zi, slices[1], slices[2]])
        crop = CellCrop(plane, plane == label, (plane > 0) & (plane != label),
                        {name: raw[..., index] for name, index in self.channels.items()})
        image = render_crop(crop, cell=self.cell_id(label),
                             display_max=dict(zip(keys, limits, strict=True)),
                             panels=self.config.panels)
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return buffer.getvalue(), zi


class FieldCropSource:
    """Combined native crops for cells captured from one cortical ROI field."""

    crop_markers = ("combined",)
    datasets = ("All datasets",)
    subsets = ("All labeled",)

    def __init__(self, roi: RoiSource, contrast: tuple[float, ...], target_id: str, revision: str):
        self.field_source = roi
        self.config = roi.config
        self.contrast = contrast
        self.target_id, self.source_id = target_id, revision
        self.display_name = roi.display_name
        self.annotation_paths = roi.annotation_paths
        self.default_annotation_path = roi.default_annotation_path

    def cell_label(self, cell: str) -> int:
        matches = self.field_source.rows.loc[self.field_source.rows.cell.eq(cell), "label"]
        if len(matches) != 1:
            raise ValueError(f"Cell is outside the cortical ROI cohort: {cell}")
        return int(matches.iloc[0])

    def crop(self, cell: str, marker: str) -> Crop:
        if marker != "combined":
            raise ValueError("Captured RHU2 cells support combined crops")
        image, plane = self.field_source.cell_crop(self.cell_label(cell), self.contrast)
        return Crop(image, cell, f"{self.display_name} · Z plane {plane}")
