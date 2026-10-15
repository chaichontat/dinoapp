"""Dataset adapters and annotation ownership for the universal review app."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from dino_markers import (
    MarkerConfig,
    MarkerContract,
    SavedFilter,
    contract_of,
    matches_saved,
    require_same_contract,
    resolve_directory,
)

from dinoapp.label_store import Label, load_labels, update_label


@dataclass(frozen=True)
class Edit:
    path: Path
    cell: str
    previous: Label | None
    value: Label


@dataclass(frozen=True)
class Crop:
    image: bytes
    title: str
    detail: str


class ReviewSource(Protocol):
    target_id: str
    source_id: str
    display_name: str
    config: MarkerConfig
    annotation_paths: tuple[Path, ...]
    default_annotation_path: Path
    subsets: tuple[str, ...]
    datasets: tuple[str, ...]
    crop_markers: tuple[str, ...]  # ordered subset of ("combined", *config.display_markers)

    def review_rows(self, *, dataset: str) -> pd.DataFrame: ...

    def correction_rows(
        self,
        live_labels: Mapping[str, Label],
        *,
        identity: SavedFilter,
        subset: str,
        dataset: str,
    ) -> pd.DataFrame: ...

    def crop(self, cell: str, marker: str) -> Crop: ...


class NativeFieldSource(Protocol):
    config: MarkerConfig
    display_name: str
    annotation_paths: tuple[Path, ...]
    default_annotation_path: Path
    gallery_predictions: pd.DataFrame
    origin: np.ndarray
    scale: np.ndarray

    def cell_id(self, label: int) -> str: ...

    def cell_crop(self, label: int) -> tuple[bytes, int]: ...

    def marker_crop(self, label: int, marker: str) -> tuple[bytes, int]: ...

    def field_at_cell(self, label: int, width_um: float) -> Any: ...


class AnnotationFiles:
    """Route a cell back to the annotation file that already owns it."""

    def __init__(self, paths: tuple[Path, ...], default: Path, *, contract: MarkerContract):
        if default not in paths:
            raise ValueError("Default annotation file must be one of the supplied files")
        self.paths = paths
        self.default = default
        self.contract = contract
        self.values: dict[str, Label] = {}
        self.owners: dict[str, Path] = {}
        for path in paths:
            require_same_contract(
                contract, contract_of(path.parent), expected_from="this review session", actual_from=str(path.parent)
            )
            frame = load_labels(path, contract=contract)
            for cell, *marker_values in frame[["cell", *contract.markers]].itertuples(index=False):
                cell = str(cell)
                if cell in self.owners:
                    raise ValueError(f"{cell} is annotated in multiple files")
                self.owners[cell] = path
                self.values[cell] = tuple(int(value) for value in marker_values)

    def destination(self, cell: str) -> Path:
        return self.owners.get(cell, self.default)

    def assign(self, cell: str, value: Label, *, expected: Label | None) -> Edit:
        current = AnnotationFiles(self.paths, self.default, contract=self.contract)
        path = current.destination(cell)
        previous = update_label(path, cell, value, contract=self.contract, expected=expected)
        self.values[cell] = value
        self.owners[cell] = path
        return Edit(path, cell, previous, value)

    def undo(self, edit: Edit) -> None:
        update_label(edit.path, edit.cell, edit.previous, contract=self.contract, expected=edit.value)
        if edit.previous is None:
            self.values.pop(edit.cell, None)
            self.owners.pop(edit.cell, None)
        else:
            self.values[edit.cell] = edit.previous
            self.owners[edit.cell] = edit.path


def _require_saved_filter(config: MarkerConfig, identity: SavedFilter) -> None:
    if len(identity) != len(config.markers):
        raise ValueError(
            f"Saved filter {identity!r} must have one entry per marker of {list(config.contract.markers)!r}"
        )


def _path_digest(path: Path) -> str:
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


class BarrageSource:
    """Read any indexed barrage with explicit or legacy crop paths."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        index_path = self.root / "index.csv"
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing barrage index: {index_path}")
        self.rows = pd.read_csv(index_path, dtype={"cell": str})
        required = {"cell", "rank"}
        missing = required - set(self.rows.columns)
        if missing:
            raise ValueError(f"Barrage index is missing columns: {sorted(missing)}")
        if self.rows["cell"].isna().any() or self.rows["cell"].duplicated().any():
            raise ValueError("Barrage index cells must be non-null and unique")
        if self.rows["rank"].duplicated().any():
            raise ValueError("Barrage ranks must be unique")
        self.rows = self.rows.sort_values(["rank", "cell"], kind="stable").reset_index(drop=True)
        self._by_cell = self.rows.set_index("cell", drop=False)
        self.config = resolve_directory(self.root)
        self.crop_markers = tuple(
            marker for marker in ("combined", *self.config.display_markers)
            if f"{marker}_image_path" in self.rows or "image_path" in self.rows
        )
        self.annotation_paths = (self.root / "labels.csv",)
        self.default_annotation_path = self.annotation_paths[0]
        self.subsets = ("All labeled",)
        if "dataset" in self.rows:
            names = sorted(self.rows["dataset"].dropna().astype(str).unique())
            self.datasets = ("All datasets", *names)
        else:
            self.datasets = ("All datasets",)
        self.display_name = self.root.name
        completion = self.root / "complete.json"
        self.target_id = f"barrage:{self.root}"
        self.source_id = f"{self.target_id}:{_path_digest(completion)}"

    def review_rows(self, *, dataset: str) -> pd.DataFrame:
        if dataset not in self.datasets:
            raise ValueError(f"Unknown barrage dataset: {dataset}")
        selected = self.rows
        if dataset != "All datasets":
            selected = selected[selected["dataset"].astype(str) == dataset]
        return selected.reset_index(drop=True)

    def correction_rows(
        self,
        live_labels: Mapping[str, Label],
        *,
        identity: SavedFilter,
        subset: str,
        dataset: str,
    ) -> pd.DataFrame:
        if subset not in self.subsets:
            raise ValueError(f"Unsupported barrage subset: {subset}")
        _require_saved_filter(self.config, identity)
        selected = self.review_rows(dataset=dataset)
        cells = [cell for cell, value in live_labels.items() if matches_saved(value, identity)]
        return selected[selected["cell"].isin(cells)].reset_index(drop=True)

    def _image_path(self, row: pd.Series, marker: str) -> Path:
        explicit = f"{marker}_image_path"
        if explicit in row.index and pd.notna(row[explicit]):
            relative = Path(str(row[explicit]))
        elif "image_path" in row.index and pd.notna(row["image_path"]):
            relative = Path(str(row["image_path"]))
        else:
            raise ValueError(f"Barrage row has no {marker} or fallback image path")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Crop path escapes the barrage root: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing barrage crop: {path}")
        return path

    def crop(self, cell: str, marker: str) -> Crop:
        if marker != "combined" and marker not in self.config.contract.markers:
            raise ValueError(f"Unexpected crop marker: {marker}")
        if cell not in self._by_cell.index:
            raise KeyError(f"Cell is absent from barrage index: {cell}")
        row = self._by_cell.loc[cell]
        path = self._image_path(row, marker)
        parts: list[str] = []
        if "dataset" in row.index and pd.notna(row["dataset"]):
            parts.append(str(row["dataset"]))
        if "roi" in row.index and pd.notna(row["roi"]):
            parts.append(f"ROI {row['roi']}")
        parts.append(f"rank {int(row['rank']) + 1:,}")
        title = f"Cell {cell.rsplit('|', 1)[-1]}"
        return Crop(path.read_bytes(), title, " · ".join(parts))


class NativeSource:
    """Adapt native datasets to opaque-cell crops and fixed gallery rosters."""

    def __init__(
        self,
        field_source: NativeFieldSource,
        *,
        label_root: Path,
    ):
        self.field_source = field_source
        self.config = field_source.config
        self.crop_markers = ("combined", *self.config.display_markers)
        self.label_root = label_root.resolve()
        self.annotation_paths = tuple(field_source.annotation_paths)
        self.default_annotation_path = field_source.default_annotation_path
        self.subsets = ("Training", "Validation", "All labeled")
        self.datasets = ("All datasets",)
        self.display_name = field_source.display_name
        self.target_id = "native:" + json.dumps({
            "reader": f"{type(field_source).__module__}.{type(field_source).__qualname__}",
            "paths": [str(path.resolve()) for path in self.annotation_paths],
            "default": str(self.default_annotation_path.resolve()),
        }, sort_keys=True)
        self.source_id = (
            f"{self.target_id}:state:{self.label_root}:model:{getattr(field_source, 'model_version', 'unknown')}"
        )
        pixel_version = getattr(field_source, 'pixel_version', None)
        if pixel_version is not None:
            self.source_id += f":display:{pixel_version}"
        self._predictions = field_source.gallery_predictions.copy()
        if self._predictions["cell"].duplicated().any():
            raise ValueError("Native review prediction cells must be unique")
        self._by_cell = self._predictions.set_index("cell", drop=False)

    def review_rows(self, *, dataset: str) -> pd.DataFrame:
        if dataset != "All datasets":
            raise ValueError(f"Unknown native dataset: {dataset}")
        return self._predictions.reset_index(drop=True)

    def correction_rows(
        self,
        live_labels: Mapping[str, Label],
        *,
        identity: SavedFilter,
        subset: str,
        dataset: str,
    ) -> pd.DataFrame:
        if subset not in self.subsets or dataset != "All datasets":
            raise ValueError("Unsupported native gallery filter")
        _require_saved_filter(self.config, identity)
        selected = self._predictions[self._predictions["cell"].isin(live_labels)]
        if subset != "All labeled":
            selected = selected[selected["split"] == subset.lower()]
        selected = selected.loc[
            selected["cell"].map(lambda cell: matches_saved(live_labels[str(cell)], identity))
        ]
        return selected.sort_values("cell", kind="stable").reset_index(drop=True)

    def cell_label(self, cell: str) -> int:
        _, separator, value = cell.rpartition("|")
        if not separator or not value.isascii() or not value.isdecimal():
            raise ValueError(f"Invalid native cell identity: {cell}")
        label = int(value)
        if label <= 0 or self.field_source.cell_id(label) != cell:
            raise ValueError(f"Invalid native cell identity: {cell}")
        return label

    def field_at_cell(self, cell: str, width_um: float):
        return self.field_source.field_at_cell(self.cell_label(cell), width_um)

    def crop(self, cell: str, marker: str) -> Crop:
        if marker != "combined" and marker not in self.config.contract.markers:
            raise ValueError(f"Unexpected crop marker: {marker}")
        label = self.cell_label(cell)
        if marker == "combined":
            image, plane_z = self.field_source.cell_crop(label)
        else:
            image, plane_z = self.field_source.marker_crop(label, marker)
        z_um = self.field_source.origin[0] + plane_z * self.field_source.scale[0]
        parts = []
        if cell in self._by_cell.index:
            row = self._by_cell.loc[cell]
            parts.extend([str(row["split"]).capitalize(), str(row["tranche"])])
        parts.append(f"Z {z_um:.1f} µm")
        return Crop(image, f"Cell {label}", " · ".join(parts))
