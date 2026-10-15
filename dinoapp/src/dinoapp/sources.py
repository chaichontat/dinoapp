"""Dataset adapters and annotation ownership for the universal review app."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

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
    annotation_paths: tuple[Path, ...]
    default_annotation_path: Path
    subsets: tuple[str, ...]
    datasets: tuple[str, ...]
    crop_markers: tuple[str, ...]

    def review_rows(self, *, dataset: str) -> pd.DataFrame: ...

    def correction_rows(
        self,
        live_labels: dict[str, Label],
        *,
        identity: Label,
        subset: str,
        dataset: str,
    ) -> pd.DataFrame: ...

    def crop(self, cell: str, marker: str) -> Crop: ...


class AnnotationFiles:
    """Route a cell back to the annotation file that already owns it."""

    def __init__(self, paths: tuple[Path, ...], default: Path):
        if default not in paths:
            raise ValueError("Default annotation file must be one of the supplied files")
        self.paths = paths
        self.default = default
        self.values: dict[str, Label] = {}
        self.owners: dict[str, Path] = {}
        for path in paths:
            for row in load_labels(path).itertuples(index=False):
                cell = str(row.cell)
                if cell in self.owners:
                    raise ValueError(f"{cell} is annotated in multiple files")
                self.owners[cell] = path
                self.values[cell] = (int(row.brdu), int(row.edu))

    def destination(self, cell: str) -> Path:
        return self.owners.get(cell, self.default)

    def assign(self, cell: str, value: Label, *, expected: Label | None) -> Edit:
        current = AnnotationFiles(self.paths, self.default)
        path = current.destination(cell)
        previous = update_label(path, cell, value, expected=expected)
        self.values[cell] = value
        self.owners[cell] = path
        return Edit(path, cell, previous, value)

    def undo(self, edit: Edit) -> None:
        update_label(edit.path, edit.cell, edit.previous, expected=edit.value)
        if edit.previous is None:
            self.values.pop(edit.cell, None)
            self.owners.pop(edit.cell, None)
        else:
            self.values[edit.cell] = edit.previous
            self.owners[edit.cell] = edit.path


def toggle_marker(value: Label, marker: str) -> Label:
    if marker == "brdu":
        return 1 - value[0], value[1]
    if marker == "edu":
        return value[0], 1 - value[1]
    raise ValueError(f"Unexpected marker: {marker!r}")


def state_label(value: Label) -> str:
    return {
        (0, 0): "BrdU− / EdU−",
        (0, 1): "BrdU− / EdU+",
        (1, 0): "BrdU+ / EdU−",
        (1, 1): "BrdU+ / EdU+",
    }[value]


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
        self.crop_markers = tuple(
            marker for marker in ("combined", "edu", "brdu")
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
        live_labels: dict[str, Label],
        *,
        identity: Label,
        subset: str,
        dataset: str,
    ) -> pd.DataFrame:
        if subset not in self.subsets:
            raise ValueError(f"Unsupported barrage subset: {subset}")
        selected = self.review_rows(dataset=dataset)
        cells = [cell for cell, value in live_labels.items() if value == identity]
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
        if marker not in {"brdu", "edu", "combined"}:
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

    crop_markers = ("combined", "edu", "brdu")

    def __init__(
        self,
        field_source: Any,
        *,
        label_root: Path,
    ):
        self.field_source = field_source
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
        live_labels: dict[str, Label],
        *,
        identity: Label,
        subset: str,
        dataset: str,
    ) -> pd.DataFrame:
        if subset not in self.subsets or dataset != "All datasets":
            raise ValueError("Unsupported native gallery filter")
        selected = self._predictions[self._predictions["cell"].isin(live_labels)]
        if subset != "All labeled":
            selected = selected[selected["split"] == subset.lower()]
        selected = selected.loc[
            selected["cell"].map(lambda cell: live_labels[str(cell)] == identity)
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
        if marker not in {"brdu", "edu", "combined"}:
            raise ValueError(f"Unexpected crop marker: {marker}")
        label = self.cell_label(cell)
        crop_method = {
            "brdu": self.field_source.brdu_cell_crop,
            "edu": self.field_source.edu_cell_crop,
            "combined": self.field_source.cell_crop,
        }[marker]
        image, plane_z = crop_method(label)
        z_um = self.field_source.origin[0] + plane_z * self.field_source.scale[0]
        parts = []
        if cell in self._by_cell.index:
            row = self._by_cell.loc[cell]
            parts.extend([str(row["split"]).capitalize(), str(row["tranche"])])
        parts.append(f"Z {z_um:.1f} µm")
        return Crop(image, f"Cell {label}", " · ".join(parts))
