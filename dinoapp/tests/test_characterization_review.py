"""Characterization of today's annotation storage and default review presentation.

The frozen files under fixtures/legacy_review were written by the code at the
commit named in that directory's README. They pin what persisted annotation
artifacts look like and prove that they keep loading and resuming. Calls into
APIs that the marker configuration refactor changes are isolated in the
invocation helpers below; the assertions in the tests stay unchanged.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest
import zarr
from PIL import Image
from skimage.segmentation import find_boundaries

from dino.zarr_source import CellCrop, metadata_digest
from dinoapp import native
from dinoapp.label_store import LabelConflictError, load_labels, update_label
from dinoapp.review_session import get_context, save_assignment, undo_edit
from dinoapp.sources import AnnotationFiles, BarrageSource


@pytest.fixture(autouse=True)
def _matplotlib_defaults():
    """Render under the startup rc settings; dino.roc applies a global style."""
    with matplotlib.rc_context():
        matplotlib.rc_file_defaults()
        yield

FIXTURES = Path(__file__).parent / "fixtures" / "legacy_review"
LEGACY_BARRAGE = FIXTURES / "barrage"
LEGACY_SESSION = "frozen01"
INDEX_CELLS = ["tiny:r1|1", "tiny:r1|2", "tiny:r1|3"]


# ---------------------------------------------------------------------------
# Invocation helpers. Later waves update these calls, never the assertions.
# ---------------------------------------------------------------------------


def _update(path: Path, cell: str, value, *, expected="any"):
    return update_label(path, cell, value, expected=expected)


def _annotations(paths: tuple[Path, ...], default: Path) -> AnnotationFiles:
    return AnnotationFiles(paths, default)


def _barrage_annotations(source: BarrageSource) -> AnnotationFiles:
    return AnnotationFiles(source.annotation_paths, source.default_annotation_path)


def _saved_values(annotations: AnnotationFiles) -> dict[str, tuple[int, int]]:
    return dict(annotations.values)


def _assign(annotations: AnnotationFiles, cell: str, value, *, expected):
    return annotations.assign(cell, value, expected=expected)


def _crop(source, cell: str, marker: str):
    return source.crop(cell, marker)


def _toggle(value, marker: str):
    return {"brdu": native.toggle_brdu, "edu": native.toggle_edu}[marker](value)


def _save(annotations, state_dir: Path, context, cell: str, value, *, expected, record):
    return save_assignment(annotations, state_dir, context, cell, value, expected=expected, record=record)


def _undo(annotations, context, state_dir: Path, edit) -> None:
    undo_edit(annotations, context, state_dir, edit)


def _render_marker_crop(crop: CellCrop, marker: str, display_max: float) -> bytes:
    return {"brdu": native.render_brdu_crop, "edu": native.render_edu_crop}[marker](crop, display_max)


def _load_native_calls(root: Path) -> np.ndarray:
    return native.FieldSource(root).calls


def _overlay_calls(field_ids: np.ndarray, predicted: np.ndarray, manual: dict[str, tuple[int, int]]) -> np.ndarray:
    source = object.__new__(native.FieldSource)
    source.calls = predicted
    return source.field_calls(SimpleNamespace(ids=field_ids), SimpleNamespace(values=manual))


FIELD_MODES = {"brdu": "BrdU", "edu": "EdU", "merge": "Cell identities", "identities": "Cell identities"}


def _render_field_pixels(
    *,
    brdu: np.ndarray,
    edu: np.ndarray,
    mask: np.ndarray,
    calls: np.ndarray,
    view: str,
    contrast: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Render today's native field and return its RGB pixels.

    "merge" is the identity view without cell borders, which shows the composite alone.
    """
    ids, inverse = np.unique(mask, return_inverse=True)
    borders = find_boundaries(mask, mode="inner") & (mask > 0)
    field = native.Field(
        images={"561": np.zeros_like(brdu), "BrdU": brdu, "EdU": edu},
        mask=mask,
        ids=ids,
        inverse=inverse.reshape(mask.shape),
        borders=borders if view == "identities" else np.zeros_like(borders),
        origin_xy_um=np.zeros(2),
        scale_xy_um=np.array([10.0, 10.0]),
        center_xy_um=np.zeros(2),
        z_um=0.0,
        z_index=0,
        seed=None,
        position=None,
        contours=[],
        boundary_info={},
    )
    url = native.render_field(
        field,
        calls,
        mode=FIELD_MODES[view],
        selected=None,
        border_alpha=1.0,
        boundary_alpha=0.0,
        contrast=np.array([[0.0, 1.0], contrast["brdu"], contrast["edu"]]),
    )
    prefix = "data:image/png;base64,"
    assert url.startswith(prefix)
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(url[len(prefix):]))).convert("RGB"))


# ---------------------------------------------------------------------------
# Builders.
# ---------------------------------------------------------------------------


def copy_barrage(tmp_path: Path) -> Path:
    root = tmp_path / "barrage"
    shutil.copytree(LEGACY_BARRAGE, root)
    return root.resolve()


def frozen_journal() -> list[dict]:
    path = LEGACY_BARRAGE / "cell-review" / f"changes-{LEGACY_SESSION}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def barrage_context(monkeypatch, source: BarrageSource, session_id: str) -> dict:
    import streamlit as st

    monkeypatch.setattr(st, "session_state", {})
    context = get_context(source.source_id)
    context["session_id"] = session_id
    return context


def frozen_crop(rank: int) -> CellCrop:
    with np.load(FIXTURES / "barrage_crops.npz") as arrays:
        stem = f"{rank:04d}"
        return CellCrop(
            labels=arrays[f"{stem}_labels"],
            mask=arrays[f"{stem}_mask"],
            neighbor_mask=arrays[f"{stem}_neighbor_mask"],
            channels={name: arrays[f"{stem}_{name}"] for name in ("pi", "edu", "brdu")},
        )


def _write_json(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def build_native_root(tmp_path: Path, predictions: pd.DataFrame) -> Path:
    """Write the smallest prod2 native review root that today's FieldSource opens."""
    root = tmp_path / "full-run2"
    shape = (1, 4, 4)
    segmentation = tmp_path / "segmentation.zarr"
    zarr.open_array(segmentation, mode="w", shape=shape, chunks=shape, dtype="u4")
    _write_json(
        {"labels": {"root": str(segmentation), "array": "",
                    "metadata_sha256": metadata_digest(segmentation, "")}},
        root / "dino-v3" / "source.json",
    )
    axes = [{"name": axis, "type": "space", "unit": "micrometer"} for axis in "zyx"]
    sources = {}
    for name in ("561", "BrdU", "EdU"):
        path = tmp_path / f"{name}.zarr"
        group = zarr.open_group(path, mode="w")
        group.attrs.update({"squisher_complete": True, "ome": {"multiscales": [{
            "axes": axes,
            "datasets": [{"path": "0", "coordinateTransformations": [
                {"type": "scale", "scale": [1.0, 1.0, 1.0]},
                {"type": "translation", "translation": [0.0, 0.0, 0.0]},
            ]}],
        }]}})
        group.create_array("0", shape=shape, chunks=shape, dtype="uint16")
        sources[name] = {"path": str(path)}
    surface = tmp_path / "surface.npz"
    np.savez(surface, vertices_xyz_um=np.array([[0.0, 0, 0], [0, 1, 0], [0, 0, 1]]), faces=np.array([[0, 1, 2]]))
    _write_json(
        {"planes": [{"z_um": 0.0}], "sources": sources, "surface": str(surface),
         "contrast_p1_p99_8": {"561": [0, 1], "BrdU": [0, 1], "EdU": [0, 1]}},
        root / "vz-native" / "native-channels.json",
    )
    trim = tmp_path / "full-run" / "svz-cut" / "trim"
    trim.mkdir(parents=True)
    for name in ("vz60-s30.npz", "vz90-vz60-s30.npz"):
        np.savez(trim / name, wall_support=np.ones(3, dtype=bool), cutoff_um=np.ones(3))
    for name in ("vz30", "vz90-nested"):
        path = root / "vz-rates" / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"label": [1]}).to_parquet(path, index=False)
        _write_json({"output_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, path.with_suffix(".json"))

    model_sha = "a" * 64
    model_dir = tmp_path / "model-v1"
    prediction_dir = model_dir / "cortex-pred" / "predictions"
    prediction_dir.mkdir(parents=True)
    predictions.to_parquet(prediction_dir / "part-0.parquet", index=False)
    _write_json(
        {"model_sha256": model_sha, "cells": len(predictions), "brdu_cutoff": 0.5, "edu_cutoff": 0.5},
        prediction_dir.parent / "complete.json",
    )
    review_dir = model_dir / "labeled-review"
    review_dir.mkdir()
    review = pd.DataFrame({
        "cell": predictions.cell, "label": predictions.label, "tranche": "t0", "split": "training",
        "p_brdu": predictions.p_brdu, "pred_brdu": predictions.brdu.astype(int),
        "p_edu": predictions.p_edu, "pred_edu": predictions.edu.astype(int),
    })
    review.to_parquet(review_dir / "predictions.parquet", index=False)
    manifest = _write_json(
        {"artifact": "dino-labeled-review-v1", "model_sha256": model_sha,
         "predictions": str(review_dir / "predictions.parquet"),
         "predictions_sha256": hashlib.sha256((review_dir / "predictions.parquet").read_bytes()).hexdigest(),
         "cells": len(review), "training_cells": len(review), "validation_cells": 0,
         "brdu_cutoff": 0.5, "edu_cutoff": 0.5},
        review_dir / "complete.json",
    )
    _write_json(
        {"artifact": "dino-review-current-v1", "model_dir": str(model_dir),
         "prediction_dir": str(prediction_dir), "model_sha256": model_sha, "cells": len(predictions),
         "review_manifest": str(manifest),
         "review_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
        root / "cell-review" / "current.json",
    )
    return root


# ---------------------------------------------------------------------------
# labels.csv storage.
# ---------------------------------------------------------------------------


def test_new_labels_file_is_created_sorted_by_cell(tmp_path):
    path = tmp_path / "barrage" / "labels.csv"

    assert _update(path, "tiny:r1|2", (1, 0)) is None
    assert path.read_bytes() == b"cell,brdu,edu\ntiny:r1|2,1,0\n"
    assert _update(path, "tiny:r1|10", (0, 1)) is None
    assert _update(path, "aa:x|1", (1, 1)) is None
    assert path.read_bytes() == b"cell,brdu,edu\naa:x|1,1,1\ntiny:r1|10,0,1\ntiny:r1|2,1,0\n"

    assert _update(path, "tiny:r1|10", (0, 0), expected=(0, 1)) == (0, 1)
    assert _update(path, "aa:x|1", None, expected=(1, 1)) == (1, 1)
    assert path.read_bytes() == b"cell,brdu,edu\ntiny:r1|10,0,0\ntiny:r1|2,1,0\n"
    assert sorted(p.name for p in path.parent.iterdir()) == ["labels.csv", "labels.csv.lock"]


def test_editing_a_labels_file_keeps_extra_columns_and_rows_outside_the_index(tmp_path):
    root = copy_barrage(tmp_path)
    shutil.copyfile(FIXTURES / "labels_extras.csv", root / "labels.csv")
    source = BarrageSource(root)
    annotations = _barrage_annotations(source)
    assert _saved_values(annotations) == {"zz:other|9": (1, 1), "tiny:r1|2": (1, 0), "aa:other|4": (0, 1)}

    _assign(annotations, "tiny:r1|1", (1, 1), expected=None)
    _assign(annotations, "tiny:r1|2", (0, 1), expected=(1, 0))

    assert (root / "labels.csv").read_bytes() == (FIXTURES / "labels_extras_edited.csv").read_bytes()
    assert (FIXTURES / "labels_extras_edited.csv").read_bytes() == (
        b"cell,brdu,edu,round,note\n"
        b"aa:other|4,0,1,3.0,\n"
        b"tiny:r1|1,1,1,,\n"
        b"tiny:r1|2,0,1,1.0,kept\n"
        b"zz:other|9,1,1,2.0,outside index\n"
    )


@pytest.mark.parametrize(
    "content",
    [
        "cell,brdu\nx|1,1\n",
        "cell,edu\nx|1,1\n",
        "brdu,edu\n1,1\n",
        "cell,brdu,edu\nx|1,1,0\nx|1,0,1\n",
        "cell,brdu,edu\n,1,0\n",
        "cell,brdu,edu\nx|1,2,0\n",
        "cell,brdu,edu\nx|1,1,-1\n",
        "cell,brdu,edu\nx|1,1,\n",
        "cell,brdu,edu\nx|1,yes,0\n",
    ],
    ids=["no-edu", "no-brdu", "no-cell", "duplicate", "missing-cell", "two", "negative", "empty", "text"],
)
def test_malformed_labels_files_are_rejected_without_rewriting(tmp_path, content):
    path = tmp_path / "labels.csv"
    path.write_text(content)

    with pytest.raises(ValueError):
        load_labels(path)
    with pytest.raises(ValueError):
        _annotations((path,), path)
    with pytest.raises(ValueError):
        _update(path, "y|2", (1, 0))
    assert path.read_text() == content


@pytest.mark.parametrize("value", [(2, 0), (0, -1), (1,), (1, 0, 1)])
def test_non_binary_values_are_rejected_before_writing(tmp_path, value):
    path = tmp_path / "labels.csv"
    path.write_bytes(b"cell,brdu,edu\nx|1,1,0\n")

    with pytest.raises(ValueError):
        _update(path, "x|1", value)
    assert path.read_bytes() == b"cell,brdu,edu\nx|1,1,0\n"


def test_stale_expected_value_is_a_conflict_and_cross_file_duplicates_fail(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_bytes(b"cell,brdu,edu\nx|1,1,0\n")
    with pytest.raises(LabelConflictError):
        _update(path, "x|1", (0, 0), expected=(0, 1))
    with pytest.raises(LabelConflictError):
        _update(path, "x|2", (0, 0), expected=(0, 1))
    assert path.read_bytes() == b"cell,brdu,edu\nx|1,1,0\n"

    other = tmp_path / "other" / "labels.csv"
    other.parent.mkdir()
    other.write_bytes(b"cell,brdu,edu\nx|1,0,0\n")
    with pytest.raises(ValueError, match="multiple files"):
        _annotations((path, other), other)


# ---------------------------------------------------------------------------
# Frozen legacy barrage.
# ---------------------------------------------------------------------------


def test_frozen_barrage_opens_with_todays_identity_and_rosters(tmp_path):
    root = copy_barrage(tmp_path)
    source = BarrageSource(root)
    digest = hashlib.sha256((root / "complete.json").read_bytes()).hexdigest()[:16]

    assert source.target_id == f"barrage:{root}"
    assert source.source_id == f"barrage:{root}:{digest}"
    assert source.display_name == "barrage"
    assert source.crop_markers == ("combined", "edu", "brdu")
    assert source.annotation_paths == (root / "labels.csv",)
    assert source.default_annotation_path == root / "labels.csv"
    assert source.subsets == ("All labeled",)
    assert source.datasets == ("All datasets", "tiny")
    assert source.review_rows(dataset="All datasets").cell.tolist() == INDEX_CELLS
    assert source.review_rows(dataset="tiny").cell.tolist() == INDEX_CELLS
    with pytest.raises(ValueError):
        source.review_rows(dataset="other")

    annotations = _barrage_annotations(source)
    assert _saved_values(annotations) == {"tiny:r1|1": (0, 1), "tiny:r1|2": (1, 0), "tiny:r1|3": (1, 1)}
    assert annotations.destination("tiny:r1|1") == root / "labels.csv"
    rows = source.correction_rows(
        _saved_values(annotations), identity=(0, 1), subset="All labeled", dataset="All datasets",
    )
    assert rows.cell.tolist() == ["tiny:r1|1"]


def test_frozen_barrage_contract_files_are_unchanged():
    run = json.loads((LEGACY_BARRAGE / "run.json").read_text())
    assert (LEGACY_BARRAGE / "complete.json").read_bytes() == (LEGACY_BARRAGE / "run.json").read_bytes()
    assert list(run) == [
        "artifact", "candidate_cells", "cmap", "display_max", "display_min", "display_overrides",
        "display_percentile", "figure_dpi", "n_bins", "n_cells", "renderer", "seed", "size", "source_sha256",
    ]
    assert run["artifact"] == "dino-zarr-barrage-v3"
    assert run["display_max"] == {"brdu": 932.0, "edu": 1308.0, "pi": 1376.0}
    index = pd.read_csv(LEGACY_BARRAGE / "index.csv", dtype={"cell": str})
    assert index.columns.tolist() == [
        "rank", "cell", "dataset", "roi", "label", "x", "y", "z", "plane_z", "area", "plane_area",
        "thumb_shape", "image_path", "source_sha256",
        "brdu_mean", "brdu_min", "brdu_max", "brdu_median", "brdu_std",
        "edu_mean", "edu_min", "edu_max", "edu_median", "edu_std",
        "brdu_image_path", "edu_image_path", "combined_image_path",
    ]
    assert (index.source_sha256 == run["source_sha256"]).all()


@pytest.mark.parametrize("marker", ["brdu", "edu", "combined"])
def test_frozen_barrage_returns_each_markers_crop_file(tmp_path, marker):
    root = copy_barrage(tmp_path)
    source = BarrageSource(root)
    index = pd.read_csv(root / "index.csv", dtype={"cell": str})

    for row in index.itertuples(index=False):
        crop = _crop(source, row.cell, marker)
        assert crop.image == (root / getattr(row, f"{marker}_image_path")).read_bytes()
        assert crop.title == f"Cell {row.label}"
        assert crop.detail == f"tiny · ROI r1 · rank {row.rank + 1}"
    with pytest.raises(KeyError):
        _crop(source, "tiny:r1|99", marker)


def test_frozen_barrage_rejects_unknown_crop_markers(tmp_path):
    source = BarrageSource(copy_barrage(tmp_path))
    with pytest.raises(ValueError):
        _crop(source, "tiny:r1|1", "pi")


def test_barrage_with_only_image_path_serves_it_for_every_marker(tmp_path):
    root = copy_barrage(tmp_path)
    index = pd.read_csv(root / "index.csv", dtype={"cell": str})
    index.drop(columns=["brdu_image_path", "edu_image_path", "combined_image_path"]).to_csv(
        root / "index.csv", index=False,
    )
    source = BarrageSource(root)

    assert source.crop_markers == ("combined", "edu", "brdu")
    for marker in ("brdu", "edu", "combined"):
        assert _crop(source, "tiny:r1|2", marker).image == (root / index.image_path[1]).read_bytes()


def build_corrected_barrage(tmp_path: Path, monkeypatch) -> tuple[Path, list[tuple[tuple, str]]]:
    """Open the frozen barrage as a corrected display with an identity pixel correction.

    The display model and Zarr reads are stubbed: each cell's crop is the frozen
    CellCrop of its rank, so only the source's marker dispatch and rendering run.
    """
    from dinoapp import corrected_barrage

    root = copy_barrage(tmp_path)
    source_dir = tmp_path / "corrected-source"
    source_dir.mkdir()
    index = pd.read_csv(root / "index.csv", dtype={"cell": str})
    index.rename(columns={"y": "centroid_y", "x": "centroid_x"})[
        ["label", "plane_z", "centroid_y", "centroid_x"]
    ].to_parquet(source_dir / "cells.parquet", index=False)
    _write_json(
        {"schema": "dino-corrected-display-v1", "model": str(tmp_path / "model.pkl"), "source": str(source_dir)},
        root / "display.json",
    )
    corrections: list[tuple[tuple, str]] = []

    class IdentityDisplay:
        identity = "stub-identity"
        description = "stub correction"

        def __init__(self, model: Path, source: Path):
            assert source == source_dir
            self.source = SimpleNamespace(
                labels=SimpleNamespace(chunks=(1, 16, 16), dtype=np.dtype("u4")),
                channels={}, offset_zyx=(0, 0, 0),
            )

        def crop(self, crop, origin, *, reference):
            corrections.append((origin, reference))
            return crop

    def read_frozen_crop(reader, row, *, size):
        assert size == 12
        return frozen_crop(int(row["label"]) - 1)

    monkeypatch.setattr(corrected_barrage, "DisplayPixels", IdentityDisplay)
    monkeypatch.setattr(corrected_barrage, "CellCropReader", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(corrected_barrage, "read_crop", read_frozen_crop)
    return root, corrections


def test_corrected_barrage_serves_rendered_crops_for_each_marker(tmp_path, monkeypatch):
    from dinoapp.corrected_barrage import CorrectedBarrageSource

    root, corrections = build_corrected_barrage(tmp_path, monkeypatch)
    source = CorrectedBarrageSource(root)
    digest = hashlib.sha256((root / "complete.json").read_bytes()).hexdigest()[:16]

    assert source.source_id == f"barrage:{root}:{digest}:display:stub-identity"
    assert source.display_name == "barrage · stub correction"
    assert source.crop_markers == ("combined", "edu", "brdu")
    index = pd.read_csv(root / "index.csv", dtype={"cell": str})
    for row in index.itertuples(index=False):
        for marker, column in (("brdu", "brdu_image_path"), ("edu", "edu_image_path"), ("combined", "image_path")):
            crop = _crop(source, row.cell, marker)
            expected = np.asarray(Image.open(root / getattr(row, column)))
            np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(crop.image))), expected)
            assert crop.title == f"Cell {row.label}"
            assert crop.detail == "stub correction"
    assert sorted(set(corrections)) == [((0, 2, 2), "pi"), ((0, 2, 16), "pi"), ((0, 16, 8), "pi")]
    with pytest.raises(KeyError):
        _crop(source, "tiny:r1|99", "brdu")
    with pytest.raises(ValueError):
        _crop(source, "tiny:r1|1", "pi")


@pytest.mark.parametrize("marker", ["brdu", "edu"])
def test_marker_crop_rendering_reproduces_the_frozen_barrage_crops(marker):
    display_max = json.loads((LEGACY_BARRAGE / "complete.json").read_text())["display_max"][marker]
    for rank in range(3):
        rendered = _render_marker_crop(frozen_crop(rank), marker, display_max)
        expected = Image.open(LEGACY_BARRAGE / f"images-{marker}" / f"{rank:04d}.png")
        actual = Image.open(io.BytesIO(rendered))
        assert actual.mode == expected.mode == "RGB"
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


# ---------------------------------------------------------------------------
# Change journal and backups.
# ---------------------------------------------------------------------------

EDIT_KEYS = {
    "assign": ["action", "source_id", "dataset_filter", "cell", "previous", "value", "path", "time_utc"],
    "gallery_toggle_edu": ["action", "source_id", "queue_id", "cell", "previous", "value", "path", "time_utc"],
    "gallery_toggle_brdu": ["action", "source_id", "queue_id", "cell", "previous", "value", "path", "time_utc"],
    "undo": ["action", "cell", "previous", "value", "path", "time_utc"],
}


def test_frozen_journal_records_positional_brdu_edu_pairs():
    records = frozen_journal()
    complete = hashlib.sha256((LEGACY_BARRAGE / "complete.json").read_bytes()).hexdigest()[:16]

    assert [record["action"] for record in records] == ["assign", "gallery_toggle_edu", "gallery_toggle_brdu", "undo"]
    assert [list(record) for record in records] == [EDIT_KEYS[record["action"]] for record in records]
    assert [(record["cell"], record["previous"], record["value"]) for record in records] == [
        ("tiny:r1|3", None, [1, 1]),
        ("tiny:r1|1", [0, 0], [0, 1]),
        ("tiny:r1|2", [1, 0], [0, 0]),
        ("tiny:r1|2", [0, 0], [1, 0]),
    ]
    for record in records:
        assert Path(record["path"]).parts[-2:] == ("barrage", "labels.csv")
        assert datetime.fromisoformat(record["time_utc"]).utcoffset().total_seconds() == 0
        if "source_id" in record:
            assert record["source_id"].startswith("barrage:")
            assert record["source_id"].endswith(f"/barrage:{complete}")


def test_frozen_backup_plus_journal_reproduces_the_saved_labels():
    backup = LEGACY_BARRAGE / "cell-review" / f"barrage-{LEGACY_SESSION}.csv"
    assert backup.read_bytes() == b"cell,brdu,edu\ntiny:r1|1,0,0\ntiny:r1|2,1,0\n"
    labels = {row.cell: [row.brdu, row.edu] for row in load_labels(backup).itertuples(index=False)}
    for record in frozen_journal():
        assert labels.get(record["cell"]) == record["previous"]
        if record["value"] is None:
            labels.pop(record["cell"])
        else:
            labels[record["cell"]] = record["value"]
    saved = load_labels(LEGACY_BARRAGE / "labels.csv")
    assert {row.cell: [row.brdu, row.edu] for row in saved.itertuples(index=False)} == labels
    assert (LEGACY_BARRAGE / "labels.csv").read_bytes() == (
        b"cell,brdu,edu\ntiny:r1|1,0,1\ntiny:r1|2,1,0\ntiny:r1|3,1,1\n"
    )


def test_resumed_session_appends_to_the_frozen_journal_and_keeps_its_backup(tmp_path, monkeypatch):
    root = copy_barrage(tmp_path)
    state_dir = root / "cell-review"
    journal = state_dir / f"changes-{LEGACY_SESSION}.jsonl"
    backup = state_dir / f"barrage-{LEGACY_SESSION}.csv"
    frozen_lines = journal.read_bytes()
    frozen_backup = backup.read_bytes()
    source = BarrageSource(root)
    annotations = _barrage_annotations(source)
    context = barrage_context(monkeypatch, source, LEGACY_SESSION)

    current = _saved_values(annotations)["tiny:r1|3"]
    edit = _save(
        annotations, state_dir, context, "tiny:r1|3", _toggle(current, "edu"), expected=current,
        record={"action": "gallery_toggle_edu", "source_id": source.source_id, "queue_id": "q-resume"},
    )

    assert edit.path == root / "labels.csv"
    assert backup.read_bytes() == frozen_backup
    assert sorted(path.name for path in state_dir.iterdir()) == [backup.name, journal.name]
    assert (root / "labels.csv").read_bytes() == b"cell,brdu,edu\ntiny:r1|1,0,1\ntiny:r1|2,1,0\ntiny:r1|3,1,0\n"
    text = journal.read_bytes()
    assert text.startswith(frozen_lines)
    new = json.loads(text[len(frozen_lines):])
    assert list(new) == list(frozen_journal()[1])
    assert {key: new[key] for key in ("action", "source_id", "queue_id", "cell", "previous", "value", "path")} == {
        "action": "gallery_toggle_edu", "source_id": source.source_id, "queue_id": "q-resume",
        "cell": "tiny:r1|3", "previous": [1, 1], "value": [1, 0], "path": str(root / "labels.csv"),
    }


def test_new_session_backs_up_labels_before_its_first_edit_and_journals_undo(tmp_path, monkeypatch):
    root = copy_barrage(tmp_path)
    state_dir = root / "cell-review"
    before = (root / "labels.csv").read_bytes()
    source = BarrageSource(root)
    annotations = _barrage_annotations(source)
    context = barrage_context(monkeypatch, source, "fresh002")

    current = _saved_values(annotations)["tiny:r1|2"]
    _save(
        annotations, state_dir, context, "tiny:r1|2", _toggle(current, "brdu"), expected=current,
        record={"action": "gallery_toggle_brdu", "source_id": source.source_id, "queue_id": "q-new"},
    )
    _save(
        annotations, state_dir, context, "tiny:r1|1", (1, 1), expected=(0, 1),
        record={"action": "assign", "source_id": source.source_id, "dataset_filter": "All datasets"},
    )
    _undo(annotations, context, state_dir, context["history"][-1])

    assert (state_dir / "barrage-fresh002.csv").read_bytes() == before
    assert (root / "labels.csv").read_bytes() == b"cell,brdu,edu\ntiny:r1|1,0,1\ntiny:r1|2,0,0\ntiny:r1|3,1,1\n"
    records = [json.loads(line) for line in (state_dir / "changes-fresh002.jsonl").read_text().splitlines()]
    assert [list(record) for record in records] == [EDIT_KEYS[record["action"]] for record in records]
    assert [(record["action"], record["cell"], record["previous"], record["value"]) for record in records] == [
        ("gallery_toggle_brdu", "tiny:r1|2", [1, 0], [0, 0]),
        ("assign", "tiny:r1|1", [0, 1], [1, 1]),
        ("undo", "tiny:r1|1", [1, 1], [0, 1]),
    ]
    assert context["history"][-1].previous == (1, 0)
    assert context["history"][-1].value == (0, 0)


# ---------------------------------------------------------------------------
# Call encoding and default presentation.
# ---------------------------------------------------------------------------


def test_native_prediction_loading_encodes_calls_as_edu_plus_twice_brdu(tmp_path):
    predictions = pd.DataFrame({
        "label": np.array([1, 2, 4, 5], dtype=np.int64),
        "cell": [f"full-run2|{label}" for label in (1, 2, 4, 5)],
        "brdu": [False, False, True, True],
        "edu": [False, True, False, True],
        "p_brdu": [0.1, 0.2, 0.8, 0.9],
        "p_edu": [0.3, 0.7, 0.2, 0.6],
    })
    calls = _load_native_calls(build_native_root(tmp_path, predictions))

    assert calls.dtype == np.uint8
    np.testing.assert_array_equal(calls, [4, 0, 1, 4, 2, 3])


def test_field_overlay_replaces_predictions_with_saved_labels_in_the_same_encoding():
    predicted = np.array([4, 0, 1, 2, 3], dtype=np.uint8)
    manual = {"full-run2|1": (1, 1), "full-run2|2": (1, 0), "full-run2|9": (0, 1), "other|3": (0, 0)}

    calls = _overlay_calls(np.array([0, 1, 2, 3, 4, 7, 9], dtype=np.uint32), predicted, manual)

    assert calls.dtype == np.uint8
    np.testing.assert_array_equal(calls, [4, 3, 2, 2, 3, 4, 1])


def test_identity_palette_is_neither_edu_brdu_dual_uncalled():
    assert native.PALETTE.dtype == np.float64
    np.testing.assert_array_equal(
        native.PALETTE,
        [[0.25, 0.25, 0.25], [0, 1, 0], [1, 0, 1], [1, 1, 1], [0.6, 0.6, 0.6]],
    )


def _field_inputs():
    rng = np.random.default_rng(7)
    brdu = rng.integers(100, 301, size=(20, 20)).astype(np.uint16)
    edu = rng.integers(50, 451, size=(20, 20)).astype(np.uint16)
    mask = np.zeros((20, 20), dtype=np.uint32)
    for label, x in enumerate((0, 3, 6, 9, 12), start=1):
        mask[1:5, x:x + 3] = label
    contrast = {"brdu": (100.0, 300.0), "edu": (50.0, 450.0)}
    scaled_brdu = (brdu - 100.0) / 200.0
    scaled_edu = (edu - 50.0) / 400.0
    return brdu, edu, mask, contrast, scaled_brdu, scaled_edu


@pytest.mark.parametrize("view", ["brdu", "edu", "merge"])
def test_field_composite_is_brdu_magenta_edu_green(view):
    brdu, edu, mask, contrast, b, e = _field_inputs()
    calls = np.array([4, 0, 1, 2, 3, 4], dtype=np.uint8)

    pixels = _render_field_pixels(brdu=brdu, edu=edu, mask=mask, calls=calls, view=view, contrast=contrast)

    zero = np.zeros_like(b)
    expected = {"brdu": (b, zero, b), "edu": (zero, e, zero), "merge": (b, e, b)}[view]
    assert pixels.shape == (20, 20, 3)
    region = np.s_[:16, :16]
    np.testing.assert_allclose(pixels[region], np.stack(expected, axis=-1)[region] * 255, atol=1)


def test_identity_borders_take_the_palette_color_of_each_call():
    brdu, edu, mask, contrast, b, e = _field_inputs()
    calls = np.array([4, 0, 1, 2, 3, 4], dtype=np.uint8)

    pixels = _render_field_pixels(brdu=brdu, edu=edu, mask=mask, calls=calls, view="identities", contrast=contrast)

    borders = find_boundaries(mask, mode="inner") & (mask > 0)
    expected = np.stack((b, e, b), axis=-1) * 255
    palette = np.array([[63.75, 63.75, 63.75], [0, 255, 0], [255, 0, 255], [255, 255, 255], [153, 153, 153]])
    expected[borders] = palette[calls[mask[borders]]]
    region = np.s_[:16, :16]
    np.testing.assert_allclose(pixels[region], expected[region], atol=1)
    np.testing.assert_allclose(pixels[1, [0, 3, 6, 9, 12]], palette[calls[1:]], atol=1)
