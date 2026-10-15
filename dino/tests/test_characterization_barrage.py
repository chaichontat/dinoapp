"""Characterization of direct-Zarr barrages and crop rendering under the BrdU/EdU/PI default.

The frozen barrage under ``fixtures/legacy_barrage`` was written by the code named in its
README. Its source manifest records absolute store roots under ``LEGACY_STORES``, so the
tests materialize the committed stores there before reading the source.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import matplotlib.axes
import matplotlib
import numpy as np
import pandas as pd
import joblib
import pytest
from sklearn.preprocessing import SplineTransformer

from dino import crop_rendering, pixel_corrections, zarr_barrage, zarr_train
from dino.artifacts import file_sha256
from dino.periodic_pixels import PeriodicPixels
from dino.zarr_source import CellCrop, manifest_digest


@pytest.fixture(autouse=True)
def _matplotlib_defaults():
    """Render under the startup rc settings; dino.roc applies a global style."""
    with matplotlib.rc_context():
        matplotlib.rc_file_defaults()
        yield

FIXTURE = Path(__file__).parent / "fixtures" / "legacy_barrage"
LEGACY_STORES = Path("/tmp/dino-wf0-legacy-barrage/stores")
LEGACY_SHA256 = "3d04ff646f0116132d780c5c2709a124c9f17f8288dbde3c3a2edec48f0d4d84"
BARRAGE_ARGS = ("--candidate-cells", "6", "--n-cells", "4", "--size", "16", "--cache-gb", "0.001")

RUN_KEYS = [
    "artifact",
    "candidate_cells",
    "cmap",
    "display_max",
    "display_min",
    "display_overrides",
    "display_percentile",
    "figure_dpi",
    "n_bins",
    "n_cells",
    "renderer",
    "seed",
    "size",
    "source_sha256",
]
INDEX_COLUMNS = [
    "rank",
    "cell",
    "dataset",
    "roi",
    "label",
    "x",
    "y",
    "z",
    "plane_z",
    "area",
    "plane_area",
    "thumb_shape",
    "image_path",
    "source_sha256",
    "brdu_mean",
    "brdu_min",
    "brdu_max",
    "brdu_median",
    "brdu_std",
    "edu_mean",
    "edu_min",
    "edu_max",
    "edu_median",
    "edu_std",
]
FROZEN_CELL_ORDER = ["dataset:roi|4", "dataset:roi|3", "dataset:roi|2", "dataset:roi|1"]
LABELED_FIRST_CELL_ORDER = ["dataset:roi|2", "dataset:roi|4", "dataset:roi|6", "dataset:roi|5"]


# Invocation helpers: later waves update these calls, never the assertions below.


def _run_barrage(source_dir: Path, out_dir: Path, *extra: str) -> None:
    argv = ["dino-zarr barrage", "--source-dir", str(source_dir), "--out-dir", str(out_dir)]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", [*argv, *BARRAGE_ARGS, *extra])
        zarr_barrage.main()


def _parse_display_max(values: list[str]) -> dict[str, float]:
    return zarr_barrage.parse_display_overrides(values)


def _feature_columns() -> tuple[str, ...]:
    return zarr_barrage.FEATURE_COLUMNS


def _render_combined_crop(crop: CellCrop, *, cell: str, display_max: dict[str, float]):
    return crop_rendering.render_crop(crop, cell=cell, display_max=display_max)


def _read_crop(reader, row: pd.Series, *, size: int, pixel_correction):
    return crop_rendering.read_crop(reader, row, size=size, pixel_correction=pixel_correction)


def _correct_channels(channels, origin, *, spatial_subtraction, pixel_correction):
    return pixel_corrections.correct_channels(
        channels,
        origin,
        reference="pi",
        spatial_subtraction=spatial_subtraction,
        pixel_correction=pixel_correction,
    )


def _write_pixel_model(directory: Path, source_sha256: str) -> tuple[Path, dict]:
    """Write a deterministic periodic field and the model bundle that names it."""
    spline = SplineTransformer(
        knots=np.tile(np.linspace(0, 1, 5)[:, None], (1, 2)), extrapolation="periodic"
    ).fit([[0.0, 0.0], [1.0, 1.0]])
    field = directory / "field.joblib"
    joblib.dump(
        dict(
            period_xy_um=np.array([11.0, 17.0]),
            periodic_spline=spline,
            periodic_center=np.full(16, 1 / 16),
            coefficient=np.random.default_rng(7).normal(0, 0.5, (16, 2)),
            targets=["brdu", "edu"],
        ),
        field,
    )
    identity = PeriodicPixels(str(field), file_sha256(field), (1.0, 1.0, 1.0)).identity
    model = directory / "model.joblib"
    joblib.dump(dict(source_sha256=source_sha256, pixel_correction=identity), model)
    return model, identity


def _load_training_rows(barrage_dirs: tuple[Path, ...]) -> pd.DataFrame:
    return zarr_train._load_training_rows_many(barrage_dirs)


def _load_training_sources(barrage_dirs: tuple[Path, ...]) -> dict:
    return zarr_train._load_training_sources_many(barrage_dirs)


# Fixture helpers.


@pytest.fixture(scope="session")
def legacy_stores() -> Path:
    """Place the committed Zarr stores at the absolute roots the frozen manifest records."""
    if not LEGACY_STORES.exists():
        LEGACY_STORES.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".stores-", dir=LEGACY_STORES.parent))
        try:
            shutil.copytree(FIXTURE / "stores", staging / "stores")
            try:
                os.rename(staging / "stores", LEGACY_STORES)
            except OSError:
                if not LEGACY_STORES.is_dir():
                    raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    return LEGACY_STORES


@pytest.fixture
def source_dir(tmp_path, legacy_stores) -> Path:
    copied = tmp_path / "source"
    shutil.copytree(FIXTURE / "source", copied)
    return copied


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _snapshot(root: Path) -> dict[Path, bytes]:
    return {path.relative_to(root): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def _assert_fresh_records(out_dir: Path, expected: dict) -> None:
    """A freshly written run.json keeps every legacy key and value, and complete.json equals it.

    The plan adds the marker contract to new barrage records and versions them, so new keys
    and the ``artifact`` value are free; legacy records are pinned exactly elsewhere.
    """
    run = _read_json(out_dir / "run.json")
    assert {key: run.get(key) for key in expected if key != "artifact"} == {
        key: value for key, value in expected.items() if key != "artifact"
    }
    assert _read_json(out_dir / "complete.json") == run


# Frozen artifact contract.


def test_frozen_barrage_records_v3_contract() -> None:
    run = _read_json(FIXTURE / "barrage" / "run.json")
    source = _read_json(FIXTURE / "barrage" / "source.json")
    display = _read_json(FIXTURE / "source" / "display.json")

    assert sorted(run) == RUN_KEYS
    assert run == {
        "artifact": "dino-zarr-barrage-v3",
        "candidate_cells": 6,
        "cmap": "magma",
        "display_max": {"brdu": 991.0, "edu": 588.0, "pi": 1295.0},
        "display_min": 0.0,
        "display_overrides": {},
        "display_percentile": 99.9,
        "figure_dpi": 200,
        "n_bins": 6,
        "n_cells": 4,
        "renderer": "v3",
        "seed": 0,
        "size": 16,
        "source_sha256": LEGACY_SHA256,
    }
    assert _read_json(FIXTURE / "barrage" / "complete.json") == run
    assert source["source_sha256"] == manifest_digest(source) == LEGACY_SHA256
    assert sorted(source["channels"]) == ["brdu", "edu", "pi"]
    assert display == {
        "artifact": "dino-zarr-display-v1",
        "percentile": 99.9,
        "samples_per_channel": 32,
        "source_sha256": LEGACY_SHA256,
        "vmax": {"brdu": 991.0, "edu": 588.0, "pi": 1295.0},
        "xy_stride": 8,
        "z_indices": [0, 1],
    }


def test_frozen_index_and_labels_layout() -> None:
    index = pd.read_csv(FIXTURE / "barrage" / "index.csv")

    assert index.columns.tolist() == INDEX_COLUMNS
    assert index["cell"].tolist() == FROZEN_CELL_ORDER
    assert index["rank"].tolist() == [0, 1, 2, 3]
    assert index["image_path"].tolist() == [
        "images-v3-16/0000_dataset_roi__4.png",
        "images-v3-16/0001_dataset_roi__3.png",
        "images-v3-16/0002_dataset_roi__2.png",
        "images-v3-16/0003_dataset_roi__1.png",
    ]
    assert set(index["thumb_shape"]) == {"(16, 16, 3)"}
    assert set(index["source_sha256"]) == {LEGACY_SHA256}
    assert (FIXTURE / "barrage" / "labels.csv").read_bytes() == (
        b"cell,brdu,edu\ndataset:roi|2,1,0\ndataset:roi|4,0,1\n"
    )


def test_feature_columns_used_for_sampling() -> None:
    assert _feature_columns() == ("brdu_mean", "brdu_std", "edu_mean", "edu_std")


# Regeneration with today's code reproduces the frozen artifacts.


@pytest.mark.parametrize("display_cache", ["frozen", "measured"])
def test_fresh_barrage_reproduces_frozen_artifacts(source_dir, tmp_path, capsys, display_cache) -> None:
    if display_cache == "measured":
        (source_dir / "display.json").unlink()
    out_dir = tmp_path / "barrage"

    _run_barrage(source_dir, out_dir)

    frozen = FIXTURE / "barrage"
    assert capsys.readouterr().out.splitlines()[-1] == "status=complete"
    _assert_fresh_records(out_dir, _read_json(frozen / "run.json"))
    for name in ("source.json", "index.csv"):
        assert (out_dir / name).read_bytes() == (frozen / name).read_bytes(), name
    assert (source_dir / "display.json").read_bytes() == (FIXTURE / "source" / "display.json").read_bytes()
    assert sorted(path.name for path in (out_dir / "images-v3-16").iterdir()) == sorted(
        path.name for path in (frozen / "images-v3-16").iterdir()
    )
    candidates = pd.read_parquet(out_dir / "candidates.parquet")
    pd.testing.assert_frame_equal(candidates, pd.read_parquet(frozen / "candidates.parquet"))
    assert candidates.columns.tolist() == [
        "cell",
        "dataset",
        "roi",
        "label",
        "area",
        "centroid_z",
        "centroid_y",
        "centroid_x",
        "plane_z",
        "plane_area",
        "brdu_mean",
        "brdu_min",
        "brdu_max",
        "brdu_median",
        "brdu_std",
        "edu_mean",
        "edu_min",
        "edu_max",
        "edu_median",
        "edu_std",
    ]


def test_fresh_index_dtypes(source_dir, tmp_path) -> None:
    out_dir = tmp_path / "barrage"

    _run_barrage(source_dir, out_dir)

    dtypes = pd.read_csv(out_dir / "index.csv").dtypes.astype(str).to_dict()
    assert dtypes == {
        **dict.fromkeys(["rank", "label", "plane_z", "area", "plane_area"], "int64"),
        **dict.fromkeys(["cell", "dataset", "roi", "thumb_shape", "image_path", "source_sha256"], "object"),
        **dict.fromkeys(
            ["x", "y", "z", *(f"{m}_{s}" for m in ("brdu", "edu") for s in ("mean", "min", "max", "median", "std"))],
            "float64",
        ),
    }


def test_display_max_override_is_recorded_per_channel(source_dir, tmp_path) -> None:
    out_dir = tmp_path / "barrage"

    _run_barrage(source_dir, out_dir, "--display-max", "brdu=5000", "--display-max", "pi=2000.5")

    expected = _read_json(FIXTURE / "barrage" / "run.json")
    expected["display_max"] = {"brdu": 5000.0, "edu": 588.0, "pi": 2000.5}
    expected["display_overrides"] = {"brdu": 5000.0, "pi": 2000.5}
    _assert_fresh_records(out_dir, expected)
    assert (source_dir / "display.json").read_bytes() == (FIXTURE / "source" / "display.json").read_bytes()


def test_display_max_accepts_exactly_reference_and_marker_keys() -> None:
    assert _parse_display_max(["pi=1", "edu=2", "brdu=3"]) == {"pi": 1.0, "edu": 2.0, "brdu": 3.0}
    for rejected in ("dapi=1", "BrdU=1", "brdu"):
        with pytest.raises(ValueError):
            _parse_display_max([rejected])


# Today's readers accept the frozen barrage.


def test_completed_legacy_barrage_resumes_as_skipped(source_dir, tmp_path, capsys) -> None:
    out_dir = tmp_path / "barrage"
    shutil.copytree(FIXTURE / "barrage", out_dir)
    before = _snapshot(out_dir)

    _run_barrage(source_dir, out_dir)

    assert capsys.readouterr().out.splitlines()[-1] == "status=skipped_complete"
    assert _snapshot(out_dir) == before


@pytest.mark.parametrize("seed", ["labels_only", "legacy_partial_run"])
def test_rebuild_preserves_legacy_annotations(source_dir, tmp_path, capsys, seed) -> None:
    out_dir = tmp_path / "barrage"
    out_dir.mkdir()
    frozen = FIXTURE / "barrage"
    shutil.copy2(frozen / "labels.csv", out_dir / "labels.csv")
    if seed == "legacy_partial_run":
        shutil.copy2(frozen / "run.json", out_dir / "run.json")
        shutil.copytree(frozen / "images-v3-16", out_dir / "images-v3-16")

    _run_barrage(source_dir, out_dir)

    assert capsys.readouterr().out.splitlines()[-1] == "status=complete"
    assert (out_dir / "labels.csv").read_bytes() == (frozen / "labels.csv").read_bytes()
    _assert_fresh_records(out_dir, _read_json(frozen / "run.json"))
    index = pd.read_csv(out_dir / "index.csv")
    assert index.columns.tolist() == INDEX_COLUMNS
    assert index["cell"].tolist() == LABELED_FIRST_CELL_ORDER
    measured = INDEX_COLUMNS[INDEX_COLUMNS.index("brdu_mean") :]
    frozen_candidates = pd.read_parquet(frozen / "candidates.parquet").set_index("cell")
    pd.testing.assert_frame_equal(
        index.set_index("cell")[measured],
        frozen_candidates.loc[LABELED_FIRST_CELL_ORDER, measured],
        check_names=False,
    )
    assert all((out_dir / path).is_file() for path in index["image_path"])


def test_reuse_index_rebuild_reproduces_frozen_order(source_dir, tmp_path) -> None:
    frozen = FIXTURE / "barrage"
    reference = tmp_path / "reference.csv"
    shutil.copy2(frozen / "index.csv", reference)
    out_dir = tmp_path / "barrage"
    out_dir.mkdir()
    shutil.copy2(frozen / "labels.csv", out_dir / "labels.csv")

    _run_barrage(source_dir, out_dir, "--reuse-index", str(reference))

    assert (out_dir / "index.csv").read_bytes() == (frozen / "index.csv").read_bytes()
    assert (out_dir / "labels.csv").read_bytes() == (frozen / "labels.csv").read_bytes()
    _assert_fresh_records(
        out_dir,
        {
            **_read_json(frozen / "run.json"),
            "reuse_index": str(reference.resolve()),
            "reuse_index_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        },
    )


def test_pixel_model_is_recorded_and_corrects_markers_in_target_order(source_dir, tmp_path) -> None:
    model, identity = _write_pixel_model(tmp_path, LEGACY_SHA256)
    out_dir = tmp_path / "barrage"

    _run_barrage(source_dir, out_dir, "--pixel-model", str(model))

    assert sorted(identity) == ["model", "model_sha256", "operation", "scale_zyx_um", "xy_origin_um"]
    assert identity["operation"] == "float32(raw) * float32(exp(-shared_xy_log_gain))"
    assert identity["model_sha256"] == file_sha256(Path(identity["model"]))
    _assert_fresh_records(out_dir, {**_read_json(FIXTURE / "barrage" / "run.json"), "pixel_correction": identity})
    index = pd.read_csv(out_dir / "index.csv")
    assert index.columns.tolist() == INDEX_COLUMNS
    assert index["cell"].tolist() == FROZEN_CELL_ORDER
    np.testing.assert_allclose(
        index["brdu_mean"], [1102.014316558838, 538.8288288116455, 999.7452583312988, 727.5512428283691], rtol=1e-6
    )
    np.testing.assert_allclose(
        index["edu_mean"], [563.9515724182129, 663.6166191101074, 334.112642288208, 610.7699928283691], rtol=1e-6
    )


def test_reuse_index_keeps_reference_columns_but_not_image_paths(source_dir, tmp_path) -> None:
    frozen = FIXTURE / "barrage"
    reference_frame = pd.read_csv(frozen / "index.csv")
    reference_frame["review_round"] = [3, 1, 4, 1]
    reference_frame["brdu_image_path"] = "stale/brdu.png"
    reference_frame["native_bin"] = ["a", "b", "c", "d"]
    reference_frame["edu_image_path"] = "stale/edu.png"
    reference_frame["combined_image_path"] = "stale/combined.png"
    reference = tmp_path / "reference.csv"
    reference_frame.to_csv(reference, index=False)
    out_dir = tmp_path / "barrage"

    _run_barrage(source_dir, out_dir, "--reuse-index", str(reference))

    index = pd.read_csv(out_dir / "index.csv")
    assert index.columns.tolist() == [*INDEX_COLUMNS, "review_round", "native_bin"]
    assert index["cell"].tolist() == FROZEN_CELL_ORDER
    assert index["review_round"].tolist() == [3, 1, 4, 1]
    assert index["native_bin"].tolist() == ["a", "b", "c", "d"]
    pd.testing.assert_frame_equal(index[INDEX_COLUMNS], pd.read_csv(frozen / "index.csv"))


@pytest.mark.parametrize(
    ("record", "extra"),
    [("complete.json", ("--size", "20")), ("run.json", ("--seed", "1"))],
)
def test_legacy_contract_mismatch_is_rejected(source_dir, tmp_path, record, extra) -> None:
    out_dir = tmp_path / "barrage"
    out_dir.mkdir()
    shutil.copy2(FIXTURE / "barrage" / record, out_dir / record)
    shutil.copy2(FIXTURE / "barrage" / "labels.csv", out_dir / "labels.csv")
    before = _snapshot(out_dir)

    with pytest.raises(ValueError):
        _run_barrage(source_dir, out_dir, *extra)

    assert _snapshot(out_dir) == before


def test_training_loader_accepts_legacy_barrage(tmp_path, legacy_stores) -> None:
    barrage = tmp_path / "barrage"
    shutil.copytree(FIXTURE / "barrage", barrage)

    rows = _load_training_rows((barrage,))
    sources = _load_training_sources((barrage,))

    assert rows.columns.tolist() == [
        "cell",
        "brdu",
        "edu",
        *INDEX_COLUMNS[: INDEX_COLUMNS.index("cell")],
        *INDEX_COLUMNS[INDEX_COLUMNS.index("cell") + 1 :],
        "centroid_x",
        "centroid_y",
        "centroid_z",
    ]
    assert rows["cell"].tolist() == ["dataset:roi|2", "dataset:roi|4"]
    assert rows["brdu"].tolist() == [1, 0]
    assert rows["edu"].tolist() == [0, 1]
    assert rows["brdu"].dtype == np.int64 and rows["edu"].dtype == np.int64
    assert rows["centroid_x"].tolist() == rows["x"].tolist() == [15.5, 8.5]
    assert rows["centroid_y"].tolist() == rows["y"].tolist() == [4.5, 15.5]
    assert rows["centroid_z"].tolist() == rows["z"].tolist() == [1.0, 1.0]
    assert set(rows["source_sha256"]) == {LEGACY_SHA256}
    assert list(sources) == [LEGACY_SHA256]
    assert set(sources[LEGACY_SHA256].channels) == {"pi", "edu", "brdu"}


# Crop rendering.


def test_combined_crop_panels_are_pi_edu_brdu(monkeypatch) -> None:
    size = 16
    labels = np.zeros((size, size), dtype=np.uint32)
    labels[5:11, 5:11] = 1
    labels[0:3, 0:3] = 2
    channels = {
        "pi": np.full((size, size), 10, dtype=np.uint16),
        "edu": np.full((size, size), 20, dtype=np.uint16),
        "brdu": np.full((size, size), 30, dtype=np.uint16),
    }
    crop = CellCrop(labels=labels, mask=labels == 1, neighbor_mask=labels == 2, channels=channels)
    titles: list[str] = []
    shown: list[tuple[int, float]] = []
    original_title = matplotlib.axes.Axes.set_title
    original_imshow = matplotlib.axes.Axes.imshow

    def record_title(self, label, *args, **kwargs):
        titles.append(label)
        return original_title(self, label, *args, **kwargs)

    def record_imshow(self, data, *args, **kwargs):
        shown.append((int(np.asarray(data).flat[0]), kwargs["vmax"]))
        return original_imshow(self, data, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", record_title)
    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", record_imshow)

    image = _render_combined_crop(crop, cell="dataset:roi|1", display_max={"pi": 100.0, "edu": 200.0, "brdu": 300.0})

    assert titles == [
        "cell=dataset:roi|1\nch=PI mean=10.0",
        "cell=dataset:roi|1\nch=EdU mean=20.0",
        "cell=dataset:roi|1\nch=BrdU mean=30.0",
    ]
    assert shown == [(10, 100.0), (20, 200.0), (30, 300.0)]
    assert image.size == (384, 128)
    assert image.mode == "RGB"


def test_read_crop_corrects_only_marker_channels_at_crop_origin() -> None:
    size = 16
    raw = {name: np.full((size, size), value, dtype=np.float32) for name, value in (("pi", 1), ("edu", 2), ("brdu", 3))}
    mask = np.zeros((size, size), dtype=bool)
    mask[6:10, 6:10] = True
    requests: list[dict] = []

    def read_cell(**kwargs):
        requests.append(kwargs)
        return CellCrop(labels=mask.astype(np.uint32), mask=mask, neighbor_mask=~mask, channels=dict(raw))

    reader = SimpleNamespace(read_cell=read_cell, offset_zyx=(2, 30, 40))
    corrections: list[tuple[str, tuple[int, int, int]]] = []

    def pixel_correction(name, pixels, origin):
        corrections.append((name, origin))
        return pixels + 100

    row = pd.Series({"label": 7, "plane_z": 3, "centroid_y": 20.4, "centroid_x": 11.6})

    crop = _read_crop(reader, row, size=size, pixel_correction=pixel_correction)

    assert requests == [{"label": 7, "plane_z": 3, "center_y": 20, "center_x": 12, "size": size}]
    assert sorted(corrections) == [("brdu", (5, 42, 44)), ("edu", (5, 42, 44))]
    assert {name: float(pixels[0, 0]) for name, pixels in crop.channels.items()} == {
        "pi": 1.0,
        "edu": 102.0,
        "brdu": 103.0,
    }


def test_correct_channels_subtracts_brdu_then_corrects_markers() -> None:
    channels = {name: np.full((2, 2), value, dtype=np.float32) for name, value in (("pi", 5), ("edu", 7), ("brdu", 11))}
    calls: list[tuple[str, ...]] = []

    def spatial_subtraction(target, reference, origin):
        calls.append(("subtract", float(target[0, 0]), float(reference[0, 0]), origin))
        return target - reference

    def pixel_correction(name, pixels, origin):
        calls.append(("correct", name, float(pixels[0, 0]), origin))
        return pixels * 10

    corrected = _correct_channels(
        channels, (1, 2, 3), spatial_subtraction=spatial_subtraction, pixel_correction=pixel_correction
    )

    assert calls == [
        ("subtract", 11.0, 5.0, (1, 2, 3)),
        ("correct", "edu", 7.0, (1, 2, 3)),
        ("correct", "brdu", 6.0, (1, 2, 3)),
    ]
    assert {name: float(pixels[0, 0]) for name, pixels in corrected.items()} == {"pi": 5.0, "edu": 70.0, "brdu": 60.0}
    assert float(channels["brdu"][0, 0]) == 11.0
