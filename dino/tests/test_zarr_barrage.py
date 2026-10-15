from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import zarr

from dino import zarr_barrage
from dino.zarr_barrage import (
    masked_channel_stats,
    parse_display_overrides,
    stratified_sample,
)
from dino.zarr_source import CellCrop, build_source_manifest, write_json_atomic


def test_masked_channel_stats_excludes_neighbors_without_thresholding() -> None:
    image = np.array([[0, 5, 1000], [7, 9, 2000]], dtype=np.uint16)
    mask = np.array([[True, True, False], [True, True, False]])

    stats = masked_channel_stats(image, mask)

    assert stats == {
        "mean": 5.25,
        "min": 0.0,
        "max": 9.0,
        "median": 6.0,
        "std": np.std([0, 5, 7, 9]),
    }


def test_stratified_sample_is_deterministic_and_bounded() -> None:
    frame = pd.DataFrame(
        {
            "cell": [f"c{i}" for i in range(100)],
            "a": np.arange(100),
            "b": np.arange(100)[::-1],
        }
    )

    first = stratified_sample(frame, columns=("a", "b"), n_rows=20, n_bins=4, seed=3)
    second = stratified_sample(frame, columns=("a", "b"), n_rows=20, n_bins=4, seed=3)

    assert first["cell"].tolist() == second["cell"].tolist()
    assert len(first) == 20
    assert first["cell"].is_unique


def test_render_crop_matches_scaled_legacy_presentation() -> None:
    size = 60
    labels = np.zeros((size, size), dtype=np.uint32)
    labels[20:40, 20:40] = 1
    labels[8:16, 8:16] = 2
    mask = labels == 1
    crop = CellCrop(
        labels=labels,
        mask=mask,
        neighbor_mask=labels == 2,
        channels={
            "pi": np.full((size, size), 4096, dtype=np.uint16),
            "edu": np.full((size, size), 4096, dtype=np.uint16),
            "brdu": np.full((size, size), 4096, dtype=np.uint16),
        },
    )

    image = zarr_barrage.render_crop(
        crop,
        cell="roi|1",
        display_max={"pi": 4096.0, "edu": 8192.0, "brdu": 16_384.0},
    )

    assert image.size == (1440, 480)
    pixels = np.asarray(image)
    panel_brightness = [int(pixels[250, x].sum()) for x in (240, 720, 1200)]
    assert panel_brightness[0] > panel_brightness[1] > panel_brightness[2]


def test_display_max_cache_reuses_matching_source_and_replaces_stale_value(tmp_path, monkeypatch) -> None:
    source = SimpleNamespace(
        manifest={
            "source_sha256": "source-a",
            "labels": {"shape": [4, 20, 20]},
            "offset_zyx": [0, 0, 0],
        }
    )
    measured = {"pi": 100.0, "edu": 200.0, "brdu": 300.0}
    calls = 0

    def measure(_source):
        nonlocal calls
        calls += 1
        return measured, 400

    monkeypatch.setattr(zarr_barrage, "_measure_display_max", measure)

    assert zarr_barrage._load_or_measure_display_max(tmp_path, source) == measured
    assert json.loads((tmp_path / "display.json").read_text()) == {
        "artifact": "dino-zarr-display-v1",
        "source_sha256": "source-a",
        "percentile": 99.9,
        "z_indices": [0, 1, 2, 3],
        "xy_stride": 8,
        "samples_per_channel": 400,
        "vmax": measured,
    }
    assert zarr_barrage._load_or_measure_display_max(tmp_path, source) == measured
    assert calls == 1

    cache_path = tmp_path / "display.json"
    stale = json.loads(cache_path.read_text())
    stale["source_sha256"] = "source-b"
    cache_path.write_text(json.dumps(stale))

    assert zarr_barrage._load_or_measure_display_max(tmp_path, source) == measured
    assert calls == 2


def test_barrage_writes_existing_labeler_schema(tmp_path, monkeypatch) -> None:
    masked = tmp_path / "dataset" / "roi" / "masked"
    masked.mkdir(parents=True)
    labels_path = masked / "labels.zarr"
    labels = zarr.open_array(labels_path, mode="w", shape=(2, 20, 20), chunks=(1, 10, 10), dtype="u4")
    centers = [(4, 4), (4, 14), (14, 4), (14, 14)]
    for label, (y, x) in enumerate(centers, start=1):
        labels[1, y - 1 : y + 2, x - 1 : x + 2] = label

    channel_paths = {}
    for name, value in (("pi", 100), ("edu", 200), ("brdu", 300)):
        path = tmp_path / f"{name}.zarr"
        array = zarr.open_array(path, mode="w", shape=(2, 30, 30), chunks=(1, 10, 10), dtype="u2")
        array[:] = value
        channel_paths[name] = path

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    manifest = build_source_manifest(
        labels_path=labels_path,
        channel_paths=channel_paths,
        offset_zyx=(0, 5, 5),
    )
    write_json_atomic(manifest, source_dir / "source.json")
    pd.DataFrame(
        [
            {
                "cell": f"roi|{label}",
                "dataset": "dataset",
                "roi": "roi",
                "label": label,
                "area": 9,
                "centroid_z": 1.0,
                "centroid_y": float(y),
                "centroid_x": float(x),
                "plane_z": 1,
                "plane_area": 9,
            }
            for label, (y, x) in enumerate(centers, start=1)
        ]
    ).to_parquet(source_dir / "cells.parquet", index=False)
    eligibility_path = tmp_path / "eligibility.parquet"
    pd.DataFrame(
        {
            "cell": [f"roi|{label}" for label in range(1, 5)],
            "inside_cortex": [True, False, True, False],
        }
    ).to_parquet(eligibility_path, index=False)
    exclude_path = tmp_path / "exclude.csv"
    pd.DataFrame({"cell": ["roi|1"]}).to_csv(exclude_path, index=False)
    out_dir = tmp_path / "barrage"
    monkeypatch.setattr(
        "sys.argv",
        [
            "dino-zarr barrage",
            "--source-dir",
            str(source_dir),
            "--out-dir",
            str(out_dir),
            "--eligibility-table",
            str(eligibility_path),
            "--exclude-index",
            str(exclude_path),
            "--candidate-cells",
            "4",
            "--n-cells",
            "2",
            "--size",
            "60",
            "--cache-gb",
            "0.01",
        ],
    )

    zarr_barrage.main()

    index = pd.read_csv(out_dir / "index.csv")
    required = {
        "rank",
        "cell",
        "image_path",
        "plane_z",
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
    }
    assert required <= set(index.columns)
    assert len(index) == 1
    assert set(index["cell"]) == {"roi|3"}
    assert set(index["brdu_mean"]) == {300.0}
    assert all((out_dir / path).is_file() for path in index["image_path"])


def test_parse_display_overrides() -> None:
    assert parse_display_overrides(["brdu=22199", "edu=20000.5"]) == {
        "brdu": 22199.0,
        "edu": 20000.5,
    }

    with pytest.raises(ValueError, match="Duplicate"):
        parse_display_overrides(["brdu=1", "brdu=2"])
    with pytest.raises(ValueError, match="finite and positive"):
        parse_display_overrides(["brdu=0"])


@pytest.fixture
def barrage_run(tmp_path, monkeypatch):
    labels_path = tmp_path / "dataset" / "roi" / "masked" / "labels.zarr"
    labels = zarr.open_array(labels_path, mode="w", shape=(1, 16, 16), chunks=(1, 8, 8), dtype="u4")
    labels[0, 6:10, 6:10] = 1
    channels = {}
    for name, value in (("pi", 100), ("brdu", 300), ("edu", 200)):
        path = tmp_path / f"{name}.zarr"
        array = zarr.open_array(path, mode="w", shape=(1, 16, 16), chunks=(1, 8, 8), dtype="u2")
        array[:] = value
        channels[name] = path
    source_dir = tmp_path / "source"
    manifest = build_source_manifest(
        labels_path=labels_path, channel_paths=channels, offset_zyx=(0, 0, 0)
    )
    write_json_atomic(manifest, source_dir / "source.json")
    pd.DataFrame([{
        "cell": "roi|1", "dataset": "dataset", "roi": "roi", "label": 1,
        "area": 16, "centroid_z": 0.0, "centroid_y": 8.0, "centroid_x": 8.0,
        "plane_z": 0, "plane_area": 16,
    }]).to_parquet(source_dir / "cells.parquet", index=False)

    def run(out_dir, *extra):
        monkeypatch.setattr("sys.argv", [
            "barrage", "--source-dir", str(source_dir), "--out-dir", str(out_dir),
            "--candidate-cells", "1", "--n-cells", "1", "--size", "8",
            "--cache-gb", "0.001", *extra,
        ])
        zarr_barrage.main()

    return run


def test_interrupted_render_does_not_publish_png(barrage_run, tmp_path, monkeypatch) -> None:
    from PIL import Image

    out_dir = tmp_path / "interrupted"

    def interrupted_save(self, path, *args, **kwargs):
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
        raise OSError("interrupted PNG save")

    monkeypatch.setattr(Image.Image, "save", interrupted_save)
    with pytest.raises(OSError, match="interrupted PNG save"):
        barrage_run(out_dir)
    assert not list(out_dir.glob("images-*/*.png"))
    assert not (out_dir / "complete.json").exists()


def test_partial_resume_rejects_corrupt_png(barrage_run, tmp_path) -> None:
    baseline = tmp_path / "baseline"
    barrage_run(baseline)
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "run.json").write_bytes((baseline / "run.json").read_bytes())
    relative = pd.read_csv(baseline / "index.csv").loc[0, "image_path"]
    image = partial / relative
    image.parent.mkdir()
    corrupt = b"\x89PNG\r\n\x1a\n"
    image.write_bytes(corrupt)

    with pytest.raises(ValueError, match="PNG"):
        barrage_run(partial)

    assert image.read_bytes() == corrupt
    assert not (partial / "complete.json").exists()


def test_reused_index_refreshes_reviewer_paths(barrage_run, tmp_path) -> None:
    from dinoapp.sources import BarrageSource

    baseline = tmp_path / "baseline"
    barrage_run(baseline)
    index = pd.read_csv(baseline / "index.csv")
    for marker in ("combined", "brdu", "edu"):
        index[f"{marker}_image_path"] = f"old_{marker}.png"
    index["review_round"] = 2
    reference = tmp_path / "reference.csv"
    index.to_csv(reference, index=False)
    refreshed = tmp_path / "refreshed"

    barrage_run(refreshed, "--reuse-index", str(reference))

    refreshed_index = pd.read_csv(refreshed / "index.csv")
    assert refreshed_index["review_round"].tolist() == [2]
    reviewer = BarrageSource(refreshed)
    image_bytes = (refreshed / refreshed_index.loc[0, "image_path"]).read_bytes()
    for marker in ("combined", "brdu", "edu"):
        assert reviewer.crop("roi|1", marker).image == image_bytes
    assert not {"combined_image_path", "brdu_image_path", "edu_image_path"} & set(refreshed_index)


@pytest.mark.parametrize("state", ["complete", "unrecognized"])
def test_no_resume_rejects_occupied_barrage_unchanged(barrage_run, tmp_path, state) -> None:
    occupied = tmp_path / "occupied"
    if state == "complete":
        barrage_run(occupied)
    else:
        occupied.mkdir()
        (occupied / "keep.txt").write_text("existing output")
    before = {p.relative_to(occupied): p.read_bytes() for p in occupied.rglob("*") if p.is_file()}

    with pytest.raises(ValueError, match="resume"):
        barrage_run(occupied, "--no-resume")

    assert {p.relative_to(occupied): p.read_bytes() for p in occupied.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("state", ["partial", "complete"])
def test_resume_reuses_valid_published_png(barrage_run, tmp_path, monkeypatch, state) -> None:
    baseline = tmp_path / "baseline"
    barrage_run(baseline)
    out_dir = tmp_path / "resumed"
    out_dir.mkdir()
    (out_dir / "run.json").write_bytes((baseline / "run.json").read_bytes())
    relative = pd.read_csv(baseline / "index.csv").loc[0, "image_path"]
    image = out_dir / relative
    image.parent.mkdir()
    original = (baseline / relative).read_bytes()
    image.write_bytes(original)
    if state == "complete":
        (out_dir / "complete.json").write_bytes((baseline / "complete.json").read_bytes())
        monkeypatch.setattr(
            zarr_barrage.Image, "open", lambda *args: pytest.fail("completed run must skip image audit")
        )
    monkeypatch.setattr(
        zarr_barrage, "render_crop", lambda *args, **kwargs: pytest.fail("published image must be reused")
    )

    barrage_run(out_dir)

    assert image.read_bytes() == original
    assert (out_dir / "complete.json").is_file()
