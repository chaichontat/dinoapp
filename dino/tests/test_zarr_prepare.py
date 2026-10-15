from __future__ import annotations

import os
import sys
import json
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import zarr

from dino import zarr_prepare
from dino.zarr_prepare import measure_block, reduce_measurements


def test_max_area_plane_tie_prefers_centroid_then_lower_z() -> None:
    volume = np.zeros((5, 3, 3), dtype=np.uint32)
    volume[1, :2, :2] = 4
    volume[3, :2, :2] = 4

    moments, planes = measure_block(volume, offset_zyx=(0, 0, 0))
    cells = reduce_measurements(moments=[moments], planes=[planes])

    assert cells.row(0, named=True)["plane_z"] == 1


@pytest.fixture
def preparation_run(tmp_path, monkeypatch):
    masked = tmp_path / "dataset" / "roi" / "masked"
    masked.mkdir(parents=True)
    labels_path = masked / "labels.zarr"
    labels = zarr.open_array(labels_path, mode="w", shape=(4, 6, 6), chunks=(2, 3, 3), dtype="u4")
    expected = np.zeros(labels.shape, dtype=np.uint32)
    expected[1:4, 1:5, 2:5] = 7
    expected[0:2, 4:6, 0:2] = 9
    labels[:] = expected

    volumes = np.zeros(10, dtype=np.uint64)
    unique, counts = np.unique(expected, return_counts=True)
    volumes[unique[unique != 0]] = counts[unique != 0]
    np.save(masked / "volumes.npy", volumes)
    pl.DataFrame({"label": [7, 9], "area": [int(volumes[7]), int(volumes[9])]}).write_parquet(
        masked / "props.parquet"
    )

    channel_paths = {}
    for name in ("pi", "edu", "brdu"):
        path = tmp_path / f"{name}.zarr"
        array = zarr.open_array(path, mode="w", shape=(4, 10, 10), chunks=(1, 4, 4), dtype="u2")
        array[:] = 1
        channel_paths[name] = path
    out_dir = tmp_path / "source"
    monkeypatch.setattr(
        "sys.argv",
        [
            "dino-zarr prepare",
            "--labels",
            str(labels_path),
            "--pi",
            str(channel_paths["pi"]),
            "--edu",
            str(channel_paths["edu"]),
            "--brdu",
            str(channel_paths["brdu"]),
            "--offset-zyx",
            "0",
            "2",
            "3",
            "--out-dir",
            str(out_dir),
            "--workers",
            "1",
        ],
    )

    return SimpleNamespace(
        out_dir=out_dir, expected=expected, channel_paths=channel_paths, labels_path=labels_path
    )


@pytest.fixture
def measured_run(preparation_run, monkeypatch):
    run = preparation_run
    labels = zarr.open_array(run.labels_path, mode="r+")
    labels.attrs["label_volumes"] = {
        "file": "volumes.npy",
        "format": "npy_uint64_indexed_by_final_label",
        "units": "voxels",
        "background_index": 0,
    }
    np.save(run.labels_path / "volumes.npy", np.load(run.labels_path.parent / "volumes.npy"))
    props = pl.DataFrame(
        {
            "label": [7, 9],
            "area": [36, 8],
            "centroid_z": [2.0, 0.5],
            "centroid_y": [2.5, 4.5],
            "centroid_x": [3.0, 0.5],
            "plane_z": [2, 0],
            "plane_area": [12, 4],
        }
    )
    run.props_path = run.out_dir.parent / "measured.parquet"
    props.write_parquet(run.props_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *sys.argv,
            "--measured-props",
            str(run.props_path),
            "--dataset",
            "prod3-2",
            "--roi",
            "recovered",
        ],
    )
    return run


def test_prepare_reuses_postprocessed_measurements_without_scanning_labels(measured_run, monkeypatch):
    def reject_scan(*args, **kwargs):
        pytest.fail("Existing validated measurements must avoid a second label scan")

    monkeypatch.setattr(zarr_prepare, "_measure_source_chunk", reject_scan)
    zarr_prepare.main()
    cells = pl.read_parquet(measured_run.out_dir / "cells.parquet")
    assert cells["cell"].to_list() == ["recovered|7", "recovered|9"]
    assert cells["dataset"].to_list() == ["prod3-2", "prod3-2"]
    for row in cells.iter_rows(named=True):
        coords = np.argwhere(measured_run.expected == row["label"])
        np.testing.assert_array_equal(
            [row["centroid_z"], row["centroid_y"], row["centroid_x"]], coords.mean(axis=0)
        )
    assert not (measured_run.out_dir / "partials").exists()
    before = (measured_run.out_dir / "complete.json").read_bytes()
    assert "measured_props_sha256" in json.loads(before)
    zarr_prepare.main()
    assert (measured_run.out_dir / "complete.json").read_bytes() == before
    props = pl.read_parquet(measured_run.props_path).with_columns(pl.lit(1.0).alias("extra"))
    props.write_parquet(measured_run.props_path)
    with pytest.raises(ValueError, match="properties changed"):
        zarr_prepare.main()


@pytest.mark.parametrize(
    "column,values,message",
    [
        ("label", [7, 7], "every postprocessed label exactly once"),
        ("area", [35, 8], "areas differ"),
        ("centroid_x", [float("nan"), 0.5], "centroids"),
        ("plane_z", [4, 0], "maximum-area planes"),
    ],
)
def test_prepare_rejects_invalid_measured_properties(measured_run, column, values, message):
    props = pl.read_parquet(measured_run.props_path).with_columns(pl.Series(column, values))
    props.write_parquet(measured_run.props_path)
    with pytest.raises(ValueError, match=message):
        zarr_prepare.main()
    assert not (measured_run.out_dir / "complete.json").exists()


@pytest.mark.parametrize(
    "identity_args",
    [
        ["--dataset", "prod3-2"],
        ["--dataset", "prod3-2", "--roi", "invalid|namespace"],
    ],
)
def test_prepare_rejects_incomplete_or_ambiguous_identity(preparation_run, monkeypatch, identity_args):
    monkeypatch.setattr(sys, "argv", [*sys.argv, *identity_args])
    with pytest.raises(ValueError, match="identity|together"):
        zarr_prepare.main()


@pytest.mark.integration
def test_prepare_writes_validated_cell_table(preparation_run) -> None:
    out_dir, expected = preparation_run.out_dir, preparation_run.expected
    zarr_prepare.main()

    cells = pl.read_parquet(out_dir / "cells.parquet")
    assert cells["label"].to_list() == [7, 9]
    assert cells["area"].to_list() == [36, 8]
    row7 = cells.filter(pl.col("label") == 7).row(0, named=True)
    coords7 = np.argwhere(expected == 7)
    assert row7["area"] == len(coords7)
    assert row7["centroid_z"] == np.mean(coords7[:, 0])
    assert row7["centroid_y"] == np.mean(coords7[:, 1])
    assert row7["centroid_x"] == np.mean(coords7[:, 2])
    assert row7["plane_z"] == 2
    assert row7["plane_area"] == 12
    assert (out_dir / "source.json").is_file()
    assert (out_dir / "complete.json").is_file()


@pytest.mark.integration
def test_prepare_resumes_partial_pair_and_rejects_changed_source(preparation_run, monkeypatch):
    run = preparation_run
    real_link = os.link

    def interrupted_link(source, destination):
        if destination.name.startswith("p-"):
            raise OSError("publication interrupted between partial files")
        real_link(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(os, "link", interrupted_link)
        with pytest.raises(OSError, match="publication interrupted"):
            zarr_prepare.main()
    before = {
        path.relative_to(run.out_dir): path.read_bytes() for path in run.out_dir.rglob("*") if path.is_file()
    }
    assert len(list((run.out_dir / "partials").glob("m-*.parquet"))) == 1

    zarr_prepare.main()

    assert pl.read_parquet(run.out_dir / "cells.parquet")["area"].to_list() == [36, 8]
    after = {
        path.relative_to(run.out_dir): path.read_bytes() for path in run.out_dir.rglob("*") if path.is_file()
    }
    assert all(after[path] == value for path, value in before.items())
    zarr_prepare.main()  # Completed preparation validates content and retains all bytes.
    with monkeypatch.context() as patch:
        patch.setattr(sys, "argv", [*sys.argv, "--no-resume"])
        with pytest.raises(FileExistsError, match="occupied destination"):
            zarr_prepare.main()
    channel = zarr.open_array(run.channel_paths["brdu"], mode="r+")
    channel[0, 0, 0] = 2
    with pytest.raises(ValueError, match="different source"):
        zarr_prepare.main()
    assert {
        path.relative_to(run.out_dir): path.read_bytes() for path in run.out_dir.rglob("*") if path.is_file()
    } == after
