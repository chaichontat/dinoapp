from __future__ import annotations

import json
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import zarr

from dino import zarr_prepare
from dino.artifacts import file_sha256
from dino.zarr_prepare import measure_block, reduce_measurements
from dino_markers import (
    DEFAULT_CONTRACT,
    ContractMismatchError,
    MarkerConfigError,
    MissingContractError,
    config_to_json,
    contract_of,
    default_config,
    load_config,
    parse_config,
)

DEFAULT_KEYS = DEFAULT_CONTRACT.channel_keys


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
    for name in ("pi", "edu", "brdu", "dapi", "pcna"):
        path = tmp_path / f"{name}.zarr"
        array = zarr.open_array(path, mode="w", shape=(4, 10, 10), chunks=(1, 4, 4), dtype="u2")
        array[:] = 1
        channel_paths[name] = path
    out_dir = tmp_path / "source"

    def argv(keys=DEFAULT_KEYS, *extra, out=None):
        return [
            "dino-zarr prepare",
            "--labels",
            str(labels_path),
            *(arg for key in keys for arg in ("--channel", f"{key}={channel_paths[key]}")),
            "--offset-zyx",
            "0",
            "2",
            "3",
            "--out-dir",
            str(out or out_dir),
            "--workers",
            "1",
            *extra,
        ]

    monkeypatch.setattr("sys.argv", argv())

    return SimpleNamespace(
        out_dir=out_dir,
        expected=expected,
        channel_paths=channel_paths,
        labels_path=labels_path,
        argv=argv,
        tmp_path=tmp_path,
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
    assert json.loads(before)["artifact"] == "dino-zarr-source-complete-v2"
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


THREE_MARKER_KEYS = ("dapi", "edu", "brdu", "pcna")
LABELS = {"pi": "PI", "dapi": "DAPI", "brdu": "BrdU", "edu": "EdU", "pcna": "PCNA"}


def write_markers(path, reference, markers, *, colors=("#ff0000", "#00ff00", "#0000ff")):
    """Write a valid markers.json for the given contract and return its path."""
    value = {
        "version": 1,
        "reference": {"key": reference, "label": LABELS[reference]},
        "markers": [
            {"key": key, "label": LABELS[key], "color": color} for key, color in zip(markers, colors)
        ],
        "presentation": {
            "crop_panels": [reference, *reversed(markers)],
            "negative_color": "#404040",
            "uncalled_color": "#999999",
        },
    }
    parse_config(value, source=str(path))
    path.write_text(json.dumps(value))
    return path


def tree_bytes(directory):
    return {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


def run_main(run, monkeypatch, keys=DEFAULT_KEYS, *extra, out=None):
    monkeypatch.setattr(sys, "argv", run.argv(keys, *extra, out=out))
    zarr_prepare.main()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def make_legacy(out_dir):
    """Turn a completed default output into the form older code wrote: v1 completion, no markers.json."""
    complete = read_json(out_dir / "complete.json")
    del complete["marker_contract"]
    complete["artifact"] = "dino-zarr-source-complete-v1"
    write_json(out_dir / "complete.json", complete)
    (out_dir / "markers.json").unlink()


@pytest.mark.integration
def test_default_prepare_records_contract_and_stores_the_default_config(preparation_run) -> None:
    out_dir = preparation_run.out_dir
    zarr_prepare.main()

    source = read_json(out_dir / "source.json")
    assert read_json(out_dir / "complete.json") == {
        "artifact": "dino-zarr-source-complete-v2",
        "cells": 2,
        "cells_sha256": file_sha256(out_dir / "cells.parquet"),
        "marker_contract": {"version": 1, "reference": "pi", "markers": ["brdu", "edu"]},
        "source_sha256": source["source_sha256"],
    }
    assert read_json(out_dir / "markers.json") == config_to_json(default_config())
    assert "marker_contract" not in source


@pytest.mark.integration
def test_three_marker_prepare_records_contract_and_config(preparation_run, monkeypatch) -> None:
    run = preparation_run
    markers = write_markers(run.tmp_path / "three.json", "dapi", ["edu", "brdu", "pcna"])

    run_main(run, monkeypatch, THREE_MARKER_KEYS, "--markers", str(markers))

    complete = read_json(run.out_dir / "complete.json")
    assert complete["artifact"] == "dino-zarr-source-complete-v2"
    assert complete["marker_contract"] == {"version": 1, "reference": "dapi", "markers": ["edu", "brdu", "pcna"]}
    assert read_json(run.out_dir / "markers.json") == config_to_json(load_config(markers))
    assert list(read_json(run.out_dir / "source.json")["channels"]) == ["brdu", "dapi", "edu", "pcna"]
    assert pl.read_parquet(run.out_dir / "cells.parquet")["label"].to_list() == [7, 9]
    assert contract_of(run.out_dir).markers == ("edu", "brdu", "pcna")


@pytest.mark.integration
def test_source_json_bytes_do_not_depend_on_the_marker_config(preparation_run, monkeypatch) -> None:
    run = preparation_run
    reordered = write_markers(run.tmp_path / "reordered.json", "pi", ["edu", "brdu"], colors=("#123456", "#abcdef"))
    other = run.tmp_path / "other"

    run_main(run, monkeypatch)
    run_main(run, monkeypatch, DEFAULT_KEYS, "--markers", str(reordered), out=other)

    assert (other / "source.json").read_bytes() == (run.out_dir / "source.json").read_bytes()
    assert read_json(other / "complete.json")["source_sha256"] == read_json(run.out_dir / "complete.json")[
        "source_sha256"
    ]
    assert read_json(other / "complete.json")["marker_contract"]["markers"] == ["edu", "brdu"]
    assert read_json(run.out_dir / "complete.json")["marker_contract"]["markers"] == ["brdu", "edu"]


@pytest.mark.parametrize(
    "channels,message",
    [
        (["pi=P", "edu=E"], r"missing \['brdu'\], unexpected \[\]"),
        (["pi=P", "edu=E", "brdu=B", "dapi=D"], r"missing \[\], unexpected \['dapi'\]"),
        (["PI=P", "edu=E", "brdu=B"], r"missing \['pi'\], unexpected \['PI'\]"),
        (["pi=P", "pi=P2", "edu=E", "brdu=B"], "'pi' given more than once"),
        (["pi", "edu=E", "brdu=B"], "expects KEY=PATH, got 'pi'"),
        (["=P", "edu=E", "brdu=B"], "expects KEY=PATH, got '=P'"),
        (["pi=", "edu=E", "brdu=B"], "expects KEY=PATH, got 'pi='"),
    ],
)
def test_prepare_rejects_invalid_channel_arguments(tmp_path, monkeypatch, capsys, channels, message) -> None:
    argv = ["dino-zarr prepare", "--labels", "L", "--offset-zyx", "0", "0", "0", "--out-dir", str(tmp_path / "o")]
    for channel in channels:
        argv += ["--channel", channel]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as raised:
        zarr_prepare.main()

    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert re.search(r"--channel .*" + message, err)
    assert not (tmp_path / "o").exists()


def test_prepare_requires_exactly_the_channels_of_a_custom_contract(preparation_run, monkeypatch, capsys) -> None:
    run = preparation_run
    markers = write_markers(run.tmp_path / "three.json", "dapi", ["edu", "brdu", "pcna"])

    with pytest.raises(SystemExit):
        run_main(run, monkeypatch, DEFAULT_KEYS, "--markers", str(markers))

    assert (
        "--channel keys must be exactly ['dapi', 'edu', 'brdu', 'pcna']: "
        "missing ['dapi', 'pcna'], unexpected ['pi']"
    ) in capsys.readouterr().err
    assert not run.out_dir.exists()


def test_prepare_rejects_an_invalid_markers_file(preparation_run, monkeypatch) -> None:
    run = preparation_run
    bad = run.tmp_path / "bad.json"
    bad.write_text(json.dumps({"version": 1}))

    with pytest.raises(MarkerConfigError, match="missing keys"):
        run_main(run, monkeypatch, DEFAULT_KEYS, "--markers", str(bad))

    assert not run.out_dir.exists()


@pytest.mark.integration
def test_prepare_rejects_reordered_markers_for_a_completed_default_output(preparation_run, monkeypatch) -> None:
    run = preparation_run
    reordered = write_markers(run.tmp_path / "reordered.json", "pi", ["edu", "brdu"])
    zarr_prepare.main()
    before = tree_bytes(run.out_dir)

    with pytest.raises(ContractMismatchError, match="marker order is part of the contract") as raised:
        run_main(run, monkeypatch, DEFAULT_KEYS, "--markers", str(reordered))

    assert str(run.out_dir) in str(raised.value)
    assert str(reordered) in str(raised.value)
    assert tree_bytes(run.out_dir) == before


@pytest.mark.integration
def test_prepare_rejects_a_stored_config_that_disagrees_with_the_completion(preparation_run, monkeypatch) -> None:
    run = preparation_run
    zarr_prepare.main()
    reordered = write_markers(run.out_dir / "markers.json.new", "pi", ["edu", "brdu"])
    reordered.replace(run.out_dir / "markers.json")
    before = tree_bytes(run.out_dir)

    with pytest.raises(ContractMismatchError) as raised:
        zarr_prepare.main()

    assert "complete.json" in str(raised.value)
    assert "markers.json" in str(raised.value)
    assert tree_bytes(run.out_dir) == before


@pytest.mark.integration
def test_prepare_rejects_a_current_completion_without_its_contract(preparation_run) -> None:
    run = preparation_run
    zarr_prepare.main()
    complete = read_json(run.out_dir / "complete.json")
    del complete["marker_contract"]
    write_json(run.out_dir / "complete.json", complete)
    before = tree_bytes(run.out_dir)

    with pytest.raises(MissingContractError, match="complete.json"):
        zarr_prepare.main()

    assert tree_bytes(run.out_dir) == before


@pytest.mark.integration
def test_prepare_skips_a_legacy_completed_output_and_adds_no_config(preparation_run, monkeypatch, capsys) -> None:
    run = preparation_run
    zarr_prepare.main()
    make_legacy(run.out_dir)
    before = tree_bytes(run.out_dir)
    capsys.readouterr()

    zarr_prepare.main()

    assert capsys.readouterr().out.splitlines() == [f"source_dir={run.out_dir}", "status=skipped_complete"]
    assert tree_bytes(run.out_dir) == before
    assert not (run.out_dir / "markers.json").exists()
    default = run.tmp_path / "default.json"
    write_json(default, config_to_json(default_config()))
    run_main(run, monkeypatch, DEFAULT_KEYS, "--markers", str(default))
    assert tree_bytes(run.out_dir) == before


@pytest.mark.integration
def test_prepare_rejects_reordered_markers_for_a_legacy_completed_output(preparation_run, monkeypatch) -> None:
    run = preparation_run
    zarr_prepare.main()
    make_legacy(run.out_dir)
    reordered = write_markers(run.tmp_path / "reordered.json", "pi", ["edu", "brdu"])
    before = tree_bytes(run.out_dir)

    with pytest.raises(ContractMismatchError, match="marker order is part of the contract"):
        run_main(run, monkeypatch, DEFAULT_KEYS, "--markers", str(reordered))

    assert tree_bytes(run.out_dir) == before


@pytest.mark.integration
def test_prepare_completes_a_partial_output_without_versioned_records_as_current(
    preparation_run, monkeypatch
) -> None:
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
    assert not (run.out_dir / "complete.json").exists()
    (run.out_dir / "markers.json").unlink()
    reordered = write_markers(run.tmp_path / "reordered.json", "pi", ["edu", "brdu"])

    run_main(run, monkeypatch, DEFAULT_KEYS, "--markers", str(reordered))

    complete = read_json(run.out_dir / "complete.json")
    assert complete["artifact"] == "dino-zarr-source-complete-v2"
    assert complete["marker_contract"]["markers"] == ["edu", "brdu"]
    assert read_json(run.out_dir / "markers.json") == config_to_json(load_config(reordered))


@pytest.mark.integration
def test_prepare_rejects_a_changed_presentation_for_a_partial_output(preparation_run, monkeypatch) -> None:
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
    recolored = write_markers(run.tmp_path / "recolored.json", "pi", ["brdu", "edu"], colors=("#111111", "#222222"))

    with pytest.raises(MarkerConfigError, match="differs from this run's marker configuration"):
        run_main(run, monkeypatch, DEFAULT_KEYS, "--markers", str(recolored))

    assert not (run.out_dir / "complete.json").exists()
