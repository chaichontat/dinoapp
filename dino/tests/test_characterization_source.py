"""Characterization of prepared direct-Zarr sources under the default BrdU/EdU/PI markers.

The frozen artifact under ``fixtures/legacy_source`` was written by the prepare code at
the commit named in its README. Tests copy it, replace the recorded root prefix, and
check that today's readers and resume path accept it unchanged.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl
import pytest

from dino import zarr_prepare
from dino.artifacts import file_sha256, read_parquet_artifact
from dino.zarr_source import (
    build_source_manifest,
    load_source,
    manifest_digest,
    open_review_source,
    write_json_atomic,
)

FIXTURE = Path(__file__).parent / "fixtures" / "legacy_source"
FROZEN_PREFIX = "/tmp/wf0-source/legacy_source"
FROZEN_SOURCE_SHA256 = "cea57ee6fa467ec068d7e71668b4452a667514f62ed066a2e17585008ff635a6"
FROZEN_CELLS_SHA256 = "5f98411eb66b09466aa8a1e8fd30caa6056860f8b7b9d7db0e4a3f185eb4d591"
LABELS_RELATIVE = "data/demo-dataset/roi-a/masked/labels.zarr"
CHANNELS_RELATIVE = {
    "pi": "data/demo-dataset/fused/pi.zarr",
    "edu": "data/demo-dataset/fused/edu.zarr",
    "brdu": "data/demo-dataset/fused/brdu.zarr",
}
CHANNEL_VALUE_OFFSET = {"pi": 1000, "edu": 2000, "brdu": 3000}
OFFSET_ZYX = (0, 2, 3)
CELL_COLUMNS = [
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
]
CELL_DTYPES = [
    pl.String,
    pl.String,
    pl.String,
    pl.UInt32,
    pl.UInt64,
    pl.Float64,
    pl.Float64,
    pl.Float64,
    pl.UInt32,
    pl.UInt64,
]
CELL_ROWS = [
    {
        "cell": "roi-a|7",
        "dataset": "demo-dataset",
        "roi": "roi-a",
        "label": 7,
        "area": 36,
        "centroid_z": 2.0,
        "centroid_y": 2.5,
        "centroid_x": 3.0,
        "plane_z": 2,
        "plane_area": 12,
    },
    {
        "cell": "roi-a|9",
        "dataset": "demo-dataset",
        "roi": "roi-a",
        "label": 9,
        "area": 8,
        "centroid_z": 0.5,
        "centroid_y": 4.5,
        "centroid_x": 0.5,
        "plane_z": 0,
        "plane_area": 4,
    },
]
ARRAY_DIGESTS = {
    "labels": {
        "content_sha256": "f68741ac91012c0cee8c82e504ec227b937f9506bf068caf7dfa7a3996b21785",
        "metadata_sha256": "09c1dd24f98fc9df83497dbd0c2e6068ca6a9efbc7ce9a5815fb397652ac6022",
    },
    "brdu": {
        "content_sha256": "0b9e5544c18ab0848cd6fcc2c3e1222ba49fe976688a9b4d472d09afb34dca9c",
        "metadata_sha256": "cd17168fe7abfc29286dfffb29186597624ea1e56dad8909381a73c8b421ce06",
    },
    "edu": {
        "content_sha256": "4b8747434bf00fcade2dddc81fa4f684bf55ca163d4258079d62170875725a42",
        "metadata_sha256": "cd17168fe7abfc29286dfffb29186597624ea1e56dad8909381a73c8b421ce06",
    },
    "pi": {
        "content_sha256": "4a217ca74df828ec21c4a6742f1374eb710c5f05daee25c62b1bc567346d8e75",
        "metadata_sha256": "cd17168fe7abfc29286dfffb29186597624ea1e56dad8909381a73c8b421ce06",
    },
}
OME = {
    "axes": [
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ],
    "coordinate_transformations": [{"scale": [1.0, 0.216, 0.216], "type": "scale"}],
}


# Invocation helpers: the only places that name the marker-specific prepare and manifest APIs.


def run_prepare(
    *,
    labels: Path,
    channels: Mapping[str, Path],
    offset_zyx: tuple[int, int, int],
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> list[str]:
    """Run ``dino-zarr prepare`` once and return its stdout lines."""
    argv = ["dino-zarr prepare", "--labels", str(labels)]
    for key in ("pi", "edu", "brdu"):
        argv += [f"--{key}", str(channels[key])]
    argv += ["--offset-zyx", *(str(value) for value in offset_zyx)]
    argv += ["--out-dir", str(out_dir), "--workers", "1"]
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", argv)
    zarr_prepare.main()
    return capsys.readouterr().out.splitlines()


def build_manifest(
    *,
    labels: Path,
    channels: Mapping[str, Path],
    offset_zyx: tuple[int, int, int],
) -> dict[str, Any]:
    return build_source_manifest(labels_path=labels, channel_paths=channels, offset_zyx=offset_zyx)


# Test-side reference implementations and fixture relocation.


def reference_digest(manifest: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "source_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True) + "\n").encode("utf-8")


def frozen_json(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE / "prepared" / name).read_text())


def relocate(tmp_path: Path) -> tuple[Path, str]:
    """Copy the frozen fixture and move its recorded roots; return (root, relocated digest)."""
    root = tmp_path.resolve() / "legacy_source"
    shutil.copytree(FIXTURE, root, ignore=shutil.ignore_patterns("README.md"))
    source_text = (root / "prepared" / "source.json").read_text().replace(FROZEN_PREFIX, str(root))
    digest = reference_digest(json.loads(source_text))
    for name in ("source.json", "complete.json"):
        path = root / "prepared" / name
        text = path.read_text().replace(FROZEN_PREFIX, str(root))
        path.write_text(text.replace(FROZEN_SOURCE_SHA256, digest))
    return root, digest


def fixture_channels(root: Path) -> dict[str, Path]:
    return {key: root / relative for key, relative in CHANNELS_RELATIVE.items()}


def expected_labels() -> np.ndarray:
    labels = np.zeros((4, 6, 6), dtype=np.uint32)
    labels[1:4, 1:5, 2:5] = 7
    labels[0:2, 4:6, 0:2] = 9
    return labels


def expected_channel(key: str) -> np.ndarray:
    return np.arange(400, dtype=np.uint16).reshape(4, 10, 10) + np.uint16(CHANNEL_VALUE_OFFSET[key])


def tree_bytes(directory: Path) -> dict[Path, bytes]:
    return {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


# Frozen artifact identity.


def test_frozen_manifest_digest_matches_stored_and_reference() -> None:
    manifest = frozen_json("source.json")

    assert manifest["source_sha256"] == FROZEN_SOURCE_SHA256
    assert manifest_digest(manifest) == FROZEN_SOURCE_SHA256
    assert reference_digest(manifest) == FROZEN_SOURCE_SHA256
    assert manifest_digest({**manifest, "source_sha256": "ignored"}) == FROZEN_SOURCE_SHA256
    assert manifest_digest(dict(reversed(list(manifest.items())))) == FROZEN_SOURCE_SHA256
    assert manifest_digest({**manifest, "offset_zyx": [0, 2, 4]}) != FROZEN_SOURCE_SHA256


def test_frozen_manifest_structure() -> None:
    path = FIXTURE / "prepared" / "source.json"
    manifest = json.loads(path.read_text())

    assert path.read_bytes() == json_bytes(manifest)
    assert list(manifest) == [
        "channels",
        "dataset",
        "labels",
        "offset_zyx",
        "roi",
        "schema",
        "source_sha256",
    ]
    assert manifest["schema"] == "dino-zarr-source-v1"
    assert manifest["dataset"] == "demo-dataset"
    assert manifest["roi"] == "roi-a"
    assert manifest["offset_zyx"] == [0, 2, 3]
    assert manifest["labels"] == {
        "array": "",
        "chunks": [2, 3, 3],
        "dtype": "uint32",
        "root": f"{FROZEN_PREFIX}/{LABELS_RELATIVE}",
        "shape": [4, 6, 6],
        **ARRAY_DIGESTS["labels"],
    }
    assert list(manifest["channels"]) == ["brdu", "edu", "pi"]
    for key, spec in manifest["channels"].items():
        assert spec == {
            "array": "0",
            "chunks": [2, 5, 5],
            "dtype": "uint16",
            "ome": OME,
            "root": f"{FROZEN_PREFIX}/{CHANNELS_RELATIVE[key]}",
            "shape": [4, 10, 10],
            **ARRAY_DIGESTS[key],
        }


def test_frozen_complete_record_and_cell_table() -> None:
    path = FIXTURE / "prepared" / "complete.json"
    complete = json.loads(path.read_text())
    cells_path = FIXTURE / "prepared" / "cells.parquet"

    assert path.read_bytes() == json_bytes(complete)
    assert complete == {
        "artifact": "dino-zarr-source-complete-v1",
        "cells": 2,
        "cells_sha256": FROZEN_CELLS_SHA256,
        "source_sha256": FROZEN_SOURCE_SHA256,
    }
    assert list(complete) == ["artifact", "cells", "cells_sha256", "source_sha256"]
    assert file_sha256(cells_path) == FROZEN_CELLS_SHA256
    cells = pl.read_parquet(cells_path)
    assert cells.columns == CELL_COLUMNS
    assert cells.dtypes == CELL_DTYPES
    assert cells.to_dicts() == CELL_ROWS


# Manifest construction.


def test_build_manifest_reproduces_frozen_bytes_and_digest(tmp_path) -> None:
    root, digest = relocate(tmp_path)
    channels = fixture_channels(root)
    shuffled = {key: channels[key] for key in ("brdu", "pi", "edu")}

    manifest = build_manifest(labels=root / LABELS_RELATIVE, channels=shuffled, offset_zyx=OFFSET_ZYX)

    assert manifest == json.loads((root / "prepared" / "source.json").read_text())
    assert manifest["source_sha256"] == digest == manifest_digest(manifest) == reference_digest(manifest)
    written = tmp_path / "written" / "source.json"
    write_json_atomic(manifest, written, overwrite=False)
    assert written.read_bytes() == (root / "prepared" / "source.json").read_bytes()
    at_frozen_prefix = json.loads(json.dumps(manifest).replace(str(root), FROZEN_PREFIX))
    assert manifest_digest(at_frozen_prefix) == FROZEN_SOURCE_SHA256


def test_build_manifest_keeps_extra_channels_and_requires_brdu(tmp_path) -> None:
    root, _ = relocate(tmp_path)
    channels = fixture_channels(root)
    extra = channels["pi"]
    mixed = {"zeta": extra, "brdu": channels["brdu"], "c561": extra, "pi": channels["pi"]}
    mixed |= {"dapi": extra, "edu": channels["edu"]}

    manifest = build_manifest(labels=root / LABELS_RELATIVE, channels=mixed, offset_zyx=OFFSET_ZYX)

    assert manifest["channels"]["dapi"] == manifest["channels"]["pi"]
    written = tmp_path / "written" / "source.json"
    write_json_atomic(manifest, written, overwrite=False)
    assert list(json.loads(written.read_text())["channels"]) == ["brdu", "c561", "dapi", "edu", "pi", "zeta"]
    assert manifest["source_sha256"] == reference_digest(manifest)
    assert manifest["source_sha256"] != reference_digest(json.loads((root / "prepared" / "source.json").read_text()))
    with pytest.raises(ValueError, match="brdu"):
        build_manifest(
            labels=root / LABELS_RELATIVE,
            channels={key: channels[key] for key in ("pi", "edu")},
            offset_zyx=OFFSET_ZYX,
        )


# Readers on the frozen legacy source.


def test_load_source_accepts_frozen_legacy_source(tmp_path) -> None:
    root, digest = relocate(tmp_path)

    source = load_source(root / "prepared")

    assert source.manifest == json.loads((root / "prepared" / "source.json").read_text())
    assert source.manifest["source_sha256"] == digest
    assert source.offset_zyx == OFFSET_ZYX
    assert list(source.channels) == ["brdu", "edu", "pi"]
    np.testing.assert_array_equal(source.labels[:], expected_labels())
    assert source.labels.dtype == np.uint32
    for key, array in source.channels.items():
        assert array.dtype == np.uint16
        np.testing.assert_array_equal(array[:], expected_channel(key))


def test_open_review_source_accepts_frozen_legacy_source(tmp_path) -> None:
    root, digest = relocate(tmp_path)

    source = open_review_source(root / "prepared")

    assert source.manifest["source_sha256"] == digest
    assert source.offset_zyx == OFFSET_ZYX
    assert list(source.channels) == ["brdu", "edu", "pi"]
    np.testing.assert_array_equal(source.labels[:], expected_labels())
    for key, array in source.channels.items():
        np.testing.assert_array_equal(array[:], expected_channel(key))


# Prepare: resume of the frozen output and a fresh run on the frozen inputs.


@pytest.mark.integration
def test_prepare_resumes_frozen_legacy_output_without_rewriting(tmp_path, monkeypatch, capsys) -> None:
    root, _ = relocate(tmp_path)
    out_dir = root / "prepared"
    before = tree_bytes(out_dir)

    lines = run_prepare(
        labels=root / LABELS_RELATIVE,
        channels=fixture_channels(root),
        offset_zyx=OFFSET_ZYX,
        out_dir=out_dir,
        monkeypatch=monkeypatch,
        capsys=capsys,
    )

    assert lines == [f"source_dir={out_dir}", "status=skipped_complete"]
    after = tree_bytes(out_dir)
    assert {path: after[path] for path in before} == before


@pytest.mark.integration
def test_prepare_reproduces_frozen_output_from_frozen_inputs(tmp_path, monkeypatch, capsys) -> None:
    root, digest = relocate(tmp_path)
    out_dir = tmp_path / "fresh"

    lines = run_prepare(
        labels=root / LABELS_RELATIVE,
        channels=fixture_channels(root),
        offset_zyx=OFFSET_ZYX,
        out_dir=out_dir,
        monkeypatch=monkeypatch,
        capsys=capsys,
    )

    assert lines == [f"source_dir={out_dir}", "cells=2", "status=complete"]
    assert (out_dir / "source.json").read_bytes() == (root / "prepared" / "source.json").read_bytes()
    complete = json.loads((out_dir / "complete.json").read_text())
    assert complete["source_sha256"] == digest
    assert complete["cells"] == 2
    assert complete["cells_sha256"] == file_sha256(out_dir / "cells.parquet")
    assert "measured_props_sha256" not in complete
    cells = pl.read_parquet(out_dir / "cells.parquet")
    assert cells.columns == CELL_COLUMNS
    assert cells.dtypes == CELL_DTYPES
    assert cells.equals(pl.read_parquet(root / "prepared" / "cells.parquet"))
    starts = [f"{z}-{y}-{x}" for z in (0, 2) for y in (0, 3) for x in (0, 3)]
    partials = sorted(path.name for path in (out_dir / "partials").iterdir())
    assert partials == sorted([f"{kind}-{start}.parquet" for kind in ("m", "p") for start in starts])
    for name in partials:
        metadata, _ = read_parquet_artifact(out_dir / "partials" / name)
        assert metadata == {"artifact": "dino-zarr-chunk-v1", "source_sha256": digest}
