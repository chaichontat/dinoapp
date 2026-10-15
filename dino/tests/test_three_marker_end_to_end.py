"""A three-marker run from prepare through review, chaining each stage's real artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import pytest
import torch
import zarr
from click.testing import CliRunner
from dino_markers import INFERENCE, MarkerContract, config_to_json, contract_of, load_config
from test_zarr_prepare import write_markers
from test_zarr_train import _patch_encoder, _SumEncoder, _train_three_markers

from dino import zarr_cli, zarr_infer
from dinoapp.label_store import load_labels
from dinoapp.sources import AnnotationFiles, BarrageSource

THREE = MarkerContract("dapi", ("edu", "brdu", "pcna"))
CELLS = 32
GRID_COLUMNS = 8


def _write_masked_labels_and_channels(root: Path) -> tuple[Path, dict[str, Path], dict[str, np.ndarray]]:
    """Thirty-two 2x2 cells, four of each assignment of the three markers, as masked labels and channel Zarrs."""
    masked = root / "dataset" / "roi" / "masked"
    masked.mkdir(parents=True)
    cells = np.arange(CELLS)
    calls = {marker: (cells // 2**bit) % 2 for bit, marker in enumerate(THREE.markers)}
    pixels = np.zeros((1, 16, 32), dtype=np.uint32)
    for cell in cells:
        row, column = divmod(int(cell), GRID_COLUMNS)
        pixels[0, 4 * row + 1 : 4 * row + 3, 4 * column + 1 : 4 * column + 3] = cell + 1
    labels_path = masked / "labels.zarr"
    zarr.open_array(labels_path, mode="w", shape=pixels.shape, chunks=(1, 16, 32), dtype="u4")[:] = pixels

    volumes = np.zeros(CELLS + 1, dtype=np.uint64)
    volumes[1:] = 4
    np.save(masked / "volumes.npy", volumes)
    pl.DataFrame({"label": cells + 1, "area": [4] * CELLS}).write_parquet(masked / "props.parquet")

    rng = np.random.default_rng(11)
    channel_paths = {}
    for key in THREE.channel_keys:
        base = np.full((1, 18, 34), 500.0)
        if key in calls:
            for cell in cells:
                row, column = divmod(int(cell), GRID_COLUMNS)
                base[0, 4 * row + 2 : 4 * row + 4, 4 * column + 2 : 4 * column + 4] += 2000.0 * calls[key][cell]
        path = root / f"{key}.zarr"
        array = zarr.open_array(path, mode="w", shape=base.shape, chunks=base.shape, dtype="u2")
        array[:] = (base + rng.uniform(0, 100, base.shape)).astype(np.uint16)
        channel_paths[key] = path
    return labels_path, channel_paths, calls


def _invoke(*args: str) -> None:
    result = CliRunner().invoke(zarr_cli.main, list(args))
    assert result.exit_code == 0, (result.output, result.exception)


@pytest.mark.integration
def test_three_marker_run_completes_prepare_barrage_label_train_infer_review(monkeypatch, tmp_path) -> None:
    contract_json = {"version": 1, "reference": "dapi", "markers": ["edu", "brdu", "pcna"]}
    assert THREE.to_json() == contract_json
    labels_path, channel_paths, calls = _write_masked_labels_and_channels(tmp_path)
    markers_file = write_markers(tmp_path / "markers_in.json", "dapi", ["edu", "brdu", "pcna"])
    expected_config = config_to_json(load_config(markers_file))

    # prepare: writes the source that every later stage reads.
    source_dir = tmp_path / "source"
    _invoke(
        "prepare", "--labels", str(labels_path),
        *(arg for key in THREE.channel_keys for arg in ("--channel", f"{key}={channel_paths[key]}")),
        "--offset-zyx", "0", "1", "1", "--out-dir", str(source_dir), "--workers", "1",
        "--markers", str(markers_file),
    )
    assert json.loads((source_dir / "complete.json").read_text())["marker_contract"] == contract_json
    assert json.loads((source_dir / "markers.json").read_text()) == expected_config
    assert contract_of(source_dir) == THREE
    prepared_cells = pl.read_parquet(source_dir / "cells.parquet")["cell"].to_list()
    assert sorted(prepared_cells) == sorted(f"roi|{n}" for n in range(1, CELLS + 1))

    # barrage: reads the prepared source and inherits its contract and config.
    barrage_dir = tmp_path / "barrage"
    _invoke(
        "barrage", "--source-dir", str(source_dir), "--out-dir", str(barrage_dir),
        "--candidate-cells", str(CELLS), "--n-cells", str(CELLS), "--size", "8", "--cache-gb", "0.001",
    )
    barrage_run = json.loads((barrage_dir / "run.json").read_text())
    assert barrage_run["marker_contract"] == contract_json
    assert json.loads((barrage_dir / "complete.json").read_text()) == barrage_run
    assert json.loads((barrage_dir / "markers.json").read_text()) == expected_config
    index = pd.read_csv(barrage_dir / "index.csv")
    assert sorted(index["cell"]) == sorted(prepared_cells)
    assert {f"{m}_mean" for m in THREE.markers} <= set(index.columns)

    # label: write labels.csv through dinoapp's label store, one triple per cell.
    source = BarrageSource(barrage_dir)
    annotations = AnnotationFiles(
        source.annotation_paths, source.default_annotation_path, contract=source.config.contract
    )
    expected_labels = {
        f"roi|{n + 1}": tuple(int(calls[marker][n]) for marker in THREE.markers) for n in range(CELLS)
    }
    for cell, value in expected_labels.items():
        annotations.assign(cell, value, expected=None)
    assert list(pd.read_csv(barrage_dir / "labels.csv").columns) == ["cell", "edu", "brdu", "pcna"]
    assert AnnotationFiles(
        source.annotation_paths, source.default_annotation_path, contract=THREE
    ).values == expected_labels

    # train: fits on the barrage with the stub encoder.
    (tmp_path / "encoder").mkdir()
    (tmp_path / "encoder" / "config.json").write_text("{}")
    (tmp_path / "weights.pth").write_text("weights")
    _patch_encoder(monkeypatch, lambda: _SumEncoder().eval())
    _train_three_markers(monkeypatch, tmp_path, out="model")
    bundle = joblib.load(tmp_path / "model" / "model.joblib")
    cache = joblib.load(tmp_path / "model" / "features.joblib")
    assert bundle["marker_contract"] == contract_json
    assert cache["metadata"]["marker_contract"] == contract_json
    assert len(bundle["train_cells"]) + len(bundle["validation_cells"]) == CELLS
    assert all(f"{marker}_model" in bundle for marker in THREE.markers)

    # infer: classifies the prepared source with the trained model and the same stub.
    monkeypatch.setattr(zarr_infer, "load_cell_dino_model", lambda **kwargs: _SumEncoder().eval())
    monkeypatch.setattr(zarr_infer, "build_cell_transform", lambda **kwargs: torch.nn.Identity())
    inference_dir = tmp_path / "inference"
    _invoke(
        "infer", "--source-dir", str(source_dir), "--model-dir", str(tmp_path / "model"),
        "--out-dir", str(inference_dir), "--batch-size", "4", "--partition-cells", "16",
        "--cache-gb", "0.001", "--prefetch-workers", "1", "--crop-workers", "1",
    )
    inference = json.loads((inference_dir / "contract.json").read_text())
    assert inference["marker_contract"] == contract_json
    assert json.loads((inference_dir / "complete.json").read_text()) == inference
    predictions = pd.concat(
        [pd.read_parquet(path) for path in sorted((inference_dir / "predictions").glob("*.parquet"))],
        ignore_index=True,
    )
    assert predictions["cell"].tolist() == prepared_cells
    assert [c for c in predictions.columns if c.startswith("p_")] == ["p_edu", "p_brdu", "p_pcna"]
    assert list(predictions.columns[-3:]) == ["edu", "brdu", "pcna"]
    for marker in THREE.markers:
        assert predictions[f"p_{marker}"].between(0, 1).all()
        assert predictions[marker].isin([0, 1]).all()
        assert predictions[marker].tolist() == [expected_labels[cell][THREE.markers.index(marker)]
                                                for cell in predictions["cell"]]

    # review: reopens the barrage and reads its config, labels, crops, and the inference calls.
    review = BarrageSource(barrage_dir)
    assert review.config == load_config(markers_file)
    assert review.config.contract == THREE
    assert INFERENCE.contract(inference, where=str(inference_dir)) == THREE
    assert load_labels(barrage_dir / "labels.csv", contract=THREE).columns.tolist() == [
        "cell", "edu", "brdu", "pcna"
    ]
    reviewed = AnnotationFiles(review.annotation_paths, review.default_annotation_path, contract=THREE)
    assert reviewed.values == expected_labels
    called = {
        cell: tuple(int(row[marker]) for marker in THREE.markers)
        for cell, row in predictions.set_index("cell").iterrows()
    }
    assert called == reviewed.values
    assert set(review.crop_markers) == {"combined", *THREE.markers}
    for marker in review.crop_markers:
        assert review.crop(prepared_cells[0], marker).image.startswith(b"\x89PNG")
