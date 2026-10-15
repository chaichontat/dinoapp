from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from dino.artifacts import (
    file_sha256,
    ordered_cell_sha256,
    read_embedding_artifact,
    read_parquet_artifact,
    write_embedding_atomic,
)
from dino.cortex_score import score_cortex

ENCODER = {
    "feature_extractor": "cell-dino",
    "weights_sha256": "w",
    "model_source_sha256": "s",
    "crop_size": 64,
    "resize_size": 224,
    "crop_eval_size": 224,
    "highpass_sigma_px": None,
    "mask_erosion_px": 0,
    "input_channels": [0, 1],
    "input_channel_names_by_target": {"brdu": ["dapi", "brdu"], "edu": ["dapi", "edu"]},
    "feature_precision": "bfloat16",
    "feature_execution": "native_sdpa",
    "pixel_correction": None,
}
CELLS = 40


@pytest.fixture
def scoring_inputs(tmp_path):
    rng = np.random.default_rng(0)
    labels = np.arange(1, CELLS + 1)
    props = pd.DataFrame({
        "label": labels.astype(np.uint32),
        "area": rng.integers(50, 100, CELLS),
        "centroid_z": rng.uniform(0, 10, CELLS),
        "centroid_y": rng.uniform(0, 50, CELLS),
        "centroid_x": rng.uniform(0, 50, CELLS),
        "intensity_brdu_mean": rng.uniform(0, 500, CELLS),
        "intensity_edu_mean": rng.uniform(0, 500, CELLS),
    })
    props.to_parquet(tmp_path / "props.parquet", index=False)
    props[["label", "area"]].assign(intensity_dapi_mean=rng.uniform(0, 300, CELLS)).to_parquet(
        tmp_path / "dapi.parquet", index=False)
    inside = np.ones(CELLS, dtype=bool)
    inside[[2, 7]] = False
    pd.DataFrame({"label": labels, "inside_cortex": inside}).to_parquet(tmp_path / "cortex.parquet", index=False)
    pd.DataFrame({"cell": ["roi|5"]}).to_csv(tmp_path / "labeled.csv", index=False)

    embedding_bundle = {**ENCODER, "schema": "dino-knn-intensity-v1"}
    joblib.dump(embedding_bundle, tmp_path / "embedding-model.joblib")
    bundle = {
        **embedding_bundle,
        "source_sha256": "source",
        "classifier_feature_mode": "marker",
        "context_features": ["log1p_cell_mean", "log_cell_over_local_mean", "log1p_dapi_mean"],
        "intensity_neighbors": 3,
        "intensity_props": str(tmp_path / "props.parquet"),
        "intensity_props_sha256": file_sha256(tmp_path / "props.parquet"),
        "intensity_scale_zyx_um": [0.6, 0.3, 0.3],
        "dapi_props": str(tmp_path / "dapi.parquet"),
        "dapi_props_sha256": file_sha256(tmp_path / "dapi.parquet"),
    }
    for target in ("brdu", "edu"):
        x = rng.normal(size=(60, 1027))
        y = (x[:, 0] + rng.normal(size=60) > 0).astype(int)
        scaler = StandardScaler().fit(x)
        bundle.update({f"{target}_feature_scaler": scaler,
                       f"{target}_model": LogisticRegression().fit(scaler.transform(x), y),
                       f"{target}_cutoff": 0.5})
    joblib.dump(bundle, tmp_path / "model.joblib")

    run = tmp_path / "infer"
    manifest = {"stores_embeddings": True, "source_sha256": "source",
                "model_sha256": file_sha256(tmp_path / "embedding-model.joblib")}
    for shard, part_labels in enumerate(np.array_split(labels, 3)):
        shard_dir = run / f"shard-{shard:03d}"
        (shard_dir / "predictions").mkdir(parents=True)
        (shard_dir / "complete.json").write_text(json.dumps({
            **manifest, "shard_count": 3, "shard_index": shard, "cells": len(part_labels), "partition_cells": 7,
        }))
        for part, start in enumerate(range(0, len(part_labels), 7)):
            chunk = part_labels[start:start + 7]
            cells = [f"roi|{label}" for label in chunk]
            pd.DataFrame({"cell": cells, "dataset": "d", "roi": "roi", "label": chunk}).to_parquet(
                shard_dir / "predictions" / f"part-{part:06d}.parquet")
            write_embedding_atomic(
                {target: torch.from_numpy(rng.normal(size=(len(chunk), 1024))).to(torch.bfloat16)
                 for target in ("brdu", "edu")} | {"cell_sha256": ordered_cell_sha256(cells)},
                shard_dir / "embeddings" / f"part-{part:06d}.pt")
    return tmp_path, bundle


def _score(tmp_path, output, workers):
    return score_cortex(model=tmp_path / "model.joblib", embedding_model=tmp_path / "embedding-model.joblib",
                        embedding_run=tmp_path / "infer", cortex_props=tmp_path / "cortex.parquet",
                        labeled_cells=tmp_path / "labeled.csv", output_dir=output, workers=workers)


def test_cortex_score_publishes_selected_cells_with_saved_classifier(scoring_inputs):
    tmp_path, bundle = scoring_inputs
    complete = _score(tmp_path, tmp_path / "out", workers=2)

    predictions = pd.concat(
        [pd.read_parquet(path) for path in sorted((tmp_path / "out" / "predictions").glob("*.parquet"))],
        ignore_index=True)
    assert list(predictions.columns) == ["cell", "dataset", "roi", "label", "p_brdu", "brdu", "p_edu", "edu"]
    assert predictions["label"].tolist() == [label for label in range(1, CELLS + 1) if label not in (3, 5, 8)]
    assert complete["cells"] == CELLS - 3 and complete["source_partitions"] == 6
    assert complete["prediction_partitions"] == 6 and complete["stores_embeddings"] is False
    assert not (tmp_path / "out" / "embeddings").exists()
    assert complete["brdu_positive"] == int(predictions["brdu"].sum())
    assert json.loads((tmp_path / "out" / "complete.json").read_text()) == complete

    path = tmp_path / "out" / "predictions" / "shard-000-part-000000.parquet"
    provenance, rows = read_parquet_artifact(path)
    assert provenance["source_rows"] == 7 and provenance["partition_rows"] == rows == 5
    assert provenance["model_sha256"] == file_sha256(tmp_path / "model.joblib")
    source = read_embedding_artifact(tmp_path / "infer" / "shard-000" / "embeddings" / "part-000000.pt")
    keep = np.array([True, True, False, True, False, True, True])

    scored = pd.read_parquet(path)
    props = pd.read_parquet(tmp_path / "props.parquet")
    dapi = pd.read_parquet(tmp_path / "dapi.parquet")
    points = props[["centroid_z", "centroid_y", "centroid_x"]].to_numpy() * [0.6, 0.3, 0.3]
    for row, label in zip(np.flatnonzero(keep), scored["label"]):
        distance = np.linalg.norm(points - points[label - 1], axis=1)
        distance[label - 1] = np.inf
        neighbors = np.argsort(distance)[:3]
        cell = props["intensity_brdu_mean"].to_numpy()
        features = np.r_[source["brdu"][row].float().numpy(), np.log1p(cell[label - 1]),
                         np.log((cell[label - 1] + 1) / (cell[neighbors].mean() + 1)),
                         np.log1p(dapi["intensity_dapi_mean"].iloc[label - 1])]
        expected = bundle["brdu_model"].predict_proba(bundle["brdu_feature_scaler"].transform(features[None]))[0, 1]
        observed = scored.loc[scored["label"] == label, "p_brdu"].item()
        assert observed == pytest.approx(expected, abs=1e-6)


def test_cortex_score_matches_serial_run_and_resumes(scoring_inputs):
    tmp_path, _ = scoring_inputs
    serial = _score(tmp_path, tmp_path / "serial", workers=1)
    parallel = _score(tmp_path, tmp_path / "parallel", workers=3)
    assert serial == parallel
    for path in (tmp_path / "serial" / "predictions").glob("*.parquet"):
        pd.testing.assert_frame_equal(pd.read_parquet(path), pd.read_parquet(tmp_path / "parallel" / "predictions" / path.name))
    assert _score(tmp_path, tmp_path / "parallel", workers=2) == parallel


def test_cortex_score_rejects_changed_embedding_identity(scoring_inputs):
    tmp_path, _ = scoring_inputs
    (tmp_path / "infer" / "shard-001" / "complete.json").write_text(
        json.dumps({"stores_embeddings": True, "source_sha256": "other", "model_sha256": "x"}))
    with pytest.raises(ValueError, match="source differs"):
        _score(tmp_path, tmp_path / "out", workers=2)


@pytest.mark.parametrize('damage', ['shard', 'partition', 'short', 'embedding'])
def test_cortex_score_rejects_incomplete_source_before_scoring(scoring_inputs, damage):
    tmp_path, _ = scoring_inputs
    shard = tmp_path / 'infer/shard-001'
    prediction = shard / 'predictions/part-000000.parquet'
    if damage == 'shard':
        shard.rename(tmp_path / 'held-shard')
    elif damage == 'partition':
        prediction.rename(tmp_path / 'held.parquet')
    elif damage == 'short':
        pd.read_parquet(prediction).iloc[:-1].to_parquet(prediction, index=False)
    else:
        (shard / 'embeddings/part-000000.pt').rename(tmp_path / 'held.pt')
    with pytest.raises(ValueError):
        _score(tmp_path, tmp_path / 'out', workers=1)
    assert not (tmp_path / 'out/predictions').exists()


def test_cortex_score_rejects_changed_exclusions_even_when_no_cells_selected(scoring_inputs):
    tmp_path, _ = scoring_inputs
    _score(tmp_path, tmp_path / 'out', workers=1)
    paths = [tmp_path / 'out/complete.json', *(tmp_path / 'out/predictions').glob('*.parquet')]
    before = {path: path.read_bytes() for path in paths}
    pd.DataFrame({'cell': [f'roi|{label}' for label in range(1, CELLS + 1)]}).to_csv(
        tmp_path / 'labeled.csv', index=False)
    with pytest.raises(ValueError, match='contract'):
        _score(tmp_path, tmp_path / 'out', workers=1)
    assert {path: path.read_bytes() for path in paths} == before


def test_cortex_score_checks_partial_output_contract_before_empty_selection(scoring_inputs):
    tmp_path, _ = scoring_inputs
    _score(tmp_path, tmp_path / 'out', workers=1)
    (tmp_path / 'out/complete.json').rename(tmp_path / 'held-complete.json')
    pd.DataFrame({'cell': [f'roi|{label}' for label in range(1, CELLS + 1)]}).to_csv(
        tmp_path / 'labeled.csv', index=False)
    with pytest.raises(ValueError, match='contract'):
        _score(tmp_path, tmp_path / 'out', workers=1)
    assert not (tmp_path / 'out/complete.json').exists()


def test_cortex_score_allows_fresh_empty_selection(scoring_inputs):
    tmp_path, _ = scoring_inputs
    pd.DataFrame({'cell': [f'roi|{label}' for label in range(1, CELLS + 1)]}).to_csv(
        tmp_path / 'labeled.csv', index=False)
    complete = _score(tmp_path, tmp_path / 'out', workers=1)
    assert complete['cells'] == complete['prediction_partitions'] == 0
    assert list((tmp_path / 'out/predictions').iterdir()) == []


def test_cortex_score_resumes_partial_output_with_same_contract(scoring_inputs):
    tmp_path, _ = scoring_inputs
    complete = _score(tmp_path, tmp_path / 'out', workers=1)
    (tmp_path / 'out/complete.json').rename(tmp_path / 'held-complete.json')
    path = tmp_path / 'out/predictions/shard-000-part-000000.parquet'
    path.rename(tmp_path / 'held.parquet')
    reused = tmp_path / 'out/predictions/shard-001-part-000000.parquet'
    before = reused.read_bytes()
    assert _score(tmp_path, tmp_path / 'out', workers=1) == complete
    assert reused.read_bytes() == before


def test_cortex_score_rejects_unexpected_output_partition(scoring_inputs):
    tmp_path, _ = scoring_inputs
    _score(tmp_path, tmp_path / 'out', workers=1)
    path = tmp_path / 'out/predictions/shard-000-part-000000.parquet'
    extra = path.with_name('stale.parquet')
    extra.write_bytes(path.read_bytes())
    with pytest.raises(ValueError):
        _score(tmp_path, tmp_path / 'out', workers=1)
    assert extra.exists()
