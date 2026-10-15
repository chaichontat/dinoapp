import json
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import zarr
from click.testing import CliRunner
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from dino import zarr_cli, zarr_infer, zarr_model
from dino.artifacts import (
    file_sha256,
    ordered_cell_sha256,
    path_sha256,
    read_embedding_artifact,
    read_parquet_artifact,
)
from dino.intensity_context import IntensityContext
from dino.zarr_source import build_source_manifest, write_json_atomic


@pytest.fixture
def inference_run(tmp_path, monkeypatch):
    labels_path = tmp_path / "ds" / "roi" / "masked" / "labels.zarr"
    labels = zarr.open_array(labels_path, mode="w", shape=(1, 8, 24), chunks=(1, 4, 6), dtype="u4")
    centers_x = [3, 9, 15, 21]
    channel_values = {"brdu": [1, 2, 4, 8], "edu": [8, 4, 2, 1], "pi": [2] * 4}
    channel_paths = {}
    for name, values in channel_values.items():
        path = tmp_path / f"{name}.zarr"
        array = zarr.open_array(path, mode="w", shape=(1, 10, 28), chunks=(1, 5, 7), dtype="u2")
        array[:] = 65_000
        for label, (center_x, value) in enumerate(zip(centers_x, values, strict=True), start=1):
            labels[0, 3:5, center_x - 1 : center_x + 1] = label
            array[0, 4:6, center_x + 1 : center_x + 3] = value * 256
        channel_paths[name] = path

    source_dir = tmp_path / "source"
    manifest = build_source_manifest(
        labels_path=labels_path, channel_paths=channel_paths, offset_zyx=(0, 1, 2)
    )
    write_json_atomic(manifest, source_dir / "source.json")
    cells = pd.DataFrame(
        {
            "cell": [f"roi|{label}" for label in range(1, 5)],
            "dataset": ["ds"] * 4,
            "roi": ["roi"] * 4,
            "label": [1, 2, 3, 4],
            "plane_z": [0] * 4,
            "centroid_y": [4.0] * 4,
            "centroid_x": centers_x,
        }
    ).iloc[[2, 0, 3, 1]].reset_index(drop=True)
    cells.to_parquet(source_dir / "cells.parquet", index=False)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    encoder_dir = tmp_path / "encoder"
    encoder_dir.mkdir()
    (encoder_dir / "config.json").write_text('{"fixture": "channel-sums"}')
    weights = encoder_dir / "weights.identity"
    weights.write_bytes(b"synthetic external encoder identity")
    bundle = {
        "schema": "dino-zarr-model-v6",
        "source_sha256": manifest["source_sha256"],
        "feature_extractor": "cell_dino_hpa_vitl16",
        "input_channels": 4,
        "weights": str(weights),
        "weights_sha256": file_sha256(weights),
        "dinov2_repo": str(encoder_dir),
        "model_source_sha256": path_sha256(encoder_dir),
        "crop_size": 4,
        "resize_size": 4,
        "crop_eval_size": 4,
        "highpass_sigma_px": 0.0,
        "mask_erosion_px": 0,
        "input_channel_names_by_target": {"brdu": ["brdu", "pi"], "edu": ["edu", "pi"]},
        "feature_precision": "bfloat16",
        "feature_execution": "native_sdpa",
        "brdu_cutoff": 0.5,
        "edu_cutoff": 0.5,
    }
    expected_embeddings = {}
    expected_probabilities = {}
    for target in ("brdu", "edu"):
        vectors = torch.tensor(list(zip(channel_values[target], [2] * 4)), dtype=torch.bfloat16)
        normalized = vectors / torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
        features = normalized.float().numpy()
        scaler = StandardScaler().fit(features)
        classifier = LogisticRegression(random_state=0).fit(
            scaler.transform(features), np.asarray(channel_values[target]) >= 4
        )
        bundle[f"{target}_feature_scaler"] = scaler
        bundle[f"{target}_model"] = classifier
        expected_embeddings[target] = normalized[cells["label"].to_numpy() - 1]
        expected_probabilities[target] = classifier.predict_proba(
            scaler.transform(expected_embeddings[target].float().numpy())
        )[:, 1]
    joblib.dump(bundle, model_dir / "model.joblib")

    class ChannelSumEncoder(torch.nn.Module):
        def forward(self, batch):
            return batch[:, :2].sum(dim=(2, 3))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(zarr_infer, "load_cell_dino_model", lambda **kwargs: ChannelSumEncoder().eval())
    monkeypatch.setattr(zarr_infer, "build_cell_transform", lambda **kwargs: torch.nn.Identity())
    out_dir = tmp_path / "inference"
    args = [
        "infer", "--source-dir", str(source_dir), "--model-dir", str(model_dir),
        "--out-dir", str(out_dir), "--store-embeddings", "--batch-size", "2",
        "--partition-cells", "2", "--cache-gb", "0.001", "--prefetch-workers", "1",
        "--crop-workers", "1",
    ]
    return SimpleNamespace(
        args=args, out_dir=out_dir, source_dir=source_dir, model_path=model_dir / "model.joblib",
        channel_paths=channel_paths, labels_path=labels_path, cells=cells, manifest=manifest,
        expected_embeddings=expected_embeddings, expected_probabilities=expected_probabilities,
        encoder_type=ChannelSumEncoder,
    )


def _output_bytes(out_dir):
    return {path.relative_to(out_dir): path.read_bytes() for path in out_dir.rglob("*") if path.is_file()}


@pytest.mark.integration
def test_infer_cli_publishes_ordered_predictions_embeddings_and_resumes(inference_run, monkeypatch) -> None:
    run = inference_run
    args, out_dir, cells = run.args, run.out_dir, run.cells
    manifest = run.manifest
    expected_embeddings, expected_probabilities = run.expected_embeddings, run.expected_probabilities
    result = CliRunner().invoke(zarr_cli.main, args)
    assert result.exit_code == 0, result.exception
    prediction_paths = sorted((out_dir / "predictions").glob("*.parquet"))
    assert len(prediction_paths) == 2
    predictions = pd.concat([pd.read_parquet(path) for path in prediction_paths], ignore_index=True)
    assert predictions["cell"].tolist() == cells["cell"].tolist()
    for target in ("brdu", "edu"):
        np.testing.assert_allclose(predictions[f"p_{target}"], expected_probabilities[target], rtol=1e-6)
        assert predictions[target].tolist() == (expected_probabilities[target] >= 0.5).tolist()
    for index, path in enumerate(prediction_paths):
        metadata, count = read_parquet_artifact(path)
        assert count == 2
        assert metadata["source_sha256"] == manifest["source_sha256"]
        assert metadata["feature_precision"] == "bfloat16"
        assert metadata["partition_index"] == index
        saved = read_embedding_artifact(out_dir / "embeddings" / f"{path.stem}.pt")
        partition = cells.iloc[index * 2 : (index + 1) * 2]
        torch.testing.assert_close(saved["cell_sha256"], ordered_cell_sha256(partition["cell"].tolist()))
        for target in ("brdu", "edu"):
            assert saved[target].dtype == torch.bfloat16
            torch.testing.assert_close(saved[target], expected_embeddings[target][index * 2 : (index + 1) * 2])
    complete = json.loads((out_dir / "complete.json").read_text())
    assert complete["cells"] == 4
    before = {path.relative_to(out_dir): path.read_bytes() for path in out_dir.rglob("*") if path.is_file()}
    monkeypatch.setattr(
        zarr_infer, "load_cell_dino_model",
        lambda **kwargs: pytest.fail("completed inference must not load the encoder"),
    )

    resumed = CliRunner().invoke(zarr_cli.main, args)

    assert resumed.exit_code == 0, resumed.exception
    assert "status=skipped_complete" in resumed.output
    assert {path.relative_to(out_dir): path.read_bytes() for path in out_dir.rglob("*") if path.is_file()} == before


@pytest.mark.integration
@pytest.mark.parametrize("change", ["cutoff", "prepared_order", "prepared_crop", "request_order", "pixel", "label", "spatial_subtraction", "no_resume"])
def test_infer_rejects_changed_run_without_rewriting(inference_run, change):
    run = inference_run
    args = list(run.args)
    if change == "request_order":
        requested = run.source_dir / "requested.csv"
        run.cells[["cell"]].to_csv(requested, index=False)
        args += ["--cells", str(requested)]
    first = CliRunner().invoke(zarr_cli.main, args)
    assert first.exit_code == 0, first.exception
    before = _output_bytes(run.out_dir)
    if change == "cutoff":
        bundle = joblib.load(run.model_path)
        bundle["brdu_cutoff"] = 0.99
        joblib.dump(bundle, run.model_path)
    elif change == "prepared_order":
        run.cells.iloc[::-1].to_parquet(run.source_dir / "cells.parquet", index=False)
    elif change == "prepared_crop":
        changed = run.cells.copy()
        changed.loc[0, "centroid_x"] += 1
        changed.to_parquet(run.source_dir / "cells.parquet", index=False)
    elif change == "request_order":
        run.cells.iloc[::-1][["cell"]].to_csv(requested, index=False)
    elif change in {"pixel", "label"}:
        path = run.labels_path if change == "label" else run.channel_paths["brdu"]
        array = zarr.open_array(path, mode="r+")
        array[0, 4, 4] = int(array[0, 4, 4]) + 1
    elif change == "spatial_subtraction":
        from sklearn.preprocessing import SplineTransformer

        from dino.spatial_pixels import SpatialPixels
        from dino.spatial_subtraction import SpatialSubtraction

        splines = tuple(SplineTransformer(n_knots=k, include_bias=True, extrapolation="constant").fit([[0], [100]]) for k in (3, 4))
        model_path = run.model_path.parent / "subtraction.joblib"
        SpatialSubtraction(splines, np.array([[0, 100], [0, 100]]), np.full(30, .2), np.zeros(30), .001, .05, {}).save(model_path)
        bundle = joblib.load(run.model_path)
        bundle["spatial_subtraction"] = SpatialPixels(str(model_path), file_sha256(model_path), (1., 1., 1.), (0., 0., 0.)).identity
        joblib.dump(bundle, run.model_path)
    else:
        args += ["--no-resume"]

    result = CliRunner().invoke(zarr_cli.main, args)

    assert result.exit_code != 0
    assert isinstance(result.exception, (ValueError, FileExistsError))
    assert _output_bytes(run.out_dir) == before


@pytest.mark.integration
@pytest.mark.parametrize("missing_embedding", [False, True])
def test_infer_resumes_valid_partial_run(inference_run, monkeypatch, missing_embedding):
    run = inference_run

    class InterruptedEncoder(run.encoder_type):
        calls = 0

        def forward(self, batch):
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("encoder interrupted after first partition")
            return super().forward(batch)

    monkeypatch.setattr(zarr_infer, "load_cell_dino_model", lambda **kwargs: InterruptedEncoder())
    interrupted = CliRunner().invoke(zarr_cli.main, run.args)
    assert isinstance(interrupted.exception, RuntimeError)
    assert "encoder interrupted" in str(interrupted.exception)
    assert not (run.out_dir / "complete.json").exists()
    if missing_embedding:
        (run.out_dir / "embeddings" / "part-000000.pt").unlink()
    before = _output_bytes(run.out_dir)
    assert (run.out_dir / "contract.json").is_file()
    assert len(list((run.out_dir / "predictions").glob("*.parquet"))) == 1
    monkeypatch.setattr(zarr_infer, "load_cell_dino_model", lambda **kwargs: run.encoder_type())

    resumed = CliRunner().invoke(zarr_cli.main, run.args)

    assert resumed.exit_code == 0, resumed.exception
    after = _output_bytes(run.out_dir)
    assert all(after[path] == content for path, content in before.items())
    predictions = pd.concat([pd.read_parquet(path) for path in sorted((run.out_dir / "predictions").glob("*.parquet"))])
    assert predictions["cell"].tolist() == run.cells["cell"].tolist()
    for target in ("brdu", "edu"):
        np.testing.assert_allclose(predictions[f"p_{target}"], run.expected_probabilities[target], rtol=1e-6)


@pytest.mark.integration
@pytest.mark.parametrize("damage", ["metadata", "row_order", "embedding_dtype", "embedding_identity", "embedding_shape", "missing_contract"])
def test_infer_rejects_invalid_published_artifacts(inference_run, damage):
    run = inference_run
    result = CliRunner().invoke(zarr_cli.main, run.args)
    assert result.exit_code == 0, result.exception
    path = run.out_dir / "predictions" / "part-000000.parquet"
    if damage == "metadata":
        table = pq.read_table(path)
        pq.write_table(table.replace_schema_metadata(None), path)
    elif damage == "row_order":
        table = pq.read_table(path)
        column = table.schema.get_field_index("cell")
        pq.write_table(table.set_column(column, "cell", pa.array(table["cell"].to_pylist()[::-1])), path)
    elif damage.startswith("embedding_"):
        path = run.out_dir / "embeddings" / "part-000000.pt"
        saved = torch.load(path, weights_only=True)
        if damage == "embedding_dtype":
            saved["brdu"] = saved["brdu"].float()
        elif damage == "embedding_identity":
            saved["cell_sha256"] = ordered_cell_sha256(["other", "cells"])
        else:
            saved["brdu"], saved["edu"] = saved["brdu"][:1], saved["edu"][:1]
        torch.save(saved, path)
    else:
        (run.out_dir / "contract.json").unlink()
    before = _output_bytes(run.out_dir)

    resumed = CliRunner().invoke(zarr_cli.main, run.args)

    assert resumed.exit_code != 0
    assert isinstance(resumed.exception, ValueError)
    assert _output_bytes(run.out_dir) == before


@pytest.mark.integration
def test_infer_resume_binds_intensity_context_content(inference_run):
    run = inference_run
    props = run.cells.sort_values("label").assign(
        centroid_z=0.0, intensity_brdu_mean=[1.0, 2.0, 4.0, 8.0],
        intensity_edu_mean=[8.0, 4.0, 2.0, 1.0],
    )
    props_path = run.source_dir / "intensities.parquet"
    props.to_parquet(props_path, index=False)
    context = IntensityContext.load(props_path, scale_zyx_um=np.ones(3))
    context_features = context.features_for_labels(run.cells["label"].to_numpy(), neighbors=1)
    bundle = joblib.load(run.model_path)
    bundle.update(
        schema="dino-knn-intensity-v1", intensity_props=str(props_path), intensity_neighbors=1,
        intensity_scale_zyx_um=[1.0, 1.0, 1.0], context_features=["log1p_cell_mean", "log_cell_over_local_mean"],
    )
    for target in ("brdu", "edu"):
        features = np.column_stack((run.expected_embeddings[target].float().numpy(), context_features[target]))
        scaler = StandardScaler().fit(features)
        bundle[f"{target}_feature_scaler"] = scaler
        bundle[f"{target}_model"] = LogisticRegression(random_state=0).fit(scaler.transform(features), [0, 1, 0, 1])
    joblib.dump(bundle, run.model_path)
    first = CliRunner().invoke(zarr_cli.main, run.args)
    assert first.exit_code == 0, first.exception
    before = _output_bytes(run.out_dir)
    props["intensity_brdu_mean"] *= 2
    props.to_parquet(props_path, index=False)

    resumed = CliRunner().invoke(zarr_cli.main, run.args)

    assert isinstance(resumed.exception, ValueError)
    assert "contract mismatch" in str(resumed.exception)
    assert _output_bytes(run.out_dir) == before


def test_inference_defaults_match_bounded_fetch_plan(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dino-zarr infer",
            "--source-dir",
            str(tmp_path / "source"),
            "--model-dir",
            str(tmp_path / "model"),
            "--out-dir",
            str(tmp_path / "output"),
        ],
    )

    config = zarr_infer._parse_args()

    assert config.batch_size == 32
    assert config.prefetch_workers == 8
    assert config.crop_workers == 8
    assert config.partition_cells == 100_000
    assert config.store_embeddings is False


def test_predict_stores_target_embeddings_in_prediction_order(monkeypatch) -> None:
    cells = pd.DataFrame(
        {
            "cell": ["a", "b", "c"],
            "dataset": ["ds"] * 3,
            "roi": ["roi"] * 3,
            "label": [1, 2, 3],
            "plane_z": [0, 0, 0],
            "centroid_y": [0.0, 0.0, 0.0],
            "centroid_x": [0.0, 0.0, 0.0],
        }
    )
    reader = SimpleNamespace(
        labels=SimpleNamespace(chunks=(10, 10, 10)),
        channels={
            "brdu": SimpleNamespace(chunks=(10, 10, 10)),
            "edu": SimpleNamespace(chunks=(10, 10, 10)),
        },
        offset_zyx=(0, 0, 0),
    )
    monkeypatch.setattr(
        zarr_infer,
        "iter_model_image_batches",
        lambda *args, **kwargs: iter([({"brdu": [0, 1], "edu": [2, 3]}, [2, 0])]),
    )
    monkeypatch.setattr(
        zarr_infer,
        "prepare_model_batch",
        lambda images, **kwargs: torch.ones((len(images), 4), dtype=torch.bfloat16),
    )
    feature_batches = iter(
        [
            torch.tensor([[20, 21], [0, 1]], dtype=torch.bfloat16),
            torch.tensor([[22, 23], [2, 3]], dtype=torch.bfloat16),
        ]
    )
    monkeypatch.setattr(
        zarr_infer,
        "extract_normalized_feature_tensor",
        lambda *args, **kwargs: next(feature_batches),
    )

    class Scaler:
        n_features_in_ = 2

        def transform(self, values):
            return values

    class Classifier:
        def predict_proba(self, values):
            probability = np.full(len(values), 0.75)
            return np.column_stack((1 - probability, probability))

    bundle = {
        "input_channel_names_by_target": {"brdu": ["brdu"], "edu": ["edu"]},
        "feature_precision": "bfloat16",
        "brdu_feature_scaler": Scaler(),
        "edu_feature_scaler": Scaler(),
        "brdu_model": Classifier(),
        "edu_model": Classifier(),
        "brdu_cutoff": 0.5,
        "edu_cutoff": 0.5,
    }

    predictions, embeddings = zarr_infer._predict(
        cells,
        bundle=bundle,
        reader=reader,
        model=object(),
        transform=object(),
        device=torch.device("cpu"),
        batch_size=2,
        prefetch_workers=1,
        crop_workers=1,
        store_embeddings=True,
    )

    assert predictions["cell"].tolist() == ["a", "b", "c"]
    assert embeddings is not None
    assert embeddings["brdu"].dtype == torch.bfloat16
    torch.testing.assert_close(
        embeddings["brdu"][[0, 2]],
        torch.tensor([[0, 1], [20, 21]], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        embeddings["edu"][[0, 2]],
        torch.tensor([[2, 3], [22, 23]], dtype=torch.bfloat16),
    )
    assert embeddings["brdu"][1].isnan().all()


def test_predict_appends_knn_intensity_context(monkeypatch) -> None:
    cells = pd.DataFrame(
        {
            "cell": ["a"],
            "dataset": ["ds"],
            "roi": ["roi"],
            "label": [1],
            "plane_z": [0],
            "centroid_y": [0.0],
            "centroid_x": [0.0],
        }
    )
    reader = SimpleNamespace(
        labels=SimpleNamespace(chunks=(1, 1, 1)),
        channels={
            "brdu": SimpleNamespace(chunks=(1, 1, 1)),
            "edu": SimpleNamespace(chunks=(1, 1, 1)),
        },
        offset_zyx=(0, 0, 0),
    )
    context_features = {
        "brdu": [np.log1p(np.array([100.0, 200.0]))],
        "edu": [np.log1p(np.array([300.0, 400.0]))],
    }
    monkeypatch.setattr(
        zarr_infer,
        "iter_model_image_batches",
        lambda *args, **kwargs: iter([({"brdu": [0], "edu": [0]}, [0])]),
    )
    monkeypatch.setattr(
        zarr_infer,
        "prepare_model_batch",
        lambda images, **kwargs: torch.ones((1, 4), dtype=torch.bfloat16),
    )
    features = iter(
        [
            torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
            torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16),
        ]
    )
    monkeypatch.setattr(
        zarr_infer,
        "extract_normalized_feature_tensor",
        lambda *args, **kwargs: next(features),
    )
    transformed: list[np.ndarray] = []

    class Scaler:
        n_features_in_ = 4

        def transform(self, values):
            transformed.append(values.copy())
            return values

    class Classifier:
        def predict_proba(self, values):
            return np.tile([0.25, 0.75], (len(values), 1))

    class Context:
        def features_for_labels(self, labels, *, neighbors):
            assert labels.tolist() == [1]
            assert neighbors == 16
            return {target: np.asarray(values) for target, values in context_features.items()}

    bundle = {
        "input_channel_names_by_target": {"brdu": ["brdu"], "edu": ["edu"]},
        "feature_precision": "bfloat16",
        "context_features": ["log1p_cell_mean", "log_cell_over_local_mean"],
        "intensity_neighbors": 16,
        "brdu_feature_scaler": Scaler(),
        "edu_feature_scaler": Scaler(),
        "brdu_model": Classifier(),
        "edu_model": Classifier(),
        "brdu_cutoff": 0.5,
        "edu_cutoff": 0.5,
    }

    predictions, _ = zarr_infer._predict(
        cells,
        bundle=bundle,
        reader=reader,
        model=object(),
        transform=object(),
        device=torch.device("cpu"),
        batch_size=1,
        prefetch_workers=1,
        crop_workers=1,
        store_embeddings=False,
        intensity_context=Context(),
    )

    np.testing.assert_allclose(
        transformed[0], [[1.0, 2.0, *context_features["brdu"][0]]]
    )
    np.testing.assert_allclose(
        transformed[1], [[3.0, 4.0, *context_features["edu"][0]]]
    )
    assert predictions.loc[0, "p_brdu"] == pytest.approx(0.75)
    assert predictions.loc[0, "p_edu"] == pytest.approx(0.75)


def test_load_cells_preserves_request_order_and_duplicates(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    pd.DataFrame(
        {
            "cell": ["a", "b", "c", "d"],
            "dataset": ["ds"] * 4,
            "roi": ["roi"] * 4,
            "label": [1, 2, 3, 4],
            "plane_z": [5, 1, 11, 3],
            "centroid_y": [10.0, 10.0, 10.0, 10.0],
            "centroid_x": [10.0, 10.0, 10.0, 1_100.0],
        }
    ).to_parquet(source_dir / "cells.parquet")
    requested_path = tmp_path / "requested.csv"
    pd.DataFrame({"cell": ["c", "a", "c"]}).to_csv(requested_path, index=False)
    cfg = SimpleNamespace(source_dir=source_dir, cells=requested_path)

    selected = zarr_infer._load_cells(cfg)

    assert selected["cell"].tolist() == ["c", "a", "c"]
    assert selected["label"].tolist() == [3, 1, 3]


def test_iter_source_partitions_is_bounded_and_ordered(tmp_path) -> None:
    path = tmp_path / "cells.parquet"
    pd.DataFrame({"cell": [f"cell-{index}" for index in range(5)]}).to_parquet(path)

    partitions = list(zarr_infer._iter_source_partitions(path, rows_per_partition=2))

    assert [len(partition) for partition in partitions] == [2, 2, 1]
    assert pd.concat(partitions, ignore_index=True)["cell"].tolist() == [
        "cell-0",
        "cell-1",
        "cell-2",
        "cell-3",
        "cell-4",
    ]


def test_iter_frame_partitions_is_bounded_and_ordered() -> None:
    frame = pd.DataFrame({"cell": [f"cell-{index}" for index in range(5)]})

    partitions = list(zarr_infer._iter_frame_partitions(frame, rows_per_partition=2))

    assert [len(partition) for partition in partitions] == [2, 2, 1]
    assert pd.concat(partitions, ignore_index=True)["cell"].tolist() == [
        "cell-0",
        "cell-1",
        "cell-2",
        "cell-3",
        "cell-4",
    ]


def test_sort_for_io_uses_all_array_chunk_coordinates() -> None:
    cells = pd.DataFrame(
        {
            "cell": ["a", "b", "c"],
            "plane_z": [1, 1, 1],
            "centroid_y": [900.0, 100.0, 100.0],
            "centroid_x": [100.0, 900.0, 100.0],
        }
    )
    reader = SimpleNamespace(
        labels=SimpleNamespace(chunks=(10, 1_000, 1_000)),
        channels={
            "brdu": SimpleNamespace(chunks=(1, 500, 500)),
            "edu": SimpleNamespace(chunks=(1, 250, 250)),
        },
        offset_zyx=(0, 0, 0),
    )

    ordered = zarr_infer.sort_cells_for_io(cells, reader)

    assert ordered["cell"].tolist() == ["c", "b", "a"]
    assert ordered["_row_order"].tolist() == [2, 1, 0]


def test_iter_image_batches_skips_invalid_cells_without_reordering(monkeypatch) -> None:
    ordered = pd.DataFrame(
        {
            "cell": ["a", "b", "c"],
            "label": [1, 2, 3],
            "plane_z": [0, 0, 11],
            "centroid_y": [1.0, 1.0, 1.0],
            "centroid_x": [1.0, 1.0, 1.0],
            "_row_order": [2, 0, 1],
            "_z_chunk": [0, 0, 1],
            "_y_chunk": [0, 0, 0],
            "_x_chunk": [0, 0, 0],
        }
    )

    def read_images(reader, row, **kwargs):
        value = ord(row["cell"]) - ord("a")
        images = {
            "brdu": np.full((4, 2, 2), value, dtype=np.uint8),
            "edu": np.full((4, 2, 2), value + 10, dtype=np.uint8),
        }
        return images, row["cell"] != "b"

    class Reader:
        def __init__(self) -> None:
            self.prefetched: list[list[tuple[int, int, int]]] = []

        def plan_cells(self, cells, *, size):
            return SimpleNamespace(cells=tuple(cells))

        def prefetch_plan(self, plan, *, workers):
            self.prefetched.append(list(plan.cells))

        def plan_fits(self, plan):
            return True

    reader = Reader()
    monkeypatch.setattr(zarr_model, "read_model_images", read_images)
    batches = list(
        zarr_infer.iter_model_image_batches(
            ordered,
            bundle={
                "crop_size": 2,
                "input_channels": 4,
                "highpass_sigma_px": 0.0,
                "mask_erosion_px": 0,
            },
            channel_names_by_target={"brdu": ("brdu",), "edu": ("edu",)},
            reader=reader,
            batch_size=3,
            prefetch_workers=3,
            crop_workers=2,
            require_mask=False,
        )
    )

    assert len(batches) == 1
    images, row_orders = batches[0]
    assert row_orders == [2, 1]
    assert [int(image[0, 0, 0]) for image in images["brdu"]] == [0, 2]
    assert [int(image[0, 0, 0]) for image in images["edu"]] == [10, 12]
    assert [len(cells) for cells in reader.prefetched] == [3]

    with pytest.raises(ValueError, match="Target cell mask is empty for b"):
        list(
            zarr_infer.iter_model_image_batches(
                ordered,
                bundle={
                    "crop_size": 2,
                    "input_channels": 4,
                    "highpass_sigma_px": 0.0,
                    "mask_erosion_px": 0,
                },
                channel_names_by_target={"brdu": ("brdu",), "edu": ("edu",)},
                reader=Reader(),
                batch_size=3,
                prefetch_workers=3,
                crop_workers=2,
                require_mask=True,
            )
        )


def test_select_shard_keeps_label_chunks_intact_and_partitions_all_cells() -> None:
    cells = pd.DataFrame(
        {
            "cell": [f"cell-{index}" for index in range(8)],
            "plane_z": [1, 2, 11, 12, 21, 22, 31, 32],
            "centroid_y": [10.0] * 8,
            "centroid_x": [10.0] * 8,
        }
    )

    shards = [
        zarr_infer._select_shard(cells, chunks=(10, 100, 100), shard_index=index, shard_count=2)
        for index in range(2)
    ]

    assert set(shards[0]["cell"]).isdisjoint(shards[1]["cell"])
    assert set(pd.concat(shards)["cell"]) == set(cells["cell"])
    assert [len(shard) for shard in shards] == [4, 4]
    assignments = {
        cell: shard_index
        for shard_index, shard in enumerate(shards)
        for cell in shard["cell"]
    }
    for first, second in (("cell-0", "cell-1"), ("cell-2", "cell-3")):
        assert assignments[first] == assignments[second]
    for shard in shards:
        assert shard["plane_z"].tolist() == sorted(shard["plane_z"].tolist())


def test_float32_embedding_storage_rejected_before_output_or_encoder(inference_run, monkeypatch) -> None:
    run = inference_run
    bundle = joblib.load(run.model_path)
    bundle["feature_precision"] = "float32"
    joblib.dump(bundle, run.model_path)
    monkeypatch.setattr(
        zarr_infer, "load_cell_dino_model",
        lambda **kwargs: pytest.fail("unsupported storage must fail before loading the encoder"),
    )

    result = CliRunner().invoke(zarr_cli.main, run.args)

    assert isinstance(result.exception, ValueError)
    assert "bfloat16" in str(result.exception)
    assert not run.out_dir.exists()


def test_float32_prediction_without_embedding_storage(inference_run) -> None:
    run = inference_run
    bundle = joblib.load(run.model_path)
    bundle["feature_precision"] = "float32"
    joblib.dump(bundle, run.model_path)
    args = [arg for arg in run.args if arg != "--store-embeddings"]

    result = CliRunner().invoke(zarr_cli.main, args)

    assert result.exit_code == 0, result.exception
    paths = sorted((run.out_dir / "predictions").glob("*.parquet"))
    predictions = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    assert predictions["cell"].tolist() == run.cells["cell"].tolist()
    assert np.isfinite(predictions[["p_brdu", "p_edu"]]).all().all()
    assert json.loads((run.out_dir / "complete.json").read_text())["feature_precision"] == "float32"
