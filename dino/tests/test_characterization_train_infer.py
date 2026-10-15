"""Characterization of direct-Zarr training, features, inference, and provenance.

These tests pin today's BrdU/EdU behavior so the marker-config refactor can prove that the
default configuration leaves persisted names, hashes, cache keys, and outputs unchanged, and that
artifacts written by today's code (``fixtures/legacy_model``) keep loading and resuming.
Calls into APIs that the refactor will change live in the helpers below; assertions do not.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from click.testing import CliRunner
from dino_markers import DEFAULT_CONTRACT
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from dino import (
    call_provenance,
    classifier_features,
    clean_labels,
    cortex_score,
    review_contracts,
    training_split,
    zarr_cli,
    zarr_infer,
    zarr_train,
)
from dino.artifacts import (
    PARQUET_METADATA_KEY,
    file_sha256,
    ordered_cell_sha256,
    path_sha256,
    read_embedding_artifact,
    read_parquet_artifact,
    write_embedding_atomic,
)
from dino.intensity_context import IntensityContext
from dino.zarr_source import load_source, manifest_digest


FIXTURE = Path(__file__).parent / "fixtures" / "legacy_model"
INFERENCE_RUN = Path("inference") / "shard-000"


class StubEncoder(torch.nn.Module):
    """Deterministic stand-in for Cell-DINO: slot sums plus a constant component."""

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        sums = batch[:, :2].sum(dim=(2, 3))
        return torch.cat([sums, torch.full_like(sums[:, :1], 300.0)], dim=1)


# --- Invocation helpers: the only places that name APIs the marker-config refactor changes. ---


def _train_cli(*, barrage_dir: Path, out_dir: Path, feature_cache: Path | None = None) -> None:
    """Train with the default input channels (no per-target input-channel flags)."""
    args = [
        "train",
        "--barrage-dir", str(barrage_dir),
        "--out-dir", str(out_dir),
        "--dinov2-repo", "encoder",
        "--weights", "encoder/weights.pth",
        "--crop-size", "4",
        "--resize-size", "4",
        "--crop-eval-size", "4",
        "--batch-size", "4",
        "--test-fraction", "0.25",
        "--cache-gb", "0.001",
        "--prefetch-workers", "1",
        "--crop-workers", "1",
    ]
    if feature_cache is not None:
        args += ["--feature-cache", str(feature_cache)]
    result = CliRunner().invoke(zarr_cli.main, args)
    assert result.exit_code == 0, result.exception


def _infer_args(*, out_dir: Path) -> list[str]:
    return [
        "infer",
        "--source-dir", ".",
        "--model-dir", "model",
        "--out-dir", str(out_dir),
        "--store-embeddings",
        "--batch-size", "4",
        "--partition-cells", "8",
        "--cache-gb", "0.001",
        "--prefetch-workers", "1",
        "--crop-workers", "1",
    ]


def _invoke_infer(*, out_dir: Path):
    return CliRunner().invoke(zarr_cli.main, _infer_args(out_dir=out_dir))


def _infer_cli(*, out_dir: Path):
    result = _invoke_infer(out_dir=out_dir)
    assert result.exit_code == 0, result.exception
    return result


def _load_source(source_dir: Path):
    return load_source(source_dir)


def _read_embeddings(path: Path) -> dict[str, torch.Tensor]:
    return read_embedding_artifact(path)


def _intensity_features(path: Path, labels: np.ndarray, *, neighbors: int):
    context = IntensityContext.load(path, scale_zyx_um=np.ones(3), markers=DEFAULT_CONTRACT.markers)
    return context, context.features_for_labels(labels, neighbors=neighbors)


def _score_cortex(*, root: Path, output_dir: Path) -> dict:
    return cortex_score.score_cortex(
        model=root / "model.joblib",
        embedding_model=root / "embedding-model.joblib",
        embedding_run=root / "infer",
        cortex_props=root / "cortex.parquet",
        labeled_cells=root / "labeled.csv",
        output_dir=output_dir,
        workers=1,
    )


def _patch_encoder(monkeypatch, encoder_factory) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for module in (zarr_train, zarr_infer):
        monkeypatch.setattr(module, "load_cell_dino_model", lambda **kwargs: encoder_factory())
        monkeypatch.setattr(module, "build_cell_transform", lambda **kwargs: torch.nn.Identity())


def _fail_encoder():
    pytest.fail("a reusable legacy artifact must not load the encoder")


def _load_inference_bundle(path: Path) -> dict:
    return zarr_infer._load_bundle(path)


def _feature_cache_key(target: str, row: pd.Series) -> str:
    return zarr_train._feature_key(target, row)


def _split(rows: pd.DataFrame, *, limit: int, test_fraction: float, seed: int):
    return training_split.split_training_rows(
        rows, markers=DEFAULT_CONTRACT.markers, limit=limit, test_fraction=test_fraction, seed=seed
    )


def _assemble(embeddings, contexts, *, target: str, mode: str) -> np.ndarray:
    return classifier_features.assemble_classifier_features(
        embeddings, contexts, target=target, mode=mode, markers=DEFAULT_CONTRACT.markers
    )


def _validate_feature_contract(bundle: dict, embedding_bundle: dict) -> None:
    classifier_features.validate_feature_contract(bundle, embedding_bundle)


def _combined_uncertainty(p_brdu, p_edu, *, brdu_cutoff: float, edu_cutoff: float) -> np.ndarray:
    return review_contracts.combined_uncertainty(
        {"brdu": p_brdu, "edu": p_edu}, {"brdu": brdu_cutoff, "edu": edu_cutoff}
    )


def _score_rows(rows: pd.DataFrame, *, embeddings, bundle, context) -> pd.DataFrame:
    return cortex_score.score_rows(
        rows, embeddings=embeddings, bundle=bundle, context=context, markers=DEFAULT_CONTRACT.markers
    )


def _load_calls(inference: Path, expected_cells: list[np.ndarray], source_hash: str):
    calls, records = call_provenance.load_calls(inference, expected_cells, source_hash)
    return calls["brdu"], calls["edu"], records


def _clean_labels_cli(monkeypatch, barrage_dir: Path) -> None:
    monkeypatch.setattr(sys, "argv", ["clean-labels", "--barrage-dir", str(barrage_dir), "--apply"])
    clean_labels.main()


# --- Frozen legacy artifacts (fixtures/legacy_model, produced by today's code). ---

SOURCE_SHA256 = "e0bfc8407b5363493fb2ec81125e82c03cd5027428387a9a06c72032c25b97c3"
WEIGHTS_SHA256 = "46ab11e9f4af11f73e4cd02ee058d4363c6d00cafe6b0b299cb2796e5961bb39"
MODEL_SOURCE_SHA256 = "53525f1db56678fdf6ea90bc9d41545a87a09fbacad7c1ba30d05f26c2071de3"
MODEL_SHA256 = "145d8e9230373b45923523129aaf3b7af1682d786946d4872b0cc8de1f1748a7"
BRDU_CUTOFF = 0.6463847160339355
EDU_CUTOFF = 0.5182463526725769

EXPECTED_BUNDLE_KEYS = {
    "schema",
    "created_at_utc",
    "source_sha256",
    "training_source_sha256s",
    "training_barrage_dirs",
    "feature_extractor",
    "input_channels",
    "input_channel_names_by_target",
    "weights",
    "weights_sha256",
    "dinov2_repo",
    "model_source_sha256",
    "crop_size",
    "resize_size",
    "crop_eval_size",
    "highpass_sigma_px",
    "mask_erosion_px",
    "feature_precision",
    "feature_execution",
    "brdu_feature_scaler",
    "edu_feature_scaler",
    "brdu_model",
    "edu_model",
    "train_cells",
    "validation_cells",
    "split_seed",
    "test_fraction",
    "brdu_best_c",
    "brdu_cv_balanced_accuracy",
    "brdu_cv_folds",
    "brdu_cutoff",
    "brdu_validation_balanced_accuracy",
    "edu_best_c",
    "edu_cv_balanced_accuracy",
    "edu_cv_folds",
    "edu_cutoff",
    "edu_validation_balanced_accuracy",
}

EXPECTED_BUNDLE_VALUES = {
    "schema": "dino-zarr-model-v6",
    "source_sha256": SOURCE_SHA256,
    "training_source_sha256s": [SOURCE_SHA256],
    "feature_extractor": "cell_dino_hpa_vitl16",
    "input_channels": 4,
    "input_channel_names_by_target": {"brdu": ["brdu", "edu"], "edu": ["brdu", "edu"]},
    "weights": "encoder/weights.pth",
    "weights_sha256": WEIGHTS_SHA256,
    "dinov2_repo": "encoder",
    "model_source_sha256": MODEL_SOURCE_SHA256,
    "crop_size": 4,
    "resize_size": 4,
    "crop_eval_size": 4,
    "highpass_sigma_px": 0.0,
    "mask_erosion_px": 0,
    "feature_precision": "bfloat16",
    "feature_execution": "native_sdpa",
    "train_cells": [
        "roi|2", "roi|3", "roi|4", "roi|5", "roi|6", "roi|8",
        "roi|9", "roi|10", "roi|11", "roi|12", "roi|13", "roi|15",
    ],
    "validation_cells": ["roi|1", "roi|7", "roi|14", "roi|16"],
    "split_seed": 0,
    "test_fraction": 0.25,
    "brdu_best_c": 0.1,
    "brdu_cv_balanced_accuracy": 1.0,
    "brdu_cv_folds": 5,
    "brdu_cutoff": BRDU_CUTOFF,
    "brdu_validation_balanced_accuracy": 1.0,
    "edu_best_c": 0.1778279410038923,
    "edu_cv_balanced_accuracy": 1.0,
    "edu_cv_folds": 5,
    "edu_cutoff": EDU_CUTOFF,
    "edu_validation_balanced_accuracy": 0.75,
}

EXPECTED_CACHE_METADATA = {
    "schema": "dino-zarr-features-v2",
    "source_sha256s": [SOURCE_SHA256],
    "weights_sha256": WEIGHTS_SHA256,
    "model_source_sha256": MODEL_SOURCE_SHA256,
    "hub_model": "cell_dino_hpa_vitl16",
    "crop_size": 4,
    "resize_size": 4,
    "crop_eval_size": 4,
    "highpass_sigma_px": 0.0,
    "mask_erosion_px": 0,
    "feature_precision": "bfloat16",
    "feature_execution": "native_sdpa",
    "input_channel_names_by_target": {"brdu": ["brdu", "edu"], "edu": ["brdu", "edu"]},
}

EXPECTED_CONTRACT = {
    "artifact": "dino-zarr-infer-v2",
    "brdu_cutoff": BRDU_CUTOFF,
    "cells": 16,
    "edu_cutoff": EDU_CUTOFF,
    "embedding_dimension": 3,
    "embedding_dtype": "bfloat16",
    "embedding_format": "pytorch-tensor-v1",
    "embedding_targets": ["brdu", "edu"],
    "feature_execution": "native_sdpa",
    "feature_precision": "bfloat16",
    "model_schema": "dino-zarr-model-v6",
    "model_sha256": MODEL_SHA256,
    "model_source_sha256": MODEL_SOURCE_SHA256,
    "partition_cells": 8,
    "prepared_cells_sha256": "fdb006ed384a08aabb5740f8911e3402c563cc1c11265471c7ef0eebe791b634",
    "requested_cells_sha256": "118a18e0b3fae7f2923d55f70cbb4a9e235de69e5f90b0d1f3750d38dca272be",
    "requested_input_sha256": None,
    "shard_count": 1,
    "shard_index": 0,
    "source_cells": 16,
    "source_sha256": SOURCE_SHA256,
    "stores_embeddings": True,
}

EXPECTED_PARTITIONS = [
    {
        "cell_sha256": "b8e7f08c0fb8b4615d758932ec03f1f05595d7a5e43229a837a7e4d4ff5c306f",
        "cells": ["roi|4", "roi|11", "roi|7", "roi|9", "roi|2", "roi|15", "roi|1", "roi|8"],
        "p_brdu": [
            0.5546132922172546, 0.6840863227844238, 0.646948516368866, 0.3648551404476166,
            0.3606303632259369, 0.7159940004348755, 0.295343816280365, 0.584473192691803,
        ],
        "p_edu": [
            0.677535891532898, 0.3004278242588043, 0.3556016683578491, 0.2977386713027954,
            0.7822545170783997, 0.258625328540802, 0.5456419587135315, 0.6278705596923828,
        ],
        "brdu": [False, True, True, False, False, True, False, False],
        "edu": [True, False, False, False, True, False, True, True],
    },
    {
        "cell_sha256": "ffdfacb8bd69ae9cb41b4d670c6d56e82a1830fb6ed30ccb61efd0c96f5c6411",
        "cells": ["roi|5", "roi|14", "roi|16", "roi|3", "roi|13", "roi|6", "roi|10", "roi|12"],
        "p_brdu": [
            0.32218143343925476, 0.4176429808139801, 0.6463847160339355, 0.6066914200782776,
            0.4260629117488861, 0.3739947974681854, 0.3918663561344147, 0.6145358681678772,
        ],
        "p_edu": [
            0.4223399758338928, 0.6272733211517334, 0.5182463526725769, 0.42230820655822754,
            0.19838839769363403, 0.744060218334198, 0.6938776969909668, 0.5745447874069214,
        ],
        "brdu": [False, False, True, False, False, False, False, False],
        "edu": [False, True, True, False, False, True, True, True],
    },
]

DEFAULT_JSON = {"version": 1, "reference": "pi", "markers": ["brdu", "edu"]}
PREDICTION_COLUMNS = ["cell", "dataset", "roi", "label", "p_brdu", "p_edu", "brdu", "edu"]
PREDICTION_TYPES = {
    "cell": pa.string(),
    "dataset": pa.string(),
    "roi": pa.string(),
    "label": pa.int64(),
    "p_brdu": pa.float32(),
    "p_edu": pa.float32(),
    "brdu": pa.bool_(),
    "edu": pa.bool_(),
}


@pytest.fixture
def legacy(tmp_path, monkeypatch) -> Path:
    """A writable copy of the frozen fixture as the working directory (its paths are relative)."""
    root = tmp_path / "legacy_model"
    shutil.copytree(FIXTURE, root)
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    return root


def _tree_bytes(root: Path) -> dict[Path, bytes]:
    return {path.relative_to(root): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def _assert_same_estimator(left, right) -> None:
    assert type(left) is type(right)
    for attribute in ("mean_", "scale_", "var_", "coef_", "intercept_", "classes_"):
        if hasattr(right, attribute):
            np.testing.assert_array_equal(getattr(left, attribute), getattr(right, attribute))
    assert left.n_features_in_ == right.n_features_in_


def _assert_reproduces_legacy_bundle(bundle: dict, legacy_bundle: dict, *, barrage_dir: Path) -> None:
    assert set(legacy_bundle) == EXPECTED_BUNDLE_KEYS
    assert set(bundle) == EXPECTED_BUNDLE_KEYS | {"marker_contract"}
    assert bundle["schema"] == "dino-zarr-model-v7"
    assert bundle["marker_contract"] == DEFAULT_JSON
    assert bundle["training_barrage_dirs"] == [str(barrage_dir.resolve())]
    pd.Timestamp(bundle["created_at_utc"])
    for key in EXPECTED_BUNDLE_KEYS - {"schema", "created_at_utc", "training_barrage_dirs"}:
        if key.endswith(("_model", "_feature_scaler")):
            _assert_same_estimator(bundle[key], legacy_bundle[key])
        else:
            assert bundle[key] == legacy_bundle[key], key


def test_legacy_bundle_keys_values_and_identities(legacy) -> None:
    bundle = joblib.load(legacy / "model" / "model.joblib")

    assert set(bundle) == EXPECTED_BUNDLE_KEYS
    assert {key: bundle[key] for key in EXPECTED_BUNDLE_VALUES} == EXPECTED_BUNDLE_VALUES
    assert len(bundle["training_barrage_dirs"]) == 1
    pd.Timestamp(bundle["created_at_utc"])
    for target in ("brdu", "edu"):
        assert isinstance(bundle[f"{target}_model"], LogisticRegression)
        assert isinstance(bundle[f"{target}_feature_scaler"], StandardScaler)
        assert bundle[f"{target}_feature_scaler"].n_features_in_ == 3
    manifest = json.loads((legacy / "source.json").read_text())
    assert manifest["source_sha256"] == manifest_digest(manifest) == bundle["source_sha256"]
    assert _load_source(Path(".")).manifest["source_sha256"] == SOURCE_SHA256
    assert file_sha256(Path(bundle["weights"])) == bundle["weights_sha256"]
    assert path_sha256(Path(bundle["dinov2_repo"])) == bundle["model_source_sha256"]
    assert file_sha256(legacy / "model" / "model.joblib") == MODEL_SHA256


def test_legacy_bundle_passes_inference_validation(legacy) -> None:
    bundle = _load_inference_bundle(legacy / "model" / "model.joblib")

    assert bundle["schema"] == "dino-zarr-model-v6"
    assert (bundle["brdu_cutoff"], bundle["edu_cutoff"]) == (BRDU_CUTOFF, EDU_CUTOFF)


def test_legacy_feature_cache_metadata_and_entry_keys(legacy) -> None:
    cache = joblib.load(legacy / "model" / "features.joblib")
    index = pd.read_csv(legacy / "index.csv")

    assert set(cache) == {"metadata", "entries"}
    assert cache["metadata"] == EXPECTED_CACHE_METADATA
    expected_keys = {
        hashlib.sha1(
            f"{target}|{row['cell']}|{row['label']}|{row['plane_z']}|"
            f"{row['centroid_y']}|{row['centroid_x']}".encode()
        ).hexdigest(): (target, row["cell"])
        for _, row in index.iterrows()
        for target in ("brdu", "edu")
    }
    assert set(cache["entries"]) == set(expected_keys)
    for key, entry in cache["entries"].items():
        assert set(entry) == {"feature"}
        assert entry["feature"].dtype == np.float32
        assert entry["feature"].shape == (3,)
    first = index.iloc[0]
    assert first["cell"] == "roi|1"
    assert _feature_cache_key("brdu", first) == "f13918526b88e9bcc0f3b2a0dd01dedaddc175c4"
    assert _feature_cache_key("edu", first) == "1ae600422a1388a383f0da8189b9b0556504b37d"
    assert expected_keys["f13918526b88e9bcc0f3b2a0dd01dedaddc175c4"] == ("brdu", "roi|1")


def test_feature_cache_key_payload_format() -> None:
    row = pd.Series({"cell": "roi|7", "label": 7, "plane_z": 2, "centroid_y": 3.5, "centroid_x": 4.0})

    assert _feature_cache_key("edu", row) == hashlib.sha1(b"edu|roi|7|7|2|3.5|4.0").hexdigest()


def test_training_reuses_legacy_feature_cache_without_encoder(legacy, monkeypatch) -> None:
    cache_path = legacy / "model" / "features.joblib"
    cache_bytes = cache_path.read_bytes()
    _patch_encoder(monkeypatch, _fail_encoder)

    _train_cli(barrage_dir=Path("."), out_dir=Path("retrain"), feature_cache=Path("model/features.joblib"))

    assert cache_path.read_bytes() == cache_bytes
    assert sorted(path.name for path in (legacy / "retrain").iterdir()) == ["model.joblib"]
    _assert_reproduces_legacy_bundle(
        joblib.load(legacy / "retrain" / "model.joblib"),
        joblib.load(legacy / "model" / "model.joblib"),
        barrage_dir=legacy,
    )


def test_fresh_training_reproduces_legacy_cache_and_bundle(legacy, monkeypatch) -> None:
    _patch_encoder(monkeypatch, lambda: StubEncoder().eval())

    _train_cli(barrage_dir=Path("."), out_dir=Path("fresh"))

    assert sorted(path.name for path in (legacy / "fresh").iterdir()) == ["features.joblib", "model.joblib"]
    cache = joblib.load(legacy / "fresh" / "features.joblib")
    legacy_cache = joblib.load(legacy / "model" / "features.joblib")
    assert legacy_cache["metadata"] == EXPECTED_CACHE_METADATA
    assert cache["metadata"] == {
        **EXPECTED_CACHE_METADATA, "schema": "dino-zarr-features-v3", "marker_contract": DEFAULT_JSON
    }
    assert set(cache["entries"]) == set(legacy_cache["entries"])
    for key, entry in legacy_cache["entries"].items():
        np.testing.assert_array_equal(cache["entries"][key]["feature"], entry["feature"])
    _assert_reproduces_legacy_bundle(
        joblib.load(legacy / "fresh" / "model.joblib"),
        joblib.load(legacy / "model" / "model.joblib"),
        barrage_dir=legacy,
    )


def test_legacy_inference_run_contract_and_layout(legacy) -> None:
    run = legacy / INFERENCE_RUN
    contract = json.loads((run / "contract.json").read_text())
    complete = json.loads((run / "complete.json").read_text())
    cells = pd.read_parquet(legacy / "cells.parquet")["cell"].astype(str).tolist()

    assert sorted(str(path.relative_to(run)) for path in run.rglob("*") if path.is_file()) == [
        "complete.json",
        "contract.json",
        "embeddings/part-000000.pt",
        "embeddings/part-000001.pt",
        "predictions/part-000000.parquet",
        "predictions/part-000001.parquet",
    ]
    assert contract == complete == EXPECTED_CONTRACT
    assert contract["model_sha256"] == file_sha256(legacy / "model" / "model.joblib")
    assert contract["prepared_cells_sha256"] == file_sha256(legacy / "cells.parquet")
    assert contract["requested_cells_sha256"] == bytes(ordered_cell_sha256(cells).tolist()).hex()
    assert contract["source_sha256"] == _load_source(Path(".")).manifest["source_sha256"]


def test_legacy_prediction_partitions_columns_values_and_provenance(legacy) -> None:
    run = legacy / INFERENCE_RUN
    bundle = joblib.load(legacy / "model" / "model.joblib")
    for index, expected in enumerate(EXPECTED_PARTITIONS):
        path = run / "predictions" / f"part-{index:06d}.parquet"
        table = pq.read_table(path)
        frame = table.to_pandas()
        provenance, rows = read_parquet_artifact(path)

        assert list(frame.columns) == PREDICTION_COLUMNS
        assert {name: table.schema.field(name).type for name in PREDICTION_COLUMNS} == PREDICTION_TYPES
        assert PARQUET_METADATA_KEY == b"fishtools.dino"
        assert json.loads(table.schema.metadata[PARQUET_METADATA_KEY]) == provenance
        assert rows == 8
        assert provenance == {
            **EXPECTED_CONTRACT,
            "partition_index": index,
            "partition_rows": 8,
            "cell_sha256": expected["cell_sha256"],
        }
        assert expected["cell_sha256"] == bytes(ordered_cell_sha256(expected["cells"]).tolist()).hex()
        assert frame["cell"].tolist() == expected["cells"]
        assert frame["dataset"].tolist() == ["ds"] * 8
        assert frame["label"].tolist() == [int(cell.split("|")[1]) for cell in expected["cells"]]
        np.testing.assert_array_equal(frame["p_brdu"], np.asarray(expected["p_brdu"], dtype=np.float32))
        np.testing.assert_array_equal(frame["p_edu"], np.asarray(expected["p_edu"], dtype=np.float32))
        assert frame["brdu"].tolist() == expected["brdu"]
        assert frame["edu"].tolist() == expected["edu"]

        embeddings = _read_embeddings(run / "embeddings" / f"part-{index:06d}.pt")
        assert set(embeddings) == {"brdu", "edu", "cell_sha256"}
        assert bytes(embeddings["cell_sha256"].tolist()).hex() == expected["cell_sha256"]
        for target in ("brdu", "edu"):
            assert embeddings[target].dtype == torch.bfloat16
            assert tuple(embeddings[target].shape) == (8, 3)
            features = embeddings[target].float().numpy()
            probability = bundle[f"{target}_model"].predict_proba(
                bundle[f"{target}_feature_scaler"].transform(features)
            )[:, 1]
            np.testing.assert_allclose(frame[f"p_{target}"], probability, rtol=1e-6)
            assert frame[target].tolist() == (probability >= bundle[f"{target}_cutoff"]).tolist()


def test_completed_legacy_inference_resumes_as_complete(legacy, monkeypatch) -> None:
    before = _tree_bytes(legacy)
    _patch_encoder(monkeypatch, _fail_encoder)

    result = _infer_cli(out_dir=INFERENCE_RUN)

    assert "status=skipped_complete" in result.output
    assert _tree_bytes(legacy) == before


def test_interrupted_legacy_inference_resumes_missing_partition(legacy, monkeypatch) -> None:
    run = legacy / INFERENCE_RUN
    frozen = legacy.parent / "frozen-run"
    shutil.copytree(run, frozen)
    for name in ("complete.json", "predictions/part-000001.parquet", "embeddings/part-000001.pt"):
        (run / name).unlink()
    kept = _tree_bytes(run)
    _patch_encoder(monkeypatch, lambda: StubEncoder().eval())

    result = _infer_cli(out_dir=INFERENCE_RUN)

    assert "status=complete" in result.output
    after = _tree_bytes(run)
    assert {path: after[path] for path in kept} == kept
    assert json.loads((run / "complete.json").read_text()) == EXPECTED_CONTRACT
    part = run / "predictions" / "part-000001.parquet"
    pd.testing.assert_frame_equal(pd.read_parquet(part), pd.read_parquet(frozen / "predictions" / part.name))
    assert read_parquet_artifact(part) == read_parquet_artifact(frozen / "predictions" / part.name)
    resumed = _read_embeddings(run / "embeddings" / "part-000001.pt")
    saved = _read_embeddings(frozen / "embeddings" / "part-000001.pt")
    assert set(resumed) == set(saved)
    for key in saved:
        assert torch.equal(resumed[key], saved[key])


def test_legacy_inference_rejects_changed_stored_contract(legacy, monkeypatch) -> None:
    contract_path = legacy / INFERENCE_RUN / "contract.json"
    contract = json.loads(contract_path.read_text())
    contract["brdu_cutoff"] = 0.5
    contract_path.write_text(json.dumps(contract))
    _patch_encoder(monkeypatch, _fail_encoder)

    result = _invoke_infer(out_dir=INFERENCE_RUN)

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "contract mismatch" in str(result.exception)


def test_fresh_inference_on_legacy_bundle_reproduces_legacy_run(legacy, monkeypatch) -> None:
    _patch_encoder(monkeypatch, lambda: StubEncoder().eval())

    result = _infer_cli(out_dir=Path("rerun"))

    assert "status=complete" in result.output
    run, rerun = legacy / INFERENCE_RUN, legacy / "rerun"
    stamped = {"artifact": "dino-zarr-infer-v3", "marker_contract": DEFAULT_JSON}
    for name in ("contract.json", "complete.json"):
        assert json.loads((rerun / name).read_text()) == {**json.loads((run / name).read_text()), **stamped}
    for index in range(2):
        part = f"part-{index:06d}"
        pd.testing.assert_frame_equal(
            pd.read_parquet(rerun / "predictions" / f"{part}.parquet"),
            pd.read_parquet(run / "predictions" / f"{part}.parquet"),
        )
        provenance, rows = read_parquet_artifact(run / "predictions" / f"{part}.parquet")
        assert read_parquet_artifact(rerun / "predictions" / f"{part}.parquet") == ({**provenance, **stamped}, rows)
        saved = _read_embeddings(run / "embeddings" / f"{part}.pt")
        rerun_saved = _read_embeddings(rerun / "embeddings" / f"{part}.pt")
        assert set(rerun_saved) == set(saved)
        for key in saved:
            assert torch.equal(rerun_saved[key], saved[key])


def test_call_provenance_loads_legacy_run(legacy) -> None:
    cells = pd.read_parquet(legacy / "cells.parquet")["cell"].to_numpy()

    brdu, edu, records = _load_calls(legacy / "inference", [cells], SOURCE_SHA256)

    assert brdu.tolist() == EXPECTED_PARTITIONS[0]["brdu"] + EXPECTED_PARTITIONS[1]["brdu"]
    assert edu.tolist() == EXPECTED_PARTITIONS[0]["edu"] + EXPECTED_PARTITIONS[1]["edu"]
    assert len(records) == 2
    for index, record in enumerate(records):
        path = legacy / INFERENCE_RUN / "predictions" / f"part-{index:06d}.parquet"
        assert set(record) == {"path", "sha256", "metadata"}
        assert record["path"] == str(path)
        assert record["sha256"] == file_sha256(path)
        assert record["metadata"] == read_parquet_artifact(path)[0]


# --- In-memory contracts. ---


def _cells(n: int) -> list[str]:
    return [f"c{i}" for i in range(n)]


@pytest.mark.parametrize(
    "brdu, edu, train_cells, validation_cells",
    [
        pytest.param(
            [0, 0, 1, 1] * 4,
            [0, 1, 0, 1] * 4,
            ["c1", "c2", "c3", "c4", "c5", "c7", "c8", "c9", "c10", "c11", "c12", "c14"],
            ["c0", "c6", "c13", "c15"],
            id="joint",
        ),
        pytest.param(
            [0] * 6 + [1] * 6,
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
            ["c1", "c2", "c3", "c4", "c6", "c7", "c8", "c9", "c10"],
            ["c0", "c5", "c11"],
            id="brdu-fallback",
        ),
        pytest.param(
            [1, 1, 1, 1, 1, 0, 0, 1, 0, 1],
            [1, 1, 0, 0, 1, 1, 0, 1, 0, 1],
            ["c0", "c1", "c3", "c4", "c6", "c7", "c8"],
            ["c2", "c5", "c9"],
            id="edu-fallback",
        ),
    ],
)
def test_training_split_stratifies_joint_then_brdu_then_edu(brdu, edu, train_cells, validation_cells) -> None:
    rows = pd.DataFrame({"cell": _cells(len(brdu)), "brdu": brdu, "edu": edu})

    training, validation = _split(rows, limit=0, test_fraction=0.25, seed=0)

    assert training["cell"].tolist() == train_cells
    assert validation["cell"].tolist() == validation_cells


def test_clean_labels_writes_canonical_columns(monkeypatch, tmp_path) -> None:
    (tmp_path / "labels.csv").write_bytes(
        b"note,edu,cell,brdu,extra\nx,1.0,r|2,0,a\ny,0,r|1,0,b\nz,0,r|1,1,c\n"
    )

    _clean_labels_cli(monkeypatch, tmp_path)

    assert (tmp_path / "labels.csv").read_bytes() == b"cell,brdu,edu\nr|1,1,0\nr|2,0,1\n"


def test_combined_uncertainty_values_including_exact_cutoff() -> None:
    p_brdu = np.array([0.25, 0.5, 0.0, 0.75, 0.9])
    p_edu = np.array([0.1, 1.0, 0.5, 0.25, 0.75])

    result = _combined_uncertainty(p_brdu, p_edu, brdu_cutoff=0.25, edu_cutoff=0.75)

    np.testing.assert_array_equal(result, [np.inf, 4.0, 4.0, 2.0, np.inf])
    assert result.dtype == np.float64


@pytest.mark.parametrize(
    "p_brdu, p_edu, brdu_cutoff, edu_cutoff",
    [
        pytest.param([0.5], [0.5, 0.5], 0.5, 0.5, id="length-mismatch"),
        pytest.param([], [], 0.5, 0.5, id="empty"),
        pytest.param([np.nan], [0.5], 0.5, 0.5, id="nonfinite"),
        pytest.param([0.5], [1.5], 0.5, 0.5, id="out-of-range"),
        pytest.param([0.5], [0.5], 0.0, 0.5, id="brdu-cutoff-at-zero"),
        pytest.param([0.5], [0.5], 0.5, 1.0, id="edu-cutoff-at-one"),
    ],
)
def test_combined_uncertainty_rejects_invalid_inputs(p_brdu, p_edu, brdu_cutoff, edu_cutoff) -> None:
    with pytest.raises(ValueError):
        _combined_uncertainty(np.asarray(p_brdu), np.asarray(p_edu), brdu_cutoff=brdu_cutoff, edu_cutoff=edu_cutoff)


def test_joint_features_concatenate_embeddings_then_contexts_brdu_first() -> None:
    embeddings = {"brdu": np.full((2, 2), 1.0), "edu": np.full((2, 2), 2.0)}
    contexts = {"brdu": np.full((2, 1), 3.0), "edu": np.full((2, 1), 4.0)}

    for target in ("brdu", "edu"):
        np.testing.assert_array_equal(
            _assemble(embeddings, contexts, target=target, mode="joint"),
            [[1.0, 1.0, 2.0, 2.0, 3.0, 4.0]] * 2,
        )
    np.testing.assert_array_equal(
        _assemble(embeddings, contexts, target="brdu", mode="marker"), [[1.0, 1.0, 3.0]] * 2
    )
    np.testing.assert_array_equal(
        _assemble(embeddings, contexts, target="edu", mode="marker"), [[2.0, 2.0, 4.0]] * 2
    )


@pytest.mark.parametrize("mode, width, accepted", [
    ("marker", 1026, True), ("joint", 2052, True), ("marker", 2052, False), ("joint", 1026, False),
])
def test_feature_contract_widths_per_mode(mode, width, accepted) -> None:
    embedding_bundle = {"schema": "dino-knn-intensity-v1"}
    bundle = {
        **embedding_bundle,
        "classifier_feature_mode": mode,
        "context_features": ["log1p_cell_mean", "log_cell_over_local_mean"],
        "brdu_feature_scaler": SimpleNamespace(n_features_in_=width),
        "edu_feature_scaler": SimpleNamespace(n_features_in_=width),
    }
    if accepted:
        _validate_feature_contract(bundle, embedding_bundle)
    else:
        with pytest.raises(ValueError, match="Classifier feature dimensions"):
            _validate_feature_contract(bundle, embedding_bundle)


class _ConstantContext:
    def __init__(self, values: dict[str, np.ndarray]) -> None:
        self.values = values

    def features_for_labels(self, labels, *, neighbors):
        assert neighbors == 5
        return {target: values[np.asarray(labels) - 1] for target, values in self.values.items()}


def _fitted(width: int, seed: int):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(20, width))
    labels = (features[:, 0] > 0).astype(int)
    scaler = StandardScaler().fit(features)
    return scaler, LogisticRegression().fit(scaler.transform(features), labels)


@pytest.mark.parametrize("mode", ["marker", "joint"])
def test_cortex_score_rows_interleave_probability_and_call(mode) -> None:
    rows = pd.DataFrame(
        {
            "cell": ["r|2", "r|1", "r|3"],
            "dataset": ["ds"] * 3,
            "roi": ["r"] * 3,
            "label": [2, 1, 3],
            "extra": [9, 9, 9],
        }
    )
    embeddings = {
        "brdu": torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.5, 0.5]], dtype=torch.bfloat16),
        "edu": torch.tensor([[0.7, 0.3], [0.2, 0.6], [0.4, 0.1]], dtype=torch.bfloat16),
    }
    context_values = {
        "brdu": np.array([[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]]),
        "edu": np.array([[4.0, 0.4], [5.0, 0.5], [6.0, 0.6]]),
    }
    width = 4 if mode == "marker" else 8
    bundle = {"classifier_feature_mode": mode, "intensity_neighbors": 5, "brdu_cutoff": 0.5, "edu_cutoff": 0.4}
    for seed, target in enumerate(("brdu", "edu")):
        bundle[f"{target}_feature_scaler"], bundle[f"{target}_model"] = _fitted(width, seed)

    result = _score_rows(rows, embeddings=embeddings, bundle=bundle, context=_ConstantContext(context_values))

    assert list(result.columns) == ["cell", "dataset", "roi", "label", "p_brdu", "brdu", "p_edu", "edu"]
    assert result["cell"].tolist() == ["r|2", "r|1", "r|3"]
    context = {target: values[[1, 0, 2]] for target, values in context_values.items()}
    values = {target: tensor.float().numpy() for target, tensor in embeddings.items()}
    for target in ("brdu", "edu"):
        if mode == "marker":
            features = np.column_stack((values[target], context[target]))
        else:
            features = np.column_stack((values["brdu"], values["edu"], context["brdu"], context["edu"]))
        expected = bundle[f"{target}_model"].predict_proba(
            bundle[f"{target}_feature_scaler"].transform(features)
        )[:, 1].astype(np.float32)
        assert result[f"p_{target}"].dtype == np.float32
        assert result[target].dtype == np.bool_
        np.testing.assert_array_equal(result[f"p_{target}"], expected)
        np.testing.assert_array_equal(result[target], expected >= bundle[f"{target}_cutoff"])


RESCORE_CONTRACT_KEYS = {
    "artifact",
    "model",
    "model_sha256",
    "embedding_model",
    "embedding_model_sha256",
    "embedding_run",
    "cortex_props",
    "cortex_props_sha256",
    "labeled_cells",
    "labeled_cells_sha256",
    "excluded_labeled_cells",
    "source_sha256",
    "model_schema",
    "embedding_dimension",
    "classifier_embedding_dimension",
    "classifier_feature_count",
    "classifier_feature_mode",
    "embedding_pca_components",
    "context_features",
    "intensity_props",
    "intensity_props_sha256",
    "dapi_props",
    "dapi_props_sha256",
    "intensity_neighbors",
    "classifier_feature_schema",
    "brdu_cutoff",
    "edu_cutoff",
    "marker_contract",
}


def _write_rescore_inputs(root: Path) -> None:
    """Six cells in one embedding shard of two partitions; label 2 is outside cortex, roi|4 is labeled."""
    rng = np.random.default_rng(3)
    labels = np.arange(1, 7)
    pd.DataFrame(
        {
            "label": labels.astype(np.uint32),
            "centroid_z": np.zeros(6),
            "centroid_y": np.zeros(6),
            "centroid_x": labels.astype(np.float64),
            "intensity_brdu_mean": [10.0, 50.0, 90.0, 20.0, 70.0, 30.0],
            "intensity_edu_mean": [80.0, 15.0, 40.0, 60.0, 25.0, 95.0],
        }
    ).to_parquet(root / "props.parquet", index=False)
    pd.DataFrame({"label": labels, "inside_cortex": labels != 2}).to_parquet(root / "cortex.parquet", index=False)
    pd.DataFrame({"cell": ["roi|4"]}).to_csv(root / "labeled.csv", index=False)
    embedding_bundle = {"schema": "dino-knn-intensity-v1", "feature_extractor": "cell_dino_hpa_vitl16"}
    joblib.dump(embedding_bundle, root / "embedding-model.joblib")
    bundle = {
        **embedding_bundle,
        "source_sha256": "source",
        "classifier_feature_mode": "joint",
        "context_features": ["log1p_cell_mean", "log_cell_over_local_mean"],
        "intensity_neighbors": 2,
        "intensity_props": str(root / "props.parquet"),
        "intensity_props_sha256": file_sha256(root / "props.parquet"),
        "intensity_scale_zyx_um": [1.0, 1.0, 1.0],
        "brdu_cutoff": 0.5,
        "edu_cutoff": 0.5,
    }
    for seed, target in enumerate(("brdu", "edu")):
        bundle[f"{target}_feature_scaler"], bundle[f"{target}_model"] = _fitted(2052, seed)
    joblib.dump(bundle, root / "model.joblib")
    shard = root / "infer" / "shard-000"
    (shard / "predictions").mkdir(parents=True)
    (shard / "complete.json").write_text(json.dumps({
        "artifact": "dino-zarr-infer-v2", "stores_embeddings": True, "source_sha256": "source",
        "model_sha256": file_sha256(root / "embedding-model.joblib"),
        "shard_count": 1, "shard_index": 0, "cells": 6, "partition_cells": 3,
    }))
    for part, chunk in enumerate((labels[:3], labels[3:])):
        cells = [f"roi|{label}" for label in chunk]
        pd.DataFrame({"cell": cells, "dataset": "ds", "roi": "roi", "label": chunk}).to_parquet(
            shard / "predictions" / f"part-{part:06d}.parquet"
        )
        write_embedding_atomic(
            {
                target: torch.from_numpy(rng.normal(size=(3, 1024))).to(torch.bfloat16)
                for target in ("brdu", "edu")
            }
            | {"cell_sha256": ordered_cell_sha256(cells)},
            shard / "embeddings" / f"part-{part:06d}.pt",
        )


def test_cortex_rescore_complete_and_partition_contracts(tmp_path) -> None:
    _write_rescore_inputs(tmp_path)
    output = tmp_path / "out"

    complete = _score_cortex(root=tmp_path, output_dir=output)

    assert json.loads((output / "complete.json").read_text()) == complete
    assert set(complete) == RESCORE_CONTRACT_KEYS | {
        "source_partitions", "prediction_partitions", "cells", "brdu_positive", "edu_positive", "stores_embeddings",
    }
    assert {key: complete[key] for key in (
        "artifact", "excluded_labeled_cells", "source_sha256", "model_schema", "embedding_dimension",
        "classifier_embedding_dimension", "classifier_feature_count", "classifier_feature_mode",
        "embedding_pca_components", "context_features", "dapi_props", "dapi_props_sha256",
        "intensity_neighbors", "classifier_feature_schema", "brdu_cutoff", "edu_cutoff",
        "source_partitions", "prediction_partitions", "cells", "stores_embeddings", "marker_contract",
    )} == {
        "artifact": "dino-cortex-rescore-v2",
        "marker_contract": DEFAULT_JSON,
        "excluded_labeled_cells": 1,
        "source_sha256": "source",
        "model_schema": "dino-knn-intensity-v1",
        "embedding_dimension": 1024,
        "classifier_embedding_dimension": 2048,
        "classifier_feature_count": 2052,
        "classifier_feature_mode": "joint",
        "embedding_pca_components": None,
        "context_features": ["log1p_cell_mean", "log_cell_over_local_mean"],
        "dapi_props": None,
        "dapi_props_sha256": None,
        "intensity_neighbors": 2,
        "classifier_feature_schema": "dino-knn-intensity-v1",
        "brdu_cutoff": 0.5,
        "edu_cutoff": 0.5,
        "source_partitions": 2,
        "prediction_partitions": 2,
        "cells": 4,
        "stores_embeddings": False,
    }
    for key, name in (("model", "model.joblib"), ("embedding_model", "embedding-model.joblib"),
                      ("cortex_props", "cortex.parquet"), ("labeled_cells", "labeled.csv")):
        assert complete[key] == str((tmp_path / name).resolve())
        assert complete[f"{key}_sha256"] == file_sha256(tmp_path / name)
    assert complete["embedding_run"] == str((tmp_path / "infer").resolve())
    assert complete["intensity_props_sha256"] == file_sha256(tmp_path / "props.parquet")

    paths = sorted((output / "predictions").glob("*.parquet"))
    assert [path.name for path in paths] == ["shard-000-part-000000.parquet", "shard-000-part-000001.parquet"]
    frames = [pd.read_parquet(path) for path in paths]
    for path, frame, cells, source_rows in zip(paths, frames, (["roi|1", "roi|3"], ["roi|5", "roi|6"]), (3, 3)):
        provenance, rows = read_parquet_artifact(path)
        source = tmp_path / "infer" / "shard-000"
        assert list(frame.columns) == ["cell", "dataset", "roi", "label", "p_brdu", "brdu", "p_edu", "edu"]
        assert frame["cell"].tolist() == cells
        assert provenance == {
            **{key: complete[key] for key in RESCORE_CONTRACT_KEYS},
            "source_prediction": str((source / "predictions" / path.name.removeprefix("shard-000-")).resolve()),
            "source_embedding": str(
                (source / "embeddings" / path.name.removeprefix("shard-000-")).with_suffix(".pt").resolve()
            ),
            "source_rows": source_rows,
            "partition_rows": rows,
        }
    scored = pd.concat(frames, ignore_index=True)
    assert complete["brdu_positive"] == int(scored["brdu"].sum())
    assert complete["edu_positive"] == int(scored["edu"].sum())


def test_intensity_context_maps_brdu_and_edu_mean_columns(tmp_path) -> None:
    path = tmp_path / "props.parquet"
    pd.DataFrame(
        {
            "label": [1, 2, 3],
            "centroid_z": [0.0, 0.0, 0.0],
            "centroid_y": [0.0, 0.0, 0.0],
            "centroid_x": [0.0, 1.0, 3.0],
            "intensity_brdu_mean": [10.0, 20.0, 40.0],
            "intensity_edu_mean": [1.0, 3.0, 7.0],
        }
    ).to_parquet(path, index=False)

    context, features = _intensity_features(path, np.array([1, 3]), neighbors=1)

    np.testing.assert_array_equal(context.means["brdu"], [10.0, 20.0, 40.0])
    np.testing.assert_array_equal(context.means["edu"], [1.0, 3.0, 7.0])
    assert list(features) == ["brdu", "edu"]
    np.testing.assert_allclose(features["brdu"], [[np.log1p(10), np.log(11 / 21)], [np.log1p(40), np.log(41 / 21)]])
    np.testing.assert_allclose(features["edu"], [[np.log1p(1), np.log(2 / 4)], [np.log1p(7), np.log(8 / 4)]])


def test_ordered_cell_digest_is_length_framed_sha256() -> None:
    expected = hashlib.sha256()
    for cell in ("roi|1", "roi|2"):
        expected.update(len(cell).to_bytes(8, "big") + cell.encode())

    digest = bytes(ordered_cell_sha256(["roi|1", "roi|2"]).tolist()).hex()

    assert digest == expected.hexdigest() == "fa30a932429282ef046fbe6f81b493435022fc3b45446557e886c775e0101e71"
