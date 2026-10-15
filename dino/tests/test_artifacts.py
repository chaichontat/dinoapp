from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from dino.artifacts import (
    array_sha256,
    file_sha256,
    path_sha256,
    read_embedding_artifact,
    read_parquet_artifact,
    row_feature_cache_key,
    validate_model_artifacts,
    write_csv_atomic,
    write_embedding_atomic,
    write_parquet_atomic,
)


@pytest.mark.integration
def test_atomic_embedding_preserves_native_bfloat16(tmp_path: Path) -> None:
    output = tmp_path / "part.pt"
    embeddings = {
        "brdu": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
        "edu": torch.ones((3, 4), dtype=torch.bfloat16),
        "cell_sha256": torch.arange(32, dtype=torch.uint8),
    }

    write_embedding_atomic(embeddings, output)
    actual = read_embedding_artifact(output)

    assert torch.equal(actual["brdu"], embeddings["brdu"])
    assert torch.equal(actual["edu"], embeddings["edu"])
    assert actual["brdu"].dtype == torch.bfloat16
    assert set(tmp_path.iterdir()) == {output}


def test_atomic_embedding_refuses_existing_destination(tmp_path: Path) -> None:
    output = tmp_path / "part.pt"
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        write_embedding_atomic(
            {
                "brdu": torch.ones((1, 2), dtype=torch.bfloat16),
                "edu": torch.ones((1, 2), dtype=torch.bfloat16),
                "cell_sha256": torch.zeros(32, dtype=torch.uint8),
            },
            output,
        )


def test_model_artifacts_reject_changed_weights(tmp_path: Path) -> None:
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"trained weights")
    repo = tmp_path / "dinov2"
    repo.mkdir()
    (repo / "hubconf.py").write_text("MODEL = 1\n", encoding="utf-8")
    bundle = {
        "weights_sha256": file_sha256(weights),
        "model_source_sha256": path_sha256(repo),
    }

    validate_model_artifacts(bundle=bundle, weights=weights, model_source=repo)
    weights.write_bytes(b"different weights")

    with pytest.raises(ValueError, match="checkpoint content does not match"):
        validate_model_artifacts(bundle=bundle, weights=weights, model_source=repo)


def test_model_artifacts_reject_changed_source(tmp_path: Path) -> None:
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"trained weights")
    repo = tmp_path / "dinov2"
    repo.mkdir()
    source = repo / "hubconf.py"
    source.write_text("MODEL = 1\n", encoding="utf-8")
    bundle = {
        "weights_sha256": file_sha256(weights),
        "model_source_sha256": path_sha256(repo),
    }
    source.write_text("MODEL = 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source content does not match"):
        validate_model_artifacts(bundle=bundle, weights=weights, model_source=repo)


def test_array_sha256_changes_with_masked_input() -> None:
    image = np.zeros((4, 4, 2), dtype=np.uint8)
    before = array_sha256(image)
    image[1, 2, 0] = 1

    assert array_sha256(image) != before


@pytest.mark.integration
def test_atomic_parquet_records_completion_metadata(tmp_path: Path) -> None:
    output = tmp_path / "scores.parquet"
    frame = pd.DataFrame(
        {"dataset": ["ds1", "ds1"], "brdu_prob": [0.1, 0.9]},
        index=pd.Index(["r1|1", "r1|2"], name="cell"),
    )
    provenance = {"version": 1, "dataset": "ds1", "model_sha256": "abc"}

    write_parquet_atomic(frame, output, provenance=provenance)
    actual_provenance, row_count = read_parquet_artifact(output)

    assert actual_provenance == provenance
    assert row_count == len(frame)
    assert set(tmp_path.iterdir()) == {output}


def test_atomic_parquet_refuses_existing_destination(tmp_path: Path) -> None:
    output = tmp_path / "scores.parquet"
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        write_parquet_atomic(pd.DataFrame({"value": [1]}), output, provenance={"version": 1})

    assert output.read_bytes() == b"existing"


def test_invalid_parquet_reports_incomplete_artifact(tmp_path: Path) -> None:
    output = tmp_path / "scores.parquet"
    output.write_bytes(b"partial")

    with pytest.raises(ValueError, match="Unable to read Cell-DINO parquet artifact"):
        read_parquet_artifact(output)


def test_atomic_joblib_replaces_existing_artifact(tmp_path: Path) -> None:
    import joblib

    from dino.artifacts import _dump_joblib_atomic

    output = tmp_path / "model.joblib"
    joblib.dump({"previous": True}, output)
    _dump_joblib_atomic({"features": np.arange(6).reshape(2, 3)}, output)

    np.testing.assert_array_equal(joblib.load(output)["features"], np.arange(6).reshape(2, 3))
    assert set(tmp_path.iterdir()) == {output}


@pytest.mark.parametrize(
    ("namespace", "expected"),
    [
        ({}, "4afc1ddcb5333f1fffd84765a445be6a1e4a39d4"),
        ({"dataset": "dsβ"}, "602d8f5224564fca48c5edc01122c40866ffc3dd"),
        ({"roi": "r2"}, "42c09b83b468dcad8184686527f2ca4d396ae777"),
        ({"cell": "r1|18"}, "76b47d70c76e9d65a2723c79e02c7ebfae9acfe3"),
    ],
)
def test_row_feature_cache_key_preserves_persisted_identity(namespace, expected) -> None:
    row = pd.Series(
        {"dataset": "dsα", "roi": "r1", "cell": "r1|17", "x": 0.5, "y": 1.5, "z": -2.5}
        | namespace
    )

    assert row_feature_cache_key(row) == expected


def test_atomic_csv_creates_parents_and_replaces_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "index.csv"
    frame = pd.DataFrame({"cell": ["r|2", "r|1"], "note": ["a,b", "α"]}, index=[5, 3])

    write_csv_atomic(frame.iloc[:1], output)
    assert output.read_bytes() == b'cell,note\nr|2,"a,b"\n'
    write_csv_atomic(frame, output)

    assert output.read_bytes() == 'cell,note\nr|2,"a,b"\nr|1,α\n'.encode("utf-8")
    assert set(output.parent.iterdir()) == {output}


@pytest.mark.parametrize("failure", ["serialization", "replace"])
def test_atomic_csv_failure_preserves_output_and_cleans_temporary(monkeypatch, tmp_path, failure) -> None:
    from dino import artifacts

    output = tmp_path / "index.csv"
    output.write_bytes(b"existing\n")
    if failure == "serialization":
        def fail_serialization(self, path, *, index):
            Path(path).write_bytes(b"partial\n")
            raise OSError("serialization failed")

        monkeypatch.setattr(pd.DataFrame, "to_csv", fail_serialization)
    else:
        def fail_replace(source, destination):
            raise OSError("replace failed")

        monkeypatch.setattr(artifacts.os, "replace", fail_replace)

    with pytest.raises(OSError, match=f"{failure} failed"):
        write_csv_atomic(pd.DataFrame({"cell": ["r|1"]}), output)

    assert output.read_bytes() == b"existing\n"
    assert set(tmp_path.iterdir()) == {output}
