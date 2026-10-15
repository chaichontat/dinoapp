from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from dino import cell_dino
from dino.artifacts import (
    array_sha256,
    file_sha256,
    read_embedding_artifact,
    read_parquet_artifact,
    require_embedding_markers,
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


def _embeddings(*keys: str, rows: int = 3) -> dict[str, torch.Tensor]:
    return {key: torch.ones((rows, 4), dtype=torch.bfloat16) for key in keys} | {
        "cell_sha256": torch.zeros(32, dtype=torch.uint8)
    }


def test_embedding_artifact_round_trips_any_valid_marker_set(tmp_path: Path) -> None:
    for name, keys in (("three", ("edu", "brdu", "pcna")), ("pi", ("pi", "edu")), ("one", ("brdu",))):
        embeddings = _embeddings(*keys)
        write_embedding_atomic(embeddings, tmp_path / f"{name}.pt")
        actual = read_embedding_artifact(tmp_path / f"{name}.pt")
        assert list(actual) == list(embeddings)
        assert all(torch.equal(actual[key], embeddings[key]) for key in embeddings)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("no_markers", "at least one marker"),
        ("no_digest", "must contain cell_sha256"),
        ("shape", "share one two-dimensional shape"),
        ("dtype", "native torch.bfloat16"),
        ("digest", "32-byte uint8"),
        ("uppercase", "must match"),
    ],
)
def test_embedding_artifact_rejects_structural_damage(tmp_path: Path, damage: str, message: str) -> None:
    artifact = _embeddings("edu", "brdu", "pcna")
    if damage == "no_markers":
        artifact = {"cell_sha256": artifact["cell_sha256"]}
    elif damage == "no_digest":
        del artifact["cell_sha256"]
    elif damage == "shape":
        artifact["pcna"] = artifact["pcna"][:2]
    elif damage == "dtype":
        artifact["pcna"] = artifact["pcna"].float()
    elif damage == "digest":
        artifact["cell_sha256"] = torch.zeros(31, dtype=torch.uint8)
    else:
        artifact["PCNA"] = artifact.pop("pcna")

    with pytest.raises(ValueError, match=message):
        write_embedding_atomic(artifact, tmp_path / "part.pt")
    assert not (tmp_path / "part.pt").exists()


def test_reading_an_invalid_embedding_artifact_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "part.pt"
    torch.save({**_embeddings("edu", "brdu"), "BrdU": torch.ones((3, 4), dtype=torch.bfloat16)}, path)

    with pytest.raises(ValueError, match="Unable to read Cell-DINO embedding artifact .*part.pt"):
        read_embedding_artifact(path)


def test_require_embedding_markers_accepts_exactly_the_contract_markers(tmp_path: Path) -> None:
    require_embedding_markers(_embeddings("edu", "brdu", "pcna"), ("pcna", "edu", "brdu"), path=tmp_path)


@pytest.mark.parametrize("keys", [("edu", "brdu"), ("edu", "brdu", "pcna", "extra")])
def test_require_embedding_markers_rejects_missing_and_extra_markers(tmp_path: Path, keys) -> None:
    path = tmp_path / "part.pt"
    with pytest.raises(ValueError, match=r"part\.pt.*holds markers .*expected \['brdu', 'edu', 'pcna'\]"):
        require_embedding_markers(_embeddings(*keys), ("edu", "brdu", "pcna"), path=path)


def test_model_artifacts_reject_changed_weights(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cell_dino, "installed_dinov2_source", lambda: "git+https://example/dinov2@a")
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"trained weights")
    bundle = {"weights_sha256": file_sha256(weights), "dinov2_source": "git+https://example/dinov2@a"}

    validate_model_artifacts(bundle=bundle, weights=weights)
    weights.write_bytes(b"different weights")

    with pytest.raises(ValueError, match="checkpoint content does not match"):
        validate_model_artifacts(bundle=bundle, weights=weights)


def test_model_artifacts_reject_another_installed_dinov2(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cell_dino, "installed_dinov2_source", lambda: "git+https://example/dinov2@b")
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"trained weights")
    bundle = {"weights_sha256": file_sha256(weights), "dinov2_source": "git+https://example/dinov2@a"}

    with pytest.raises(ValueError, match=r"dinov2@b.*dinov2@a"):
        validate_model_artifacts(bundle=bundle, weights=weights)


def test_legacy_model_artifacts_skip_the_source_check(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        cell_dino, "installed_dinov2_source", lambda: pytest.fail("a legacy bundle must not check dinov2")
    )
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"trained weights")
    bundle = {"weights_sha256": file_sha256(weights), "model_source_sha256": "legacy checkout digest"}

    validate_model_artifacts(bundle=bundle, weights=weights)
    weights.write_bytes(b"different weights")

    with pytest.raises(ValueError, match="checkpoint content does not match"):
        validate_model_artifacts(bundle=bundle, weights=weights)


def test_model_artifacts_require_a_source_identity(tmp_path: Path) -> None:
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"trained weights")

    with pytest.raises(ValueError, match="dinov2_source"):
        validate_model_artifacts(bundle={"weights_sha256": file_sha256(weights)}, weights=weights)


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
