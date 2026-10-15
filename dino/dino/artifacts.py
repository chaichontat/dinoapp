from __future__ import annotations

import hashlib
import itertools
import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dino_markers import MarkerContract

from dino.cell_dino import require_dinov2_source

if TYPE_CHECKING:
    import torch


PARQUET_METADATA_KEY = b"fishtools.dino"


def _dump_joblib_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def row_feature_cache_key(row: pd.Series) -> str:
    payload = "|".join(
        [
            str(row["dataset"]),
            str(row["roi"]),
            str(row["cell"]),
            str(int(np.rint(float(row["x"])))),
            str(int(np.rint(float(row["y"])))),
            str(int(np.rint(float(row["z"])))),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Expected file for SHA-256 digest: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def validate_model_artifacts(
    *,
    bundle: Mapping[str, Any],
    weights: Path,
) -> None:
    """Check the checkpoint, and the installed dinov2 when the bundle records it.

    Legacy bundles record a hash of a local checkout (model_source_sha256) instead of
    dinov2_source; their source is not checked.
    """
    expected_weights = bundle.get("weights_sha256")
    if not isinstance(expected_weights, str):
        raise ValueError("Model bundle is missing weights_sha256; retrain the classifier bundle.")
    if "dinov2_source" not in bundle and not isinstance(bundle.get("model_source_sha256"), str):
        raise ValueError("Model bundle records no dinov2_source; retrain the classifier bundle.")
    if file_sha256(weights) != expected_weights:
        raise ValueError(f"Cell-DINO checkpoint content does not match the trained bundle: {weights}")
    if "dinov2_source" in bundle:
        require_dinov2_source(str(bundle["dinov2_source"]))


def write_parquet_atomic(frame: pd.DataFrame, path: Path, *, provenance: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing Cell-DINO artifact: {path}")
    table = pa.Table.from_pandas(frame, preserve_index=True)
    metadata = dict(table.schema.metadata or {})
    metadata[PARQUET_METADATA_KEY] = json.dumps(dict(provenance), sort_keys=True).encode("utf-8")
    table = table.replace_schema_metadata(metadata)

    with tempfile.NamedTemporaryFile(prefix=".dino-", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        pq.write_table(table, temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_parquet_artifact(path: Path) -> tuple[dict[str, Any], int]:
    try:
        parquet = pq.ParquetFile(path)
        metadata = parquet.schema_arrow.metadata or {}
        encoded = metadata.get(PARQUET_METADATA_KEY)
        if encoded is None:
            raise ValueError(f"Parquet is missing Cell-DINO completion metadata: {path}")
        provenance = json.loads(encoded.decode("utf-8"))
        if not isinstance(provenance, dict):
            raise ValueError(f"Invalid Cell-DINO completion metadata in {path}")
        return provenance, parquet.metadata.num_rows
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unable to read Cell-DINO parquet artifact {path}: {exc}") from exc


def ordered_cell_sha256(cells: Iterable[str]) -> torch.Tensor:
    """Hash an ordered cell sequence with length framing to prevent ambiguous joins."""

    import torch

    digest = hashlib.sha256()
    for cell in cells:
        encoded = cell.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return torch.tensor(list(digest.digest()), dtype=torch.uint8)


def _validate_embedding_artifact(artifact: Mapping[str, torch.Tensor]) -> None:
    import torch

    if "cell_sha256" not in artifact:
        raise ValueError("Embedding artifact must contain cell_sha256")
    cell_sha256 = artifact["cell_sha256"]
    if cell_sha256.dtype != torch.uint8 or cell_sha256.shape != (32,):
        raise ValueError("cell_sha256 must be a 32-byte uint8 tensor")
    keys = tuple(key for key in artifact if key != "cell_sha256")
    if not keys:
        raise ValueError("Embedding artifact must contain at least one marker tensor")
    # The contract's key rules apply to marker keys; any reference outside them completes a contract.
    reference = next(name for name in map("ref{}".format, itertools.count()) if name not in keys)
    MarkerContract(reference, keys)
    shape = artifact[keys[0]].shape
    for key in keys:
        tensor = artifact[key]
        if tensor.device.type != "cpu":
            raise ValueError(f"Embedding tensor {key!r} must be stored on CPU")
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"Embedding tensor {key!r} must use native torch.bfloat16")
        if tensor.ndim != 2 or tensor.shape != shape:
            raise ValueError(
                f"Embedding tensors must share one two-dimensional shape; {key!r} is {tuple(tensor.shape)}"
                f", expected {tuple(shape)}"
            )


def require_embedding_markers(
    artifact: Mapping[str, torch.Tensor], markers: Sequence[str], *, path: Path
) -> None:
    found = set(artifact) - {"cell_sha256"}
    if found != set(markers):
        raise ValueError(
            f"Embedding artifact {path} holds markers {sorted(found)}, expected {sorted(markers)}"
        )


def write_embedding_atomic(artifact: Mapping[str, torch.Tensor], path: Path) -> None:
    """Publish a validated, native-BF16 embedding partition without overwriting data."""

    import torch

    _validate_embedding_artifact(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing Cell-DINO artifact: {path}")
    with tempfile.NamedTemporaryFile(prefix=".dino-", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(artifact), temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_embedding_artifact(path: Path) -> dict[str, torch.Tensor]:
    import torch

    try:
        artifact = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(artifact, dict):
            raise ValueError("Embedding artifact must contain a tensor dictionary")
        _validate_embedding_artifact(artifact)
        return artifact
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Unable to read Cell-DINO embedding artifact {path}: {exc}") from exc
