from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from dino_markers import INFERENCE, cutoff_key, probability_column, require_same_contract


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_calls(
    inference: Path,
    expected_cells: list[np.ndarray],
    source_hash: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    """Require prediction identities to match reconstructed embedding rows exactly."""
    if not expected_cells:
        raise ValueError(f"No inference shards requested: {inference}")
    parts: dict[str, list[np.ndarray]] = {}
    records = []
    model_hash = None
    contract = None
    contract_from = None
    for shard, cells in enumerate(expected_cells):
        print(f"Validating shard {shard}: {len(cells):,} cell identities", flush=True)
        directory = inference / f"shard-{shard:03d}"
        complete_path = directory / "complete.json"
        complete = json.loads(complete_path.read_text())
        shard_contract = INFERENCE.contract(complete, where=str(complete_path))
        if contract is None:
            contract, contract_from = shard_contract, str(complete_path)
            parts = {marker: [] for marker in contract.markers}
        require_same_contract(contract, shard_contract, expected_from=contract_from, actual_from=str(complete_path))
        if (
            complete["source_sha256"] != source_hash
            or complete["cells"] != len(cells)
            or complete["shard_index"] != shard
            or complete["shard_count"] != len(expected_cells)
        ):
            raise ValueError(f"Inference source/shard contract differs: {directory}")
        if model_hash is not None and complete["model_sha256"] != model_hash:
            raise ValueError("Inference shards use different models")
        model_hash = complete["model_sha256"]
        partition_size = complete["partition_cells"]
        count = (len(cells) + partition_size - 1) // partition_size
        paths = [directory / "predictions" / f"part-{i:06d}.parquet" for i in range(count)]
        if set(paths) != set((directory / "predictions").glob("*.parquet")):
            raise ValueError(f"Prediction partitions missing or unexpected: {directory}")
        for index, path in enumerate(paths):
            table = pq.read_table(
                path, columns=["cell", *contract.markers, *(probability_column(m) for m in contract.markers)]
            )
            metadata = json.loads(table.schema.metadata[b"fishtools.dino"])
            expected = cells[index * partition_size : (index + 1) * partition_size]
            if any(metadata.get(key) != value for key, value in complete.items()):
                raise ValueError(f"Partition provenance differs from shard: {path}")
            INFERENCE.contract(metadata, where=str(path))
            if metadata["partition_index"] != index or metadata["partition_rows"] != len(expected):
                raise ValueError(f"Partition index/length differs: {path}")
            if not np.array_equal(table["cell"].to_numpy(), expected):
                raise ValueError(f"Prediction cell order differs from UMAP reconstruction: {path}")
            for target in contract.markers:
                if table[target].null_count or table[probability_column(target)].null_count:
                    raise ValueError(f"Missing {target} predictions: {path}")
                calls = table[target].to_numpy()
                probabilities = table[probability_column(target)].to_numpy()
                if not np.isfinite(probabilities).all() or not np.all(
                    (probabilities >= 0) & (probabilities <= 1)
                ):
                    raise ValueError(f"Invalid {target} probabilities: {path}")
                if not np.array_equal(calls, probabilities >= complete[cutoff_key(target)]):
                    raise ValueError(f"{target} calls disagree with recorded cutoff: {path}")
                parts[target].append(calls)
            records.append({"path": str(path), "sha256": sha256(path), "metadata": metadata})
    return {marker: np.concatenate(parts[marker]) for marker in contract.markers}, records
