"""Rescore persisted Cell-DINO embeddings for every cortical cell, one process per partition.

This reproduces the prediction contract of `full-run2/score-cortex.py` (per-partition prediction
parquets with identical provenance metadata and `complete.json`) with the same numerical path,
distributing independent partitions over worker processes. Embeddings are read from the source
run and not copied into the output.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from dino_markers import (
    CORTEX_RESCORE,
    INFERENCE,
    MarkerContract,
    cutoff_key,
    model_key,
    positive_count_key,
    probability_column,
    require_same_contract,
    scaler_key,
)

from dino.artifacts import (
    file_sha256,
    ordered_cell_sha256,
    read_embedding_artifact,
    read_parquet_artifact,
    require_embedding_markers,
    write_parquet_atomic,
)
from dino.classifier_features import (
    EMBEDDING_WIDTH,
    validate_feature_contract,
    Context,
    assemble_classifier_features,
    load_classifier_context,
)
from dino.zarr_source import write_json_atomic

def load_cortex_lookup(path: Path) -> np.ndarray:
    table = pq.read_table(path, columns=["label", "inside_cortex"])
    if table["label"].null_count or table["inside_cortex"].null_count:
        raise ValueError(f"Cortex selection contains null values: {path}")
    labels = table["label"].combine_chunks().to_numpy(zero_copy_only=False)
    if not np.array_equal(labels, np.arange(1, table.num_rows + 1, dtype=labels.dtype)):
        raise ValueError("Cortex labels must be ordered and contiguous from one")
    lookup = np.zeros(table.num_rows + 1, dtype=np.bool_)
    lookup[1:] = table["inside_cortex"].combine_chunks().to_numpy(zero_copy_only=False)
    return lookup


def score_rows(
    rows: pd.DataFrame,
    *,
    embeddings: dict[str, torch.Tensor],
    bundle: dict[str, object],
    context: Context,
    markers: Sequence[str],
) -> pd.DataFrame:
    """Apply the saved scaler and logistic model to embeddings plus kNN intensity context."""
    labels = rows["label"].to_numpy(dtype=np.int64)
    context_features = context.features_for_labels(labels, neighbors=int(bundle["intensity_neighbors"]))
    values = {target: embeddings[target].float().numpy() for target in markers}
    mode = str(bundle.get("classifier_feature_mode", "marker"))
    result = rows.loc[:, ["cell", "dataset", "roi", "label"]].reset_index(drop=True)
    for target in markers:
        features = assemble_classifier_features(
            values, context_features, target=target, mode=mode, markers=markers
        )
        scaler = bundle[scaler_key(target)]
        probability = bundle[model_key(target)].predict_proba(scaler.transform(features))[:, 1].astype(np.float32)
        result[probability_column(target)] = probability
        result[target] = probability >= float(bundle[cutoff_key(target)])
    return result


class _Worker:
    bundle: dict[str, object]
    context: Context
    cortex: np.ndarray
    labeled: set[str]
    run_record: dict[str, object]
    markers: tuple[str, ...]
    output: Path


def _init_worker(
    model: Path,
    cortex: np.ndarray,
    labeled: set[str],
    run_record: dict[str, object],
    markers: tuple[str, ...],
    output: Path,
) -> None:
    _Worker.bundle = joblib.load(model)
    _Worker.context = load_classifier_context(_Worker.bundle)
    if len(_Worker.context.points_zyx_um) + 1 != len(cortex):
        raise ValueError("Cortex selection and intensity context cell counts differ")
    _Worker.cortex, _Worker.labeled, _Worker.run_record = cortex, labeled, run_record
    _Worker.markers, _Worker.output = markers, output


def _score_partition(
    shard_name: str, prediction_path: Path, embedding_path: Path
) -> tuple[int, dict[str, int]] | None:
    """Score and publish one source partition; return its cortical row count and positives per marker."""
    markers = _Worker.markers
    source = pq.read_table(prediction_path, columns=["cell", "dataset", "roi", "label"]).to_pandas()
    artifact = read_embedding_artifact(embedding_path)
    require_embedding_markers(artifact, markers, path=embedding_path)
    cells = source["cell"].astype(str)
    if len(source) != artifact[markers[0]].shape[0] or not torch.equal(
        artifact["cell_sha256"], ordered_cell_sha256(cells.tolist())
    ):
        raise ValueError(f"Prediction and embedding identities differ: {prediction_path}")
    labels = source["label"].to_numpy(dtype=np.int64)
    if labels.min() < 1 or labels.max() >= len(_Worker.cortex):
        raise ValueError(f"Source labels lie outside cortex lookup: {prediction_path}")
    keep = _Worker.cortex[labels] & ~cells.isin(_Worker.labeled).to_numpy()
    selected = source.loc[keep].reset_index(drop=True)
    if selected.empty:
        return None
    destination = _Worker.output / "predictions" / f"{shard_name}-{prediction_path.name}"
    part_contract = {
        **_Worker.run_record,
        "source_prediction": str(prediction_path.resolve()),
        "source_embedding": str(embedding_path.resolve()),
        "source_rows": len(source),
        "partition_rows": len(selected),
    }
    if destination.is_file():
        provenance, rows = read_parquet_artifact(destination)
        if provenance != part_contract or rows != len(selected):
            raise ValueError(f"Existing output has a different contract: {destination}")
        scored = pq.read_table(destination, columns=list(markers)).to_pandas()
    else:
        scored = score_rows(
            selected,
            embeddings={target: artifact[target][torch.from_numpy(keep)] for target in markers},
            bundle=_Worker.bundle,
            context=_Worker.context,
            markers=markers,
        )
        write_parquet_atomic(scored, destination, provenance=part_contract)
    return len(selected), {target: int(scored[target].sum()) for target in markers}


def _partitions(
    embedding_run: Path, *, source_sha256: str, model_sha256: str, contract: MarkerContract
) -> list[tuple[str, Path, Path]]:
    tasks = []
    shard_count = None
    shard_indices = set()
    for shard_dir in sorted(embedding_run.glob("shard-*")):
        complete_path = shard_dir / "complete.json"
        manifest = json.loads(complete_path.read_text())
        if not manifest.get("stores_embeddings"):
            raise ValueError(f"Embedding run did not persist embeddings: {shard_dir}")
        if manifest.get("source_sha256") != source_sha256:
            raise ValueError(f"Embedding source differs from the fitted model: {shard_dir}")
        if manifest.get("model_sha256") != model_sha256:
            raise ValueError(f"Embedding model identity differs for {shard_dir}")
        require_same_contract(
            contract,
            INFERENCE.contract(manifest, where=str(complete_path)),
            expected_from="embedding model bundle",
            actual_from=str(complete_path),
        )
        for key in ("shard_count", "shard_index", "cells", "partition_cells"):
            if type(manifest.get(key)) is not int:
                raise ValueError(f"Invalid embedding completion {key}: {shard_dir}")
        count, index = manifest["shard_count"], manifest["shard_index"]
        cells, partition_cells = manifest["cells"], manifest["partition_cells"]
        if (count < 1 or not 0 <= index < count or cells < 0 or partition_cells < 1
                or index in shard_indices or (shard_count is not None and count != shard_count)):
            raise ValueError(f"Invalid embedding shard roster: {shard_dir}")
        shard_count = count
        shard_indices.add(index)
        expected = {f"part-{part:06d}.parquet" for part in range((cells + partition_cells - 1) // partition_cells)}
        paths = sorted((shard_dir / "predictions").glob("*.parquet"))
        if {path.name for path in paths} != expected:
            raise ValueError(f"Incomplete embedding prediction partitions: {shard_dir}")
        for part, path in enumerate(paths):
            expected_rows = min(partition_cells, cells - part * partition_cells)
            if pq.ParquetFile(path).metadata.num_rows != expected_rows:
                raise ValueError(f"Embedding prediction partition row count differs: {path}")
            embedding_path = shard_dir / "embeddings" / path.with_suffix(".pt").name
            if not embedding_path.is_file():
                raise ValueError(f"Missing embedding partition: {embedding_path}")
            tasks.append((shard_dir.name, path, embedding_path))
    if shard_count is not None and shard_indices != set(range(shard_count)):
        raise ValueError(f"Incomplete embedding shard roster: {embedding_run}")
    return tasks


def score_cortex(
    *,
    model: Path,
    embedding_model: Path,
    embedding_run: Path,
    cortex_props: Path,
    labeled_cells: Path,
    output_dir: Path,
    workers: int,
) -> dict[str, object]:
    if workers < 1:
        raise ValueError("workers must be positive")
    bundle = joblib.load(model)
    contract = validate_feature_contract(bundle, joblib.load(embedding_model))
    markers = contract.markers
    embedding_model_sha256 = file_sha256(embedding_model)
    cortex = load_cortex_lookup(cortex_props)
    labeled = set(pd.read_csv(labeled_cells, usecols=["cell"])["cell"].astype(str))
    mode = str(bundle.get("classifier_feature_mode", "marker"))
    fields = {
        "model": str(model.resolve()),
        "model_sha256": file_sha256(model),
        "embedding_model": str(embedding_model.resolve()),
        "embedding_model_sha256": embedding_model_sha256,
        "embedding_run": str(embedding_run.resolve()),
        "cortex_props": str(cortex_props.resolve()),
        "cortex_props_sha256": file_sha256(cortex_props),
        "labeled_cells": str(labeled_cells.resolve()),
        "labeled_cells_sha256": file_sha256(labeled_cells),
        "excluded_labeled_cells": len(labeled),
        "source_sha256": str(bundle["source_sha256"]),
        "model_schema": str(bundle["schema"]),
        "embedding_dimension": EMBEDDING_WIDTH,
        "classifier_embedding_dimension": EMBEDDING_WIDTH if mode == "marker" else EMBEDDING_WIDTH * len(markers),
        "classifier_feature_count": int(bundle[scaler_key(markers[0])].n_features_in_),
        "classifier_feature_mode": mode,
        "embedding_pca_components": None,
        "context_features": list(bundle["context_features"]),
        "intensity_props": bundle["intensity_props"],
        "intensity_props_sha256": bundle["intensity_props_sha256"],
        "dapi_props": bundle.get("dapi_props"),
        "dapi_props_sha256": bundle.get("dapi_props_sha256"),
        "intensity_neighbors": int(bundle["intensity_neighbors"]),
        "classifier_feature_schema": str(bundle["schema"]),
        **{cutoff_key(marker): float(bundle[cutoff_key(marker)]) for marker in markers},
    }
    expected = CORTEX_RESCORE.stamp(fields, contract)
    tasks = _partitions(
        embedding_run, source_sha256=str(bundle["source_sha256"]), model_sha256=embedding_model_sha256,
        contract=contract,
    )
    if not tasks:
        raise ValueError(f"No embedding partitions found under {embedding_run}")
    complete_path = output_dir / "complete.json"
    prediction_dir = output_dir / "predictions"
    if complete_path.exists():
        recorded = json.loads(complete_path.read_text())
        recorded_from = complete_path
    else:
        recorded_from = next(iter(sorted(prediction_dir.glob("*.parquet"))), None)
        recorded = read_parquet_artifact(recorded_from)[0] if recorded_from is not None else None
    if recorded is None:
        run_record = expected
    else:
        normalized = CORTEX_RESCORE.normalize(recorded, where=str(recorded_from))
        if any(key not in normalized or normalized[key] != value for key, value in expected.items()):
            raise ValueError(f"Existing completion has a different contract: {recorded_from}")
        run_record = {key: value for key, value in recorded.items() if key in expected}
    source_by_output = {
        f"{shard_name}-{prediction_path.name}": (prediction_path, embedding_path)
        for shard_name, prediction_path, embedding_path in tasks
    }
    for path in prediction_dir.glob("*.parquet"):
        if path.name not in source_by_output:
            raise ValueError(f"Unexpected existing output partition: {path}")
        prediction_path, embedding_path = source_by_output[path.name]
        provenance, rows = read_parquet_artifact(path)
        expected_partition = {
            **run_record,
            "source_prediction": str(prediction_path.resolve()),
            "source_embedding": str(embedding_path.resolve()),
            "source_rows": pq.ParquetFile(prediction_path).metadata.num_rows,
            "partition_rows": rows,
        }
        if provenance != expected_partition or rows < 1:
            raise ValueError(f"Existing output has a different contract: {path}")
    prediction_dir.mkdir(parents=True, exist_ok=True)
    total, output_parts = 0, 0
    expected_outputs = set()
    positive = {target: 0 for target in markers}
    # Spawned workers each load the bundle and kNN context; forking after pyarrow has started
    # its thread pools is unsafe.
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(model, cortex, labeled, run_record, markers, output_dir),
    ) as executor:
        results = executor.map(_score_partition, *zip(*tasks))
        for source_parts, (task, result) in enumerate(zip(tasks, results), start=1):
            if result is not None:
                expected_outputs.add(f"{task[0]}-{task[1].name}")
                rows, partition_positive = result
                output_parts += 1
                total += rows
                for target in markers:
                    positive[target] += partition_positive[target]
            if source_parts % 20 == 0:
                print(f"parts={source_parts} cortical_rows={total}", flush=True)
    if {path.name for path in prediction_dir.glob("*.parquet")} != expected_outputs:
        raise ValueError(f"Output partition roster differs from selected source partitions: {prediction_dir}")
    complete = {
        **run_record,
        "source_partitions": len(tasks),
        "prediction_partitions": output_parts,
        "cells": total,
        **{positive_count_key(target): positive[target] for target in markers},
        "stores_embeddings": False,
    }
    write_json_atomic(complete, output_dir / "complete.json")
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--embedding-run", type=Path, required=True)
    parser.add_argument("--cortex-props", type=Path, required=True)
    parser.add_argument("--labeled-cells", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    complete = score_cortex(
        model=args.model,
        embedding_model=args.embedding_model,
        embedding_run=args.embedding_run,
        cortex_props=args.cortex_props,
        labeled_cells=args.labeled_cells,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    print(json.dumps(complete, indent=2), flush=True)
    print("status=complete", flush=True)


if __name__ == "__main__":
    main()
