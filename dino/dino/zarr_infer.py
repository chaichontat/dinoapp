from __future__ import annotations

import argparse
import heapq
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from dino_markers import (
    INFERENCE,
    INTENSITY_MODEL,
    ZARR_MODEL,
    MarkerContract,
    bundle_contract,
    contract_of,
    cutoff_key,
    load_config,
    model_key,
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
    validate_model_artifacts,
    write_embedding_atomic,
    write_parquet_atomic,
)
from dino.intensity_context import IntensityContext
from dino.zarr_model import (
    FEATURE_EXECUTION,
    MODEL_INPUT_CHANNELS,
    build_cell_transform,
    extract_normalized_feature_tensor,
    feature_dtype,
    iter_model_image_batches,
    iter_prefetched_batches,
    load_cell_dino_model,
    model_crop_reader,
    prepare_model_batch,
    sort_cells_for_io,
)
from dino.zarr_source import CellCropReader, load_source, write_json_atomic


@dataclass(frozen=True)
class InferConfig:
    source_dir: Path
    model_dir: Path
    out_dir: Path
    markers: MarkerContract | None
    cells: Path | None
    dinov2_repo: Path | None
    weights: Path | None
    batch_size: int
    cache_gb: float
    prefetch_workers: int
    crop_workers: int
    partition_cells: int
    shard_index: int
    shard_count: int
    resume: bool
    store_embeddings: bool


def _load_bundle(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    kinds = (ZARR_MODEL, INTENSITY_MODEL)
    if not isinstance(bundle, dict) or not any(kind.recognizes(bundle) for kind in kinds):
        schemas = sorted(version for kind in kinds for version in (kind.legacy, kind.current))
        raise ValueError(f"Expected one of {schemas}: {path}")
    contract = bundle_contract(bundle, where=str(path))
    required = {
        "source_sha256",
        "feature_extractor",
        "input_channels",
        "weights",
        "weights_sha256",
        "dinov2_repo",
        "model_source_sha256",
        "crop_size",
        "resize_size",
        "crop_eval_size",
        "highpass_sigma_px",
        "mask_erosion_px",
        "input_channel_names_by_target",
        "feature_precision",
        "feature_execution",
        *(
            name
            for marker in contract.markers
            for name in (scaler_key(marker), model_key(marker), cutoff_key(marker))
        ),
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"Direct-Zarr model bundle missing keys: {missing}")
    if "spatial_subtraction" in bundle:
        from dino.spatial_pixels import SpatialPixels, require_spatial_subtraction_contract

        require_spatial_subtraction_contract(contract, artifact=str(path))
        SpatialPixels.from_spec(bundle["spatial_subtraction"])
    if INTENSITY_MODEL.recognizes(bundle):
        context_required = {
            "intensity_props",
            "intensity_neighbors",
            "intensity_scale_zyx_um",
            "context_features",
        }
        missing_context = sorted(context_required - set(bundle))
        if missing_context:
            raise ValueError(f"Intensity-context model bundle missing keys: {missing_context}")
        if bundle["context_features"] != [
            "log1p_cell_mean",
            "log_cell_over_local_mean",
        ]:
            raise ValueError("Unsupported intensity-context feature contract")
    if bundle["feature_execution"] != FEATURE_EXECUTION:
        raise ValueError(
            f"Expected feature_execution={FEATURE_EXECUTION!r}, got {bundle['feature_execution']!r}"
        )
    return bundle


def _load_cells(cfg: InferConfig) -> pd.DataFrame:
    source_cells_path = cfg.source_dir / "cells.parquet"
    if not source_cells_path.is_file():
        raise FileNotFoundError(f"Missing prepared cell table: {source_cells_path}")
    source_cells = pd.read_parquet(source_cells_path)
    if cfg.cells is None:
        return source_cells.reset_index(drop=True)
    requested = pd.read_parquet(cfg.cells) if cfg.cells.suffix == ".parquet" else pd.read_csv(cfg.cells)
    if "cell" in requested:
        key = "cell"
        requested_key = requested[key].astype(str)
        source_key = source_cells[key].astype(str)
    elif "label" in requested:
        key = "label"
        requested_key = requested[key].astype(np.uint32)
        source_key = source_cells[key].astype(np.uint32)
    else:
        raise ValueError("--cells table must contain cell or label")
    source = source_cells.assign(_request_key=source_key)
    if source["_request_key"].duplicated().any():
        duplicates = source.loc[source["_request_key"].duplicated(keep=False), "_request_key"].unique()
        raise ValueError(f"Prepared cells contain ambiguous {key} values: {duplicates[:10].tolist()}")
    selection = pd.DataFrame(
        {
            "_request_key": requested_key,
            "_request_order": np.arange(len(requested), dtype=np.int64),
        }
    )
    selected = selection.merge(
        source,
        on="_request_key",
        how="left",
        sort=False,
        validate="many_to_one",
        indicator=True,
    )
    missing = selected["_merge"] != "both"
    if missing.any():
        values = selected.loc[missing, "_request_key"].drop_duplicates().head(10).tolist()
        raise ValueError(f"Requested {key} values were not found in prepared cells: {values}")
    return (
        selected.sort_values("_request_order", kind="stable")
        .drop(columns=["_request_key", "_request_order", "_merge"])
        .reset_index(drop=True)
    )


def _iter_source_partitions(path: Path, *, rows_per_partition: int) -> Iterator[pd.DataFrame]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=rows_per_partition):
        yield batch.to_pandas().reset_index(drop=True)


def _iter_frame_partitions(
    frame: pd.DataFrame, *, rows_per_partition: int
) -> Iterator[pd.DataFrame]:
    for start in range(0, len(frame), rows_per_partition):
        yield frame.iloc[start : start + rows_per_partition].reset_index(drop=True)


def _select_shard(
    cells: pd.DataFrame,
    *,
    chunks: tuple[int, int, int],
    shard_index: int,
    shard_count: int,
) -> pd.DataFrame:
    """Balance whole label chunks with deterministic longest-processing-time scheduling."""
    if shard_count == 1:
        return cells.reset_index(drop=True)
    z_chunk, y_chunk, x_chunk = chunks
    chunk_keys = pd.DataFrame(
        {
            "z": cells["plane_z"].to_numpy(dtype=np.int64) // z_chunk,
            "y": np.rint(cells["centroid_y"]).to_numpy(dtype=np.int64) // y_chunk,
            "x": np.rint(cells["centroid_x"]).to_numpy(dtype=np.int64) // x_chunk,
        }
    )
    chunk_codes, unique_chunks = pd.factorize(pd.MultiIndex.from_frame(chunk_keys), sort=False)
    counts = np.bincount(chunk_codes)
    chunk_order = sorted(
        range(len(counts)),
        key=lambda code: (-int(counts[code]), *unique_chunks[code]),
    )
    shard_loads = [(0, index) for index in range(shard_count)]
    chunk_shards = np.empty(len(counts), dtype=np.int64)
    for code in chunk_order:
        load, index = heapq.heappop(shard_loads)
        chunk_shards[code] = index
        heapq.heappush(shard_loads, (load + int(counts[code]), index))
    selected = chunk_shards[chunk_codes] == shard_index
    shard = cells.loc[selected].copy()
    shard_chunks = chunk_keys.loc[selected]
    shard["_shard_z"] = shard_chunks["z"].to_numpy()
    shard["_shard_y"] = shard_chunks["y"].to_numpy()
    shard["_shard_x"] = shard_chunks["x"].to_numpy()
    return (
        shard.sort_values(
            [
                "_shard_z",
                "_shard_y",
                "_shard_x",
                "plane_z",
                "centroid_y",
                "centroid_x",
                "cell",
            ],
            kind="stable",
        )
        .drop(columns=["_shard_z", "_shard_y", "_shard_x"])
        .reset_index(drop=True)
    )


def _predict(
    cells: pd.DataFrame,
    *,
    bundle: dict[str, Any],
    reader: CellCropReader,
    model: Any,
    transform: Any,
    device: torch.device,
    batch_size: int,
    prefetch_workers: int,
    crop_workers: int,
    store_embeddings: bool,
    markers: Sequence[str],
    intensity_context: IntensityContext | None = None,
) -> tuple[pd.DataFrame, dict[str, torch.Tensor] | None]:
    ordered = sort_cells_for_io(cells, reader)
    probabilities_out = {marker: np.full(len(cells), np.nan, dtype=np.float32) for marker in markers}
    calls_out = {marker: np.zeros(len(cells), dtype=bool) for marker in markers}
    valid_out = np.zeros(len(cells), dtype=bool)
    channel_names_by_target = {
        target: tuple(bundle["input_channel_names_by_target"][target]) for target in markers
    }
    precision = str(bundle["feature_precision"])
    dtype = feature_dtype(precision)
    embeddings: dict[str, torch.Tensor] | None = None
    context_dimensions = len(bundle.get("context_features", ()))
    context_features = (
        intensity_context.features_for_labels(
            cells["label"].to_numpy(dtype=np.int64),
            neighbors=int(bundle["intensity_neighbors"]),
        )
        if intensity_context is not None
        else {}
    )
    if bool(context_dimensions) != bool(context_features):
        raise ValueError("Model bundle and loaded intensity context disagree")
    if store_embeddings:
        dimensions = {
            int(bundle[scaler_key(target)].n_features_in_) - context_dimensions for target in markers
        }
        if len(dimensions) != 1:
            raise ValueError("Marker classifiers require different embedding dimensions")
        dimension = dimensions.pop()
        embeddings = {
            target: torch.full(
                (len(cells), dimension),
                torch.nan,
                dtype=dtype,
            )
            for target in markers
        }
    image_batches = iter_model_image_batches(
        ordered,
        bundle=bundle,
        channel_names_by_target=channel_names_by_target,
        reader=reader,
        batch_size=batch_size,
        prefetch_workers=prefetch_workers,
        crop_workers=crop_workers,
        require_mask=False,
    )
    for pending_images, pending_row_orders in iter_prefetched_batches(image_batches):
        for target in markers:
            batch = prepare_model_batch(
                pending_images[target],
                transform=transform,
                device=device,
                dtype=dtype,
            )
            feature_tensor = extract_normalized_feature_tensor(
                model,
                batch,
                precision=precision,
            )
            if embeddings is not None:
                embeddings[target][pending_row_orders] = feature_tensor
            features = feature_tensor.float().numpy()
            if context_dimensions:
                features = np.column_stack(
                    (features, context_features[target][pending_row_orders])
                )
            scaled = bundle[scaler_key(target)].transform(features)
            probability = bundle[model_key(target)].predict_proba(scaled)[:, 1]
            probabilities_out[target][pending_row_orders] = probability
            calls_out[target][pending_row_orders] = probability >= float(bundle[cutoff_key(target)])
        valid_out[pending_row_orders] = True
    result = cells.loc[:, ["cell", "dataset", "roi", "label"]].copy().reset_index(drop=True)
    for marker in markers:
        result[probability_column(marker)] = probabilities_out[marker]
    for marker in markers:
        result[marker] = pd.array(calls_out[marker], dtype="boolean")
    result.loc[~valid_out, list(markers)] = pd.NA
    return result, embeddings


def _parse_args() -> InferConfig:
    parser = argparse.ArgumentParser(description="Run Cell-DINO directly over prepared Zarr cells.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--markers",
        type=Path,
        default=None,
        help="Marker configuration asserted against the model bundle; its contract must equal the bundle's.",
    )
    parser.add_argument("--cells", type=Path, default=None)
    parser.add_argument("--dinov2-repo", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-gb", type=float, default=4.0)
    parser.add_argument("--prefetch-workers", type=int, default=8)
    parser.add_argument("--crop-workers", type=int, default=8)
    parser.add_argument("--partition-cells", type=int, default=100_000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--store-embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    if (
        args.batch_size <= 0
        or args.cache_gb <= 0
        or args.prefetch_workers <= 0
        or args.crop_workers <= 0
        or args.partition_cells <= 0
    ):
        parser.error(
            "--batch-size, --cache-gb, --prefetch-workers, --crop-workers, and "
            "--partition-cells must be positive"
        )
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-count must be positive and --shard-index must be within the count")
    return InferConfig(
        source_dir=args.source_dir.expanduser(),
        model_dir=args.model_dir.expanduser(),
        out_dir=args.out_dir.expanduser(),
        markers=load_config(args.markers.expanduser()).contract if args.markers is not None else None,
        cells=args.cells.expanduser() if args.cells is not None else None,
        dinov2_repo=args.dinov2_repo.expanduser() if args.dinov2_repo is not None else None,
        weights=args.weights.expanduser() if args.weights is not None else None,
        batch_size=args.batch_size,
        cache_gb=args.cache_gb,
        prefetch_workers=args.prefetch_workers,
        crop_workers=args.crop_workers,
        partition_cells=args.partition_cells,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        resume=args.resume,
        store_embeddings=args.store_embeddings,
    )


def _validate_partition(
    part_path: Path,
    embedding_path: Path,
    *,
    cells: pd.DataFrame,
    run_record: dict[str, Any],
    markers: Sequence[str],
    partition_index: int,
) -> bool:
    """Validate any published half before reuse, including interrupted pairs."""
    cell_ids = cells["cell"].astype(str).tolist()
    cell_digest = ordered_cell_sha256(cell_ids)
    part_contract = {
        **run_record,
        "partition_index": partition_index,
        "partition_rows": len(cells),
        "cell_sha256": bytes(cell_digest.tolist()).hex(),
    }
    if part_path.exists():
        metadata, rows = read_parquet_artifact(part_path)
        if metadata != part_contract or rows != len(cells):
            raise ValueError(f"Prediction partition contract mismatch: {part_path}")
        saved_ids = pd.read_parquet(part_path, columns=["cell"])["cell"].astype(str).tolist()
        if saved_ids != cell_ids:
            raise ValueError(f"Prediction partition cell order mismatch: {part_path}")
    if run_record["stores_embeddings"] and embedding_path.exists():
        embeddings = read_embedding_artifact(embedding_path)
        require_embedding_markers(embeddings, markers, path=embedding_path)
        if not torch.equal(embeddings["cell_sha256"], cell_digest) or any(
            embeddings[target].shape != (len(cells), run_record["embedding_dimension"])
            for target in markers
        ):
            raise ValueError(f"Embedding partition cell identity or shape mismatch: {embedding_path}")
    return part_path.is_file() and (not run_record["stores_embeddings"] or embedding_path.is_file())


def main() -> None:
    cfg = _parse_args()
    model_path = cfg.model_dir / "model.joblib"
    bundle = _load_bundle(model_path)
    contract = bundle_contract(bundle, where=str(model_path))
    markers = contract.markers
    if cfg.markers is not None:
        require_same_contract(
            cfg.markers, contract, expected_from="--markers", actual_from=str(model_path)
        )
    require_same_contract(
        contract_of(cfg.source_dir), contract, expected_from=str(cfg.source_dir), actual_from=str(model_path)
    )
    if cfg.store_embeddings and bundle["feature_precision"] != "bfloat16":
        raise ValueError("--store-embeddings requires a bfloat16 model bundle")
    source = load_source(cfg.source_dir)
    source_sha256 = str(source.manifest["source_sha256"])
    if bundle["source_sha256"] != source_sha256:
        raise ValueError("Model bundle was trained from a different direct-Zarr source")
    source_cells_path = cfg.source_dir / "cells.parquet"
    if not source_cells_path.is_file():
        raise FileNotFoundError(f"Missing prepared cell table: {source_cells_path}")
    source_cell_count = pq.ParquetFile(source_cells_path).metadata.num_rows
    intensity_context = None
    if INTENSITY_MODEL.recognizes(bundle):
        intensity_context = IntensityContext.load(
            Path(bundle["intensity_props"]),
            scale_zyx_um=np.asarray(bundle["intensity_scale_zyx_um"], dtype=np.float64),
            markers=markers,
        )
        if len(intensity_context.points_zyx_um) != source_cell_count:
            raise ValueError("Intensity context and prepared source cell counts differ")
    if cfg.cells is None and cfg.shard_count == 1:
        cell_count = source_cell_count

        def cell_partitions() -> Iterator[pd.DataFrame]:
            return _iter_source_partitions(source_cells_path, rows_per_partition=cfg.partition_cells)
    else:
        source_cells = _load_cells(cfg)
        shard_cells = _select_shard(
            source_cells,
            chunks=tuple(source.labels.chunks),
            shard_index=cfg.shard_index,
            shard_count=cfg.shard_count,
        )
        cell_count = len(shard_cells)

        def cell_partitions() -> Iterator[pd.DataFrame]:
            return _iter_frame_partitions(shard_cells, rows_per_partition=cfg.partition_cells)

    output_path = cfg.out_dir / "predictions"
    embedding_path = cfg.out_dir / "embeddings"
    complete_path = cfg.out_dir / "complete.json"
    contract_path = cfg.out_dir / "contract.json"
    fields = {
        "prepared_cells_sha256": file_sha256(source_cells_path),
        "requested_input_sha256": file_sha256(cfg.cells) if cfg.cells is not None else None,
        "requested_cells_sha256": bytes(ordered_cell_sha256(
            str(cell) for partition in cell_partitions() for cell in partition["cell"]
        ).tolist()).hex(),
        "source_sha256": source_sha256,
        "model_source_sha256": str(bundle["model_source_sha256"]),
        "model_sha256": file_sha256(model_path),
        "cells": cell_count,
        "source_cells": source_cell_count,
        "shard_index": cfg.shard_index,
        "shard_count": cfg.shard_count,
        **{cutoff_key(marker): float(bundle[cutoff_key(marker)]) for marker in markers},
        "feature_precision": str(bundle["feature_precision"]),
        "feature_execution": str(bundle["feature_execution"]),
        "partition_cells": cfg.partition_cells,
        "stores_embeddings": cfg.store_embeddings,
        "model_schema": str(bundle["schema"]),
    }
    if intensity_context is not None:
        fields["intensity_props_sha256"] = file_sha256(Path(bundle["intensity_props"]))
    if cfg.store_embeddings:
        fields.update(
            {
                "embedding_format": "pytorch-tensor-v1",
                "embedding_dtype": "bfloat16",
                "embedding_targets": list(markers),
                "embedding_dimension": int(bundle[scaler_key(markers[0])].n_features_in_)
                - len(bundle.get("context_features", ())),
            }
        )
    expected = INFERENCE.stamp(fields, contract)
    repo = cfg.dinov2_repo or Path(bundle["dinov2_repo"]).expanduser()
    weights = cfg.weights or Path(bundle["weights"]).expanduser()
    validate_model_artifacts(bundle=bundle, weights=weights, model_source=repo)
    if cfg.out_dir.exists() and any(cfg.out_dir.iterdir()):
        if not cfg.resume:
            raise FileExistsError(f"Refusing --no-resume with occupied destination: {cfg.out_dir}")
        if not contract_path.is_file():
            raise ValueError(f"Inference output contract mismatch: {cfg.out_dir}")
        run_record = json.loads(contract_path.read_text())
        if INFERENCE.normalize(run_record, where=str(contract_path)) != expected:
            raise ValueError(f"Inference output contract mismatch: {cfg.out_dir}")
    else:
        run_record = expected
    if complete_path.exists() and json.loads(complete_path.read_text()) != run_record:
        raise ValueError(f"Inference completion contract mismatch: {complete_path}")

    completed_partitions = []
    expected_predictions = set()
    expected_embeddings = set()
    for index, partition in enumerate(cell_partitions()):
        part = output_path / f"part-{index:06d}.parquet"
        embedding = embedding_path / f"part-{index:06d}.pt"
        expected_predictions.add(part)
        if cfg.store_embeddings:
            expected_embeddings.add(embedding)
        completed_partitions.append(_validate_partition(
            part, embedding, cells=partition, run_record=run_record, markers=markers, partition_index=index,
        ))
    for directory, expected_paths in (
        (output_path, expected_predictions), (embedding_path, expected_embeddings),
    ):
        if directory.exists() and set(directory.iterdir()) - expected_paths:
            raise ValueError(f"Unexpected inference artifacts: {directory}")
    if complete_path.exists():
        if not all(completed_partitions):
            raise ValueError(f"Completed inference has missing partitions: {cfg.out_dir}")
        print(f"output_path={output_path}")
        print("status=skipped_complete")
        return

    feature_extractor = str(bundle["feature_extractor"])
    if feature_extractor not in MODEL_INPUT_CHANNELS:
        raise ValueError(f"Unsupported feature extractor: {feature_extractor}")
    if not contract_path.exists():
        write_json_atomic(run_record, contract_path, overwrite=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_cell_dino_model(
        repo=repo,
        hub_model=feature_extractor,
        weights=weights,
        device=device,
        precision=str(bundle["feature_precision"]),
    )
    transform = build_cell_transform(
        repo=repo,
        resize_size=int(bundle["resize_size"]),
        crop_size=int(bundle["crop_eval_size"]),
    )
    input_channel_names_by_target = {
        target: tuple(channel_names)
        for target, channel_names in bundle["input_channel_names_by_target"].items()
    }
    input_channel_names = tuple(
        dict.fromkeys(
            name
            for target in markers
            for name in input_channel_names_by_target[target]
        )
    )
    reader = model_crop_reader(
        source,
        cache_gb=cfg.cache_gb,
        channel_names=input_channel_names,
        spatial_subtraction=bundle.get("spatial_subtraction"),
    )
    output_path.mkdir(parents=True, exist_ok=True)
    if cfg.store_embeddings:
        embedding_path.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    try:
        for partition_index, partition_cells in enumerate(cell_partitions()):
            part_path = output_path / f"part-{partition_index:06d}.parquet"
            embedding_part_path = embedding_path / f"part-{partition_index:06d}.pt"
            if completed_partitions[partition_index]:
                rows_written += len(partition_cells)
                continue
            predictions, embeddings = _predict(
                partition_cells,
                bundle=bundle,
                reader=reader,
                model=model,
                transform=transform,
                device=device,
                batch_size=cfg.batch_size,
                prefetch_workers=cfg.prefetch_workers,
                crop_workers=cfg.crop_workers,
                store_embeddings=cfg.store_embeddings,
                markers=markers,
                intensity_context=intensity_context,
            )
            part_contract = {
                **run_record,
                "partition_index": partition_index,
                "partition_rows": len(predictions),
                "cell_sha256": bytes(ordered_cell_sha256(predictions["cell"].astype(str)).tolist()).hex(),
            }
            if not part_path.exists():
                write_parquet_atomic(predictions, part_path, provenance=part_contract)
            if embeddings is not None and not embedding_part_path.exists():
                write_embedding_atomic(
                    {
                        **embeddings,
                        "cell_sha256": ordered_cell_sha256(predictions["cell"].astype(str).tolist()),
                    },
                    embedding_part_path,
                )
            rows_written += len(predictions)
    finally:
        reader.close()
    write_json_atomic(run_record, complete_path, overwrite=False)
    print(f"output_path={output_path}")
    print(f"rows={rows_written}")
    print("status=complete")


if __name__ == "__main__":
    main()
