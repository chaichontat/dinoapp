from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from squisher_segment.segment.regionprops import (
    finalize_measurements,
    measure_block,
    reduce_measurements as reduce_measurements,
)

from dino.artifacts import file_sha256, read_parquet_artifact, write_parquet_atomic
from dino.zarr_source import (
    build_source_manifest,
    resolve_array,
    write_json_atomic,
)


@dataclass(frozen=True)
class PrepareConfig:
    labels: Path
    pi: Path
    edu: Path
    brdu: Path
    offset_zyx: tuple[int, int, int]
    out_dir: Path
    workers: int
    resume: bool
    measured_props: Path | None = None
    dataset: str | None = None
    roi: str | None = None


def _write_polars_atomic(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".dino-", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.write_parquet(temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _chunk_bounds(
    shape: tuple[int, ...], chunks: tuple[int, ...]
) -> list[tuple[int, int, int, int, int, int]]:
    bounds = []
    for z0 in range(0, shape[0], chunks[0]):
        for y0 in range(0, shape[1], chunks[1]):
            for x0 in range(0, shape[2], chunks[2]):
                bounds.append(
                    (
                        z0,
                        min(z0 + chunks[0], shape[0]),
                        y0,
                        min(y0 + chunks[1], shape[1]),
                        x0,
                        min(x0 + chunks[2], shape[2]),
                    )
                )
    return bounds


def _measure_source_chunk(
    labels_root: str,
    labels_array: str,
    bounds: tuple[int, int, int, int, int, int],
) -> tuple[tuple[int, int, int, int, int, int], pl.DataFrame, pl.DataFrame]:
    labels, _ = resolve_array(Path(labels_root), array_path=labels_array or None)
    z0, z1, y0, y1, x0, x1 = bounds
    block = np.asarray(labels[z0:z1, y0:y1, x0:x1])
    moments, planes = measure_block(block, offset_zyx=(z0, y0, x0))
    return bounds, moments, planes


def _partial_paths(partial_dir: Path, bounds: tuple[int, int, int, int, int, int]) -> tuple[Path, Path]:
    z0, _, y0, _, x0, _ = bounds
    suffix = f"{z0}-{y0}-{x0}.parquet"
    return partial_dir / f"m-{suffix}", partial_dir / f"p-{suffix}"


def _partials_complete(moment_path: Path, plane_path: Path, *, source_sha256: str) -> bool:
    expected = {"artifact": "dino-zarr-chunk-v1", "source_sha256": source_sha256}
    for path in (moment_path, plane_path):
        if path.exists():
            metadata, _ = read_parquet_artifact(path)
            if metadata != expected:
                raise ValueError(f"Preparation partial belongs to a different source: {path}")
    return moment_path.is_file() and plane_path.is_file()


def _reduce_partial_files(partial_dir: Path) -> pl.DataFrame:
    moments = (
        pl.scan_parquet(str(partial_dir / "m-*.parquet"))
        .group_by("label")
        .agg(pl.col("area", "z_sum", "y_sum", "x_sum").sum())
        .collect()
    )
    planes = (
        pl.scan_parquet(str(partial_dir / "p-*.parquet"))
        .group_by("label", "plane_z")
        .agg(pl.col("plane_area").sum())
        .collect()
    )
    return finalize_measurements(moments, planes)


def _validate_areas(cells: pl.DataFrame, *, labels_path: Path) -> None:
    masked_dir = labels_path.parent
    volumes_path = masked_dir / "volumes.npy"
    props_path = masked_dir / "props.parquet"
    if not volumes_path.is_file() or not props_path.is_file():
        raise FileNotFoundError(
            f"Expected masked volume references beside labels: {volumes_path}, {props_path}"
        )
    cell_labels = cells["label"].to_numpy().astype(np.int64, copy=False)
    cell_areas = cells["area"].to_numpy().astype(np.uint64, copy=False)
    volumes = np.load(volumes_path, mmap_mode="r")
    if cell_labels.max(initial=0) >= len(volumes):
        raise ValueError("Cell label exceeds masked/volumes.npy index range")
    if not np.array_equal(cell_areas, np.asarray(volumes[cell_labels], dtype=np.uint64)):
        raise ValueError("Reduced label areas do not match masked/volumes.npy")

    props = pl.read_parquet(props_path).select(
        pl.col("label").cast(pl.UInt32), pl.col("area").cast(pl.UInt64)
    )
    observed = cells.select("label", "area").sort("label")
    expected = props.sort("label")
    if not observed.equals(expected):
        raise ValueError("Reduced label areas do not match masked/props.parquet")


def _parse_args() -> PrepareConfig:
    parser = argparse.ArgumentParser(description="Index masked labels for direct-Zarr Cell-DINO.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--pi", type=Path, required=True)
    parser.add_argument("--edu", type=Path, required=True)
    parser.add_argument("--brdu", type=Path, required=True)
    parser.add_argument("--offset-zyx", type=int, nargs=3, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--measured-props", type=Path, help="Existing regionprops table for these postprocessed labels."
    )
    parser.add_argument("--dataset", help="Explicit dataset identity, paired with --roi.")
    parser.add_argument("--roi", help="Explicit ROI/cell namespace, paired with --dataset.")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return PrepareConfig(
        labels=args.labels.expanduser(),
        pi=args.pi.expanduser(),
        edu=args.edu.expanduser(),
        brdu=args.brdu.expanduser(),
        offset_zyx=tuple(args.offset_zyx),
        out_dir=args.out_dir.expanduser(),
        workers=args.workers,
        resume=args.resume,
        measured_props=args.measured_props,
        dataset=args.dataset,
        roi=args.roi,
    )


def _load_measured_cells(path: Path, labels_path: Path) -> pl.DataFrame:
    labels, _ = resolve_array(labels_path)
    volume_spec = labels.attrs.get("label_volumes")
    if volume_spec != {
        "file": "volumes.npy",
        "format": "npy_uint64_indexed_by_final_label",
        "units": "voxels",
        "background_index": 0,
    }:
        raise ValueError("Measured properties require the postprocessed label volume reference")
    cells = (
        pl.read_parquet(path)
        .select("label", "area", "centroid_z", "centroid_y", "centroid_x", "plane_z", "plane_area")
        .sort("label")
    )
    for key in ("label", "area", "plane_z", "plane_area"):
        if not cells.schema[key].is_integer() or cells[key].null_count():
            raise ValueError(f"Measured properties require non-null integer {key}")
    volumes = np.load(labels_path / "volumes.npy", mmap_mode="r", allow_pickle=False)
    cell_labels = cells["label"].to_numpy()
    expected_labels = np.flatnonzero(volumes[1:]) + 1
    if not np.array_equal(cell_labels, expected_labels):
        raise ValueError("Measured properties must cover every postprocessed label exactly once")
    if not np.array_equal(cells["area"].to_numpy(), volumes[cell_labels]):
        raise ValueError("Measured property areas differ from postprocessed label volumes")
    centroids = cells.select("centroid_z", "centroid_y", "centroid_x").to_numpy()
    if (
        not np.isfinite(centroids).all()
        or np.any(centroids < 0)
        or np.any(centroids > np.asarray(labels.shape) - 1)
    ):
        raise ValueError("Measured centroids must lie within the label grid")
    planes = cells["plane_z"].to_numpy()
    plane_areas = cells["plane_area"].to_numpy()
    if (
        np.any(planes < 0)
        or np.any(planes >= labels.shape[0])
        or np.any(plane_areas <= 0)
        or np.any(plane_areas > cells["area"].to_numpy())
    ):
        raise ValueError("Measured maximum-area planes are invalid")
    return cells


def _measure_cells(
    config: PrepareConfig,
    *,
    labels_root: str,
    labels_array: str,
    source_sha256: str,
) -> pl.DataFrame:
    partial_dir = config.out_dir / "partials"
    partial_dir.mkdir(exist_ok=True)
    labels, _ = resolve_array(config.labels, array_path=labels_array or None)
    bounds = _chunk_bounds(tuple(labels.shape), tuple(labels.chunks))
    pending = [
        bound
        for bound in bounds
        if not (
            config.resume
            and _partials_complete(*_partial_paths(partial_dir, bound), source_sha256=source_sha256)
        )
    ]
    logger.info(f"Label chunks total={len(bounds)} pending={len(pending)} workers={config.workers}")
    provenance = {"artifact": "dino-zarr-chunk-v1", "source_sha256": source_sha256}
    if config.workers == 1:
        results = (_measure_source_chunk(labels_root, labels_array, bound) for bound in pending)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=config.workers)
        results = executor.map(
            _measure_source_chunk,
            [labels_root] * len(pending),
            [labels_array] * len(pending),
            pending,
        )
    try:
        for index, (bound, moments, planes) in enumerate(results, start=1):
            moment_path, plane_path = _partial_paths(partial_dir, bound)
            if not moment_path.exists():
                write_parquet_atomic(moments.to_pandas(), moment_path, provenance=provenance)
            if not plane_path.exists():
                write_parquet_atomic(planes.to_pandas(), plane_path, provenance=provenance)
            if index % 10 == 0 or index == len(pending):
                logger.info(f"Measured pending chunks={index}/{len(pending)}")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    cells = _reduce_partial_files(partial_dir)
    _validate_areas(cells, labels_path=config.labels)
    return cells


def main() -> None:
    config = _parse_args()
    manifest = build_source_manifest(
        labels_path=config.labels,
        channel_paths={"pi": config.pi, "edu": config.edu, "brdu": config.brdu},
        offset_zyx=config.offset_zyx,
        dataset=config.dataset,
        roi=config.roi,
    )
    source_sha256 = str(manifest["source_sha256"])
    if not config.resume and config.out_dir.exists() and any(config.out_dir.iterdir()):
        raise FileExistsError(f"Refusing --no-resume with occupied destination: {config.out_dir}")
    config.out_dir.mkdir(parents=True, exist_ok=True)
    source_path = config.out_dir / "source.json"
    complete_path = config.out_dir / "complete.json"
    cells_path = config.out_dir / "cells.parquet"
    if source_path.is_file():
        existing = json.loads(source_path.read_text())
        if existing != manifest:
            raise ValueError(f"Output directory belongs to a different source: {config.out_dir}")
    elif any(config.out_dir.iterdir()):
        raise ValueError(f"Occupied preparation directory has no source manifest: {config.out_dir}")
    if complete_path.exists():
        complete = json.loads(complete_path.read_text())
        if complete.get("source_sha256") != source_sha256 or complete.get("cells_sha256") != file_sha256(
            cells_path
        ):
            raise ValueError(f"Preparation completion contract mismatch: {complete_path}")
        if config.measured_props is not None and complete.get("measured_props_sha256") != file_sha256(
            config.measured_props
        ):
            raise ValueError("Prepared source measured properties changed")
        print(f"source_dir={config.out_dir}")
        print("status=skipped_complete")
        return
    if not source_path.exists():
        write_json_atomic(manifest, source_path, overwrite=False)

    if config.measured_props is not None:
        cells = _load_measured_cells(config.measured_props, config.labels)
    else:
        cells = _measure_cells(
            config,
            labels_root=manifest["labels"]["root"],
            labels_array=manifest["labels"]["array"],
            source_sha256=source_sha256,
        )
    cells = cells.with_columns(
        pl.lit(manifest["dataset"]).alias("dataset"),
        pl.lit(manifest["roi"]).alias("roi"),
        (pl.lit(f"{manifest['roi']}|") + pl.col("label").cast(pl.String)).alias("cell"),
    ).select(
        "cell",
        "dataset",
        "roi",
        "label",
        "area",
        "centroid_z",
        "centroid_y",
        "centroid_x",
        "plane_z",
        "plane_area",
    )
    if cells_path.exists():
        if not pl.read_parquet(cells_path).equals(cells):
            raise ValueError(f"Prepared cell table differs from measured labels: {cells_path}")
    else:
        _write_polars_atomic(cells, cells_path)
    complete = {
        "artifact": "dino-zarr-source-complete-v1",
        "source_sha256": source_sha256,
        "cells": cells.height,
        "cells_sha256": file_sha256(cells_path),
    }
    if config.measured_props is not None:
        complete["measured_props_sha256"] = file_sha256(config.measured_props)
    write_json_atomic(complete, complete_path, overwrite=False)
    print(f"source_dir={config.out_dir}")
    print(f"cells={cells.height}")
    print("status=complete")


if __name__ == "__main__":
    main()
