from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dino_markers import (
    BARRAGE,
    BARRAGE_STATS,
    MarkerConfig,
    declared_contract,
    image_path_column,
    load_config,
    require_same_contract,
    resolve_directory,
    save_config,
    stat_column,
)
from loguru import logger
from PIL import Image

from dino.artifacts import write_csv_atomic
from dino.crop_rendering import BARRAGE_CMAP, FIGURE_DPI, read_crop, render_crop
from dino.zarr_source import (
    CellCropReader,
    OpenSource,
    load_source,
    write_json_atomic,
)


DISPLAY_MIN = 0.0
DISPLAY_PERCENTILE = 99.9
DISPLAY_SAMPLE_Z_PLANES = 32
DISPLAY_SAMPLE_XY_STRIDE = 8
RENDERER_VERSION = "v3"


@dataclass(frozen=True)
class BarrageConfig:
    config: MarkerConfig
    source_dir: Path
    out_dir: Path
    candidate_cells: int
    n_cells: int
    size: int
    seed: int
    n_bins: int
    cache_gb: float
    eligibility_table: Path | None
    eligibility_column: str
    exclude_index: Path | None
    resume: bool
    pixel_model: Path | None = None
    reuse_index: Path | None = None
    display_overrides: Mapping[str, float] | None = None


def feature_columns(markers: Sequence[str]) -> tuple[str, ...]:
    """The mean and standard deviation columns that stratify the feature sample, per marker."""
    return tuple(stat_column(marker, stat) for marker in markers for stat in ("mean", "std"))


def parse_display_overrides(values: list[str], *, keys: Sequence[str]) -> dict[str, float]:
    """Parse explicit channel display maxima without changing the source cache."""
    overrides: dict[str, float] = {}
    for value in values:
        channel, separator, maximum_text = value.partition("=")
        if separator != "=" or channel not in keys:
            raise ValueError(f"Display maximum must be KEY=VALUE for one of {keys}")
        try:
            maximum = float(maximum_text)
        except ValueError as error:
            raise ValueError(f"Invalid display maximum: {value}") from error
        if not np.isfinite(maximum) or maximum <= DISPLAY_MIN:
            raise ValueError(f"Display maximum must be finite and positive: {value}")
        if channel in overrides:
            raise ValueError(f"Duplicate display maximum for {channel}")
        overrides[channel] = maximum
    return overrides


def masked_channel_stats(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if image.shape != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {image.shape} != {mask.shape}")
    values = np.asarray(image[mask], dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot measure an empty target-cell mask")
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
    }


def _display_cache_contract(source: OpenSource) -> dict[str, object]:
    z_size = int(source.manifest["labels"]["shape"][0])
    z_indices = np.unique(np.linspace(0, z_size - 1, DISPLAY_SAMPLE_Z_PLANES, dtype=np.int64)).tolist()
    return {
        "artifact": "dino-zarr-display-v1",
        "source_sha256": str(source.manifest["source_sha256"]),
        "percentile": DISPLAY_PERCENTILE,
        "z_indices": z_indices,
        "xy_stride": DISPLAY_SAMPLE_XY_STRIDE,
    }


def _measure_display_max(source: OpenSource) -> tuple[dict[str, float], int]:
    """Estimate per-channel display maxima over a deterministic sample of the label ROI."""
    contract = _display_cache_contract(source)
    z_offset, y_offset, x_offset = source.offset_zyx
    _, y_size, x_size = (int(value) for value in source.manifest["labels"]["shape"])
    maxima: dict[str, float] = {}
    samples_per_channel = 0

    for name, array in source.channels.items():
        counts = np.zeros(np.iinfo(array.dtype).max + 1, dtype=np.uint64)
        channel_samples = 0
        for local_z in contract["z_indices"]:
            plane = np.asarray(
                array[
                    z_offset + int(local_z),
                    y_offset : y_offset + y_size,
                    x_offset : x_offset + x_size,
                ]
            )
            sampled = plane[::DISPLAY_SAMPLE_XY_STRIDE, ::DISPLAY_SAMPLE_XY_STRIDE]
            counts += np.bincount(sampled.ravel(), minlength=counts.size).astype(np.uint64)
            channel_samples += int(sampled.size)
        target = int(np.ceil(channel_samples * DISPLAY_PERCENTILE / 100.0))
        maximum = int(np.searchsorted(np.cumsum(counts), target, side="left"))
        if maximum <= DISPLAY_MIN:
            raise ValueError(
                f"Non-positive {DISPLAY_PERCENTILE:g}th percentile for display channel {name}: {maximum}"
            )
        maxima[name] = float(maximum)
        if samples_per_channel not in (0, channel_samples):
            raise ValueError(
                f"Display sampling count mismatch for {name}: {channel_samples} != {samples_per_channel}"
            )
        samples_per_channel = channel_samples
    return maxima, samples_per_channel


def _load_or_measure_display_max(source_dir: Path, source: OpenSource) -> dict[str, float]:
    cache_path = source_dir / "display.json"
    contract = _display_cache_contract(source)
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text())
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid display cache JSON: {cache_path}") from error
        if not isinstance(cached, Mapping):
            raise ValueError(f"Display cache must contain a JSON object: {cache_path}")
        if all(cached.get(key) == value for key, value in contract.items()):
            cached_maxima = cached.get("vmax")
            if not isinstance(cached_maxima, Mapping) or set(cached_maxima) != set(source.channels):
                raise ValueError(f"Invalid display maxima in cache: {cache_path}")
            maxima = {name: float(maximum) for name, maximum in cached_maxima.items()}
            if not all(np.isfinite(maximum) and maximum > DISPLAY_MIN for maximum in maxima.values()):
                raise ValueError(f"Display maxima must be finite and positive: {cache_path}")
            logger.info(f"Using cached p{DISPLAY_PERCENTILE:g} display maxima from {cache_path}: {maxima}")
            return maxima
        logger.info(f"Display cache is stale; recomputing {cache_path}")

    maxima, samples_per_channel = _measure_display_max(source)
    write_json_atomic(
        {**contract, "samples_per_channel": samples_per_channel, "vmax": maxima},
        cache_path,
    )
    logger.info(f"Cached p{DISPLAY_PERCENTILE:g} display maxima at {cache_path}: {maxima}")
    return maxima


def stratified_sample(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    n_rows: int,
    n_bins: int,
    seed: int,
) -> pd.DataFrame:
    if n_rows <= 0 or n_bins < 2:
        raise ValueError("n_rows must be positive and n_bins must be at least 2")
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing stratification columns: {missing}")
    if len(frame) <= n_rows:
        return frame.copy().reset_index(drop=True)

    work = frame.copy()
    bin_columns: list[str] = []
    for index, column in enumerate(columns):
        values = work[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite stratification values in {column}")
        percentiles = pd.Series(values).rank(method="average", pct=True).to_numpy()
        bin_column = f"_bin_{index}"
        work[bin_column] = np.minimum((percentiles * n_bins).astype(np.int16), n_bins - 1)
        bin_columns.append(bin_column)

    rng = np.random.default_rng(seed)
    work["_random"] = rng.random(len(work))
    work = work.sort_values([*bin_columns, "_random"], kind="stable")
    work["_round"] = work.groupby(bin_columns, sort=False).cumcount()
    selected = work.sort_values(["_round", "_random"], kind="stable").head(n_rows)
    return selected.drop(columns=[*bin_columns, "_random", "_round"]).reset_index(drop=True)


def _geometry_sample(cells: pd.DataFrame, *, n_rows: int, seed: int) -> pd.DataFrame:
    work = cells.copy()
    work["_log_area"] = np.log1p(work["area"].to_numpy(dtype=np.float64))
    sampled = stratified_sample(
        work,
        columns=("_log_area", "centroid_z", "centroid_y", "centroid_x"),
        n_rows=n_rows,
        n_bins=8,
        seed=seed,
    )
    return sampled.drop(columns="_log_area")


def _feature_sample(
    cells: pd.DataFrame, *, markers: Sequence[str], n_rows: int, n_bins: int, seed: int
) -> pd.DataFrame:
    work = cells.copy()
    columns = []
    for column in feature_columns(markers):
        transformed = f"_log_{column}"
        work[transformed] = np.log1p(work[column].to_numpy(dtype=np.float64))
        columns.append(transformed)
    sampled = stratified_sample(
        work,
        columns=tuple(columns),
        n_rows=n_rows,
        n_bins=n_bins,
        seed=seed,
    )
    return sampled.drop(columns=columns)


def _prioritize_existing(
    source: pd.DataFrame,
    *,
    existing_cells: list[str],
    n_rows: int,
    sampler: Callable[[pd.DataFrame, int], pd.DataFrame],
) -> pd.DataFrame:
    existing_set = set(existing_cells)
    existing_mask = source["cell"].astype(str).isin(existing_set)
    existing = source[existing_mask].drop_duplicates("cell")
    remaining_n = max(0, n_rows - len(existing))
    remaining = source[~existing_mask]
    selected = sampler(remaining, remaining_n) if remaining_n else remaining.iloc[:0]
    return pd.concat([existing, selected], ignore_index=True).head(n_rows)


def _sort_for_io(frame: pd.DataFrame, *, label_chunks: tuple[int, int, int]) -> pd.DataFrame:
    cz, cy, cx = label_chunks
    work = frame.copy()
    work["_io_z"] = work["plane_z"].astype(np.int64) // cz
    work["_io_y"] = np.rint(work["centroid_y"]).astype(np.int64) // cy
    work["_io_x"] = np.rint(work["centroid_x"]).astype(np.int64) // cx
    return work.sort_values(["_io_z", "_io_y", "_io_x", "cell"], kind="stable").drop(
        columns=["_io_z", "_io_y", "_io_x"]
    )


def _measure_candidates(
    candidates: pd.DataFrame,
    *,
    reader: CellCropReader,
    size: int,
    markers: Sequence[str],
    pixel_correction=None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordered = _sort_for_io(candidates, label_chunks=tuple(reader.labels.chunks))
    for index, (_, row) in enumerate(ordered.iterrows(), start=1):
        crop = read_crop(reader, row, size=size, pixel_correction=pixel_correction)
        if not crop.mask.any():
            raise ValueError(f"Target cell mask is empty on plane_z for {row['cell']}")
        output = row.to_dict()
        for marker in markers:
            stats = masked_channel_stats(crop.channels[marker], crop.mask)
            output.update({stat_column(marker, name): value for name, value in stats.items()})
        rows.append(output)
        if index % 1000 == 0:
            logger.info(f"Measured candidate cells={index}/{len(ordered)}")
    return pd.DataFrame(rows)


def _read_record(path: Path) -> dict[str, object]:
    record = json.loads(path.read_text())
    if not isinstance(record, dict):
        raise ValueError(f"Barrage record must be a JSON object: {path}")
    return record


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _apply_eligibility(
    cells: pd.DataFrame,
    *,
    table_path: Path,
    column: str,
) -> pd.DataFrame:
    eligibility = pd.read_parquet(table_path, columns=["cell", column])
    if eligibility["cell"].isna().any() or eligibility[column].isna().any():
        raise ValueError(f"Eligibility table contains null cell or {column} values: {table_path}")
    if eligibility["cell"].duplicated().any():
        raise ValueError(f"Eligibility table contains duplicate cells: {table_path}")
    if not pd.api.types.is_bool_dtype(eligibility[column]):
        raise ValueError(f"Eligibility column must be boolean: {column}")

    source_cells = cells["cell"].astype(str)
    eligibility = eligibility.assign(cell=eligibility["cell"].astype(str))
    selected = eligibility.set_index("cell")[column].reindex(source_cells)
    if len(eligibility) != len(cells) or selected.isna().any():
        raise ValueError("Eligibility table must contain every source cell exactly once")
    filtered = cells.loc[selected.to_numpy()].copy().reset_index(drop=True)
    if filtered.empty:
        raise ValueError(f"Eligibility table selects no cells with {column}=true: {table_path}")
    return filtered


def _exclude_index(cells: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, int, int]:
    excluded = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "cell" not in excluded:
        raise ValueError(f"Exclusion index is missing cell column: {path}")
    if excluded["cell"].isna().any() or excluded["cell"].duplicated().any():
        raise ValueError(f"Exclusion index cells must be non-null and unique: {path}")
    excluded_cells = set(excluded["cell"].astype(str))
    keep = ~cells["cell"].astype(str).isin(excluded_cells)
    removed = int((~keep).sum())
    return cells.loc[keep].reset_index(drop=True), len(excluded_cells), removed


def _parse_args() -> BarrageConfig:
    parser = argparse.ArgumentParser(description="Generate a direct-Zarr marker labeling barrage.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--markers",
        type=Path,
        help="Marker configuration asserted against the source; its contract must equal the source's.",
    )
    parser.add_argument(
        "--pixel-model",
        type=Path,
        help="Model bundle containing the shared pixel correction.",
    )
    parser.add_argument(
        "--reuse-index",
        type=Path,
        help="Regenerate exactly these cells in their existing review order.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--candidate-cells", type=int, default=20_000)
    parser.add_argument("--n-cells", type=int, default=4_000)
    parser.add_argument("--size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-bins", type=int, default=6)
    parser.add_argument("--cache-gb", type=float, default=4.0)
    parser.add_argument(
        "--display-max",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a rendered channel maximum without changing the source display cache.",
    )
    parser.add_argument("--eligibility-table", type=Path)
    parser.add_argument("--eligibility-column", default="inside_cortex")
    parser.add_argument("--exclude-index", type=Path)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.candidate_cells < args.n_cells:
        parser.error("--candidate-cells must be at least --n-cells")
    if args.n_cells <= 0 or args.size <= 0 or args.cache_gb <= 0:
        parser.error("cell counts, crop size, and cache size must be positive")
    source_dir = args.source_dir.expanduser()
    config = resolve_directory(
        source_dir, supplied=load_config(args.markers.expanduser()) if args.markers else None
    )
    try:
        display_overrides = parse_display_overrides(args.display_max, keys=config.contract.channel_keys)
    except ValueError as error:
        parser.error(str(error))
    return BarrageConfig(
        config=config,
        source_dir=source_dir,
        out_dir=args.out_dir.expanduser(),
        candidate_cells=args.candidate_cells,
        n_cells=args.n_cells,
        size=args.size,
        seed=args.seed,
        n_bins=args.n_bins,
        cache_gb=args.cache_gb,
        eligibility_table=(args.eligibility_table.expanduser() if args.eligibility_table else None),
        eligibility_column=args.eligibility_column,
        exclude_index=(args.exclude_index.expanduser() if args.exclude_index else None),
        resume=args.resume,
        pixel_model=args.pixel_model,
        reuse_index=args.reuse_index,
        display_overrides=display_overrides,
    )


def main() -> None:
    cfg = _parse_args()
    if not cfg.resume and cfg.out_dir.exists() and any(cfg.out_dir.iterdir()):
        raise ValueError("Output directory is nonempty; enable --resume or use a new directory")
    source = load_source(cfg.source_dir)
    contract = cfg.config.contract
    missing_channels = [key for key in contract.channel_keys if key not in source.channels]
    if missing_channels:
        raise ValueError(
            f"Source {cfg.source_dir} has no channels {missing_channels} required by the marker contract: "
            f"{contract}"
        )
    declared = declared_contract(cfg.out_dir)
    if declared is not None:
        require_same_contract(
            contract, declared, expected_from=str(cfg.source_dir), actual_from=str(cfg.out_dir)
        )
    manifest = source.manifest
    reader = CellCropReader(
        labels=source.labels,
        channels=source.channels,
        offset_zyx=source.offset_zyx,
        cache_bytes=int(cfg.cache_gb * 1024**3),
    )
    pixel_correction = None
    pixel_contract = {}
    if cfg.pixel_model is not None:
        from dino.periodic_pixels import PeriodicPixels, require_periodic_targets

        bundle = joblib.load(cfg.pixel_model)
        if bundle["source_sha256"] != manifest["source_sha256"]:
            raise ValueError("Pixel correction model uses a different source")
        correction = bundle["pixel_correction"]
        pixel_correction = PeriodicPixels(
            correction["model"],
            correction["model_sha256"],
            tuple(correction["scale_zyx_um"]),
        )
        require_periodic_targets(pixel_correction, contract, artifact=str(cfg.pixel_model))
        pixel_contract = {"pixel_correction": correction}
    source_sha256 = str(manifest["source_sha256"])
    display_max = _load_or_measure_display_max(cfg.source_dir, source)
    display_overrides = dict(cfg.display_overrides or {})
    display_max.update(display_overrides)
    cells_path = cfg.source_dir / "cells.parquet"
    if not cells_path.is_file():
        raise FileNotFoundError(f"Missing prepared cell table: {cells_path}")
    cells = pd.read_parquet(cells_path)
    eligibility_contract: dict[str, object] = {}
    if cfg.eligibility_table is not None:
        if not cfg.eligibility_table.is_file():
            raise FileNotFoundError(f"Missing eligibility table: {cfg.eligibility_table}")
        cells = _apply_eligibility(
            cells,
            table_path=cfg.eligibility_table,
            column=cfg.eligibility_column,
        )
        eligibility_contract = {
            "eligibility_table": str(cfg.eligibility_table.resolve()),
            "eligibility_sha256": _sha256(cfg.eligibility_table),
            "eligibility_column": cfg.eligibility_column,
            "eligible_cells": len(cells),
        }
    exclusion_contract: dict[str, object] = {}
    if cfg.exclude_index is not None:
        if not cfg.exclude_index.is_file():
            raise FileNotFoundError(f"Missing exclusion index: {cfg.exclude_index}")
        cells, excluded_cells, excluded_eligible_cells = _exclude_index(cells, cfg.exclude_index)
        if cells.empty:
            raise ValueError(f"Exclusion index removes every eligible cell: {cfg.exclude_index}")
        exclusion_contract = {
            "exclude_index": str(cfg.exclude_index.resolve()),
            "exclude_index_sha256": _sha256(cfg.exclude_index),
            "excluded_cells": excluded_cells,
            "excluded_eligible_cells": excluded_eligible_cells,
            "sampling_pool_cells": len(cells),
        }
    reference = None
    reuse_contract = {}
    if cfg.reuse_index is not None:
        reference = pd.read_csv(cfg.reuse_index)
        if reference.cell.duplicated().any() or not reference.cell.isin(cells.cell).all():
            raise ValueError("Reused barrage index contains duplicate or unavailable cells")
        if len(reference) != cfg.n_cells:
            raise ValueError("Reused index row count must equal --n-cells")
        cells = cells.set_index("cell").loc[reference.cell].reset_index()
        reuse_contract = {
            "reuse_index": str(cfg.reuse_index.resolve()),
            "reuse_index_sha256": _sha256(cfg.reuse_index),
        }
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    complete_path = cfg.out_dir / "complete.json"
    run_path = cfg.out_dir / "run.json"
    expected = BARRAGE.stamp(
        {
            "source_sha256": source_sha256,
            "candidate_cells": cfg.candidate_cells,
            "n_cells": cfg.n_cells,
            "size": cfg.size,
            "seed": cfg.seed,
            "n_bins": cfg.n_bins,
            "display_min": DISPLAY_MIN,
            "display_percentile": DISPLAY_PERCENTILE,
            "display_max": display_max,
            "display_overrides": display_overrides,
            "cmap": BARRAGE_CMAP,
            "figure_dpi": FIGURE_DPI,
            "renderer": RENDERER_VERSION,
            "crop_panels": [list(panel) for panel in cfg.config.panels],
            **eligibility_contract,
            **exclusion_contract,
            **pixel_contract,
            **reuse_contract,
        },
        contract,
    )
    run_record = _read_record(run_path) if run_path.exists() else None
    if complete_path.is_file():
        complete = _read_record(complete_path)
        if BARRAGE.normalize(complete, where=str(complete_path)) == expected and run_record in (
            None,
            complete,
        ):
            print(f"barrage_dir={cfg.out_dir}")
            print("status=skipped_complete")
            return
        raise ValueError("Existing barrage has a different rendering contract; use a new output directory")

    if run_record is None:
        run_record = expected
    elif BARRAGE.normalize(run_record, where=str(run_path)) != expected:
        raise ValueError("Partial barrage has a different rendering contract")
    if run_record["artifact"] == BARRAGE.current:
        save_config(cfg.config, cfg.out_dir)
    write_json_atomic(run_record, run_path)
    write_json_atomic(manifest, cfg.out_dir / "source.json")
    labels_path = cfg.out_dir / "labels.csv"
    existing_cells: list[str] = []
    if labels_path.is_file():
        labels = pd.read_csv(labels_path)
        missing_columns = [column for column in ("cell", *contract.markers) if column not in labels.columns]
        if missing_columns:
            raise ValueError(f"Existing labels.csv is missing columns {missing_columns}: {labels_path}")
        existing_cells = labels["cell"].astype(str).drop_duplicates().tolist()

    candidates = _prioritize_existing(
        cells,
        existing_cells=existing_cells,
        n_rows=min(cfg.candidate_cells, len(cells)),
        sampler=lambda frame, n: _geometry_sample(frame, n_rows=n, seed=cfg.seed),
    )
    measured = _measure_candidates(
        candidates, reader=reader, size=cfg.size, markers=contract.markers, pixel_correction=pixel_correction
    )
    _atomic_parquet(measured, cfg.out_dir / "candidates.parquet")
    selected = _prioritize_existing(
        measured,
        existing_cells=existing_cells,
        n_rows=min(cfg.n_cells, len(measured)),
        sampler=lambda frame, n: _feature_sample(
            frame, markers=contract.markers, n_rows=n, n_bins=cfg.n_bins, seed=cfg.seed
        ),
    ).reset_index(drop=True)

    if reference is not None:
        selected = measured.set_index("cell").loc[reference.cell].reset_index()
    images_dir = cfg.out_dir / f"images-{RENDERER_VERSION}-{cfg.size}"
    images_dir.mkdir(exist_ok=True)
    index_rows: list[dict[str, object]] = []
    ordered = _sort_for_io(selected, label_chunks=tuple(reader.labels.chunks))
    rendered: dict[str, str] = {}
    ranks = {str(cell): rank for rank, cell in enumerate(selected["cell"].astype(str))}
    for _, row in ordered.iterrows():
        cell = str(row["cell"])
        rank = ranks[cell]
        safe_cell = re.sub(r"[^a-zA-Z0-9._-]+", "_", cell.replace("|", "__")).strip("._-")
        relative = images_dir.relative_to(cfg.out_dir) / f"{rank:04d}_{safe_cell or 'cell'}.png"
        image_path = cfg.out_dir / relative
        if image_path.is_file():
            try:
                with Image.open(image_path) as image:
                    if image.format != "PNG":
                        raise ValueError(f"Existing barrage image is not a PNG: {image_path}")
                    image.verify()
            except (OSError, SyntaxError) as error:
                raise ValueError(f"Invalid existing barrage PNG: {image_path}") from error
        else:
            crop = read_crop(reader, row, size=cfg.size, pixel_correction=pixel_correction)
            with tempfile.NamedTemporaryFile(dir=image_path.parent, suffix=".tmp", delete=False) as f:
                temporary = Path(f.name)
            try:
                render_crop(crop, cell=cell, display_max=display_max, panels=cfg.config.panels).save(
                    temporary, format="PNG"
                )
                os.replace(temporary, image_path)
            finally:
                temporary.unlink(missing_ok=True)
        rendered[cell] = str(relative)

    for rank, (_, row) in enumerate(selected.iterrows()):
        output: dict[str, object] = {
            "rank": rank,
            "cell": str(row["cell"]),
            "dataset": str(row["dataset"]),
            "roi": str(row["roi"]),
            "label": int(row["label"]),
            "x": float(row["centroid_x"]),
            "y": float(row["centroid_y"]),
            "z": float(row["centroid_z"]),
            "plane_z": int(row["plane_z"]),
            "area": int(row["area"]),
            "plane_area": int(row["plane_area"]),
            "thumb_shape": str((cfg.size, cfg.size, 3)),
            "image_path": rendered[str(row["cell"])],
            "source_sha256": source_sha256,
        }
        for marker in contract.markers:
            for statistic in BARRAGE_STATS:
                column = stat_column(marker, statistic)
                output[column] = float(row[column])
        index_rows.append(output)
    index = pd.DataFrame(index_rows)
    if reference is not None:
        # Retain native-frame bins and review rounds while refreshing measured intensities.
        artifact_paths = {"combined_image_path", *(image_path_column(marker) for marker in contract.markers)}
        for column in reference.columns.difference([*index.columns, *artifact_paths], sort=False):
            index[column] = reference[column].to_numpy()
    write_csv_atomic(index, cfg.out_dir / "index.csv")
    write_json_atomic(run_record, complete_path)
    print(f"barrage_dir={cfg.out_dir}")
    print(f"candidates={len(measured)}")
    print(f"cells={len(selected)}")
    print("status=complete")


if __name__ == "__main__":
    main()
