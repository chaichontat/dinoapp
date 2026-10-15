from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from dino_markers import (
    DEFAULT_CONTRACT,
    FEATURE_CACHE,
    ZARR_MODEL,
    MarkerContract,
    contract_of,
    cutoff_key,
    load_config,
    metric_key,
    model_key,
    require_same_contract,
    scaler_key,
)
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from dino.artifacts import _dump_joblib_atomic, file_sha256
from dino.cell_dino import installed_dinov2_source
from dino.training_split import split_training_rows
from dino.zarr_model import (
    DEFAULT_INPUT_CHANNEL_NAMES,
    FEATURE_EXECUTION,
    FEATURE_PRECISIONS,
    MODEL_INPUT_CHANNELS,
    balanced_accuracy_cutoff,
    build_cell_transform,
    extract_normalized_features,
    feature_dtype,
    iter_model_image_batches,
    iter_prefetched_batches,
    load_cell_dino_model,
    model_crop_reader,
    prepare_model_batch,
    sort_cells_for_io,
)
from dino.zarr_source import OpenSource, load_source


@dataclass(frozen=True)
class TrainConfig:
    barrage_dirs: tuple[Path, ...]
    target_source_dir: Path | None
    out_dir: Path
    feature_cache: Path | None
    hub_model: str
    weights: Path
    crop_size: int
    resize_size: int
    crop_eval_size: int
    batch_size: int
    cv_folds: int
    test_fraction: float
    limit: int
    seed: int
    cache_gb: float
    prefetch_workers: int
    crop_workers: int
    highpass_sigma_px: float
    mask_erosion_px: int
    feature_precision: str
    input_channel_names_by_target: Mapping[str, tuple[str, ...]]
    contract: MarkerContract

    @property
    def input_channels(self) -> int:
        return MODEL_INPUT_CHANNELS[self.hub_model]


def resolve_input_channels(
    values: Sequence[str], contract: MarkerContract, *, slots: int
) -> dict[str, tuple[str, ...]]:
    """Parse repeated `TARGET=CHANNEL` flags into each marker's ordered model input channels."""
    channels: dict[str, list[str]] = {marker: [] for marker in contract.markers}
    for value in values:
        target, separator, channel = value.partition("=")
        if not separator or not target or not channel:
            raise ValueError(f"--input-channel must be TARGET=CHANNEL, got {value!r}")
        if target not in channels:
            raise ValueError(f"--input-channel target {target!r} is not a marker of {contract}")
        if channel not in contract.channel_keys:
            raise ValueError(
                f"--input-channel channel {channel!r} is not a channel of {contract}; "
                f"channels are {list(contract.channel_keys)}"
            )
        if channel in channels[target]:
            raise ValueError(f"--input-channel values for {target!r} must be unique, got {channel!r} twice")
        channels[target].append(channel)
    if contract == DEFAULT_CONTRACT:
        channels = {target: names or list(DEFAULT_INPUT_CHANNEL_NAMES) for target, names in channels.items()}
    else:
        missing = [target for target, names in channels.items() if not names]
        if missing:
            raise ValueError(f"--input-channel is required for every marker of {contract}; missing {missing}")
    too_many = {target: len(names) for target, names in channels.items() if len(names) > slots}
    if too_many:
        raise ValueError(f"More --input-channel values were supplied than the model's {slots} slots accept: {too_many}")
    return {target: tuple(names) for target, names in channels.items()}


def _training_contract(
    barrage_dirs: Sequence[Path], target_source_dir: Path | None, markers: Path | None
) -> MarkerContract:
    """The contract shared by every barrage and the target source, which `--markers` must equal."""
    contract = contract_of(barrage_dirs[0])
    for directory in (*barrage_dirs[1:], target_source_dir or barrage_dirs[0]):
        require_same_contract(
            contract,
            contract_of(directory),
            expected_from=str(barrage_dirs[0]),
            actual_from=str(directory),
        )
    if markers is not None:
        require_same_contract(
            load_config(markers).contract, contract, expected_from="--markers", actual_from=str(barrage_dirs[0])
        )
    return contract


def _load_training_rows(barrage_dir: Path, *, contract: MarkerContract) -> pd.DataFrame:
    index_path = barrage_dir / "index.csv"
    labels_path = barrage_dir / "labels.csv"
    if not index_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(f"Expected index.csv and labels.csv under {barrage_dir}")
    index = pd.read_csv(index_path)
    labels = pd.read_csv(labels_path)
    required_index = {
        "cell",
        "label",
        "centroid_z",
        "centroid_y",
        "centroid_x",
        "plane_z",
        "source_sha256",
    }
    # Barrage compatibility uses x/y/z names; normalize without changing index.csv.
    if "centroid_x" not in index and "x" in index:
        index["centroid_x"] = index["x"]
    if "centroid_y" not in index and "y" in index:
        index["centroid_y"] = index["y"]
    if "centroid_z" not in index and "z" in index:
        index["centroid_z"] = index["z"]
    missing = sorted(required_index - set(index.columns))
    if missing:
        raise ValueError(f"Direct-Zarr index.csv missing required columns: {missing}")
    required_labels = {"cell", *contract.markers}
    missing = sorted(required_labels - set(labels.columns))
    if missing:
        raise ValueError(f"labels.csv missing required columns: {missing}")
    rows = labels.merge(index, on="cell", how="inner").drop_duplicates("cell", keep="last")
    for marker in contract.markers:
        rows[marker] = rows[marker].astype(np.int64)
    return rows.reset_index(drop=True)


def _load_training_sources(barrage_dir: Path) -> dict[str, OpenSource]:
    source_paths = [barrage_dir / "source.json", *sorted(barrage_dir.glob("*/source.json"))]
    sources: dict[str, OpenSource] = {}
    for source_path in source_paths:
        if not source_path.is_file():
            continue
        source = load_source(source_path.parent)
        sources[str(source.manifest["source_sha256"])] = source
    if not sources:
        raise FileNotFoundError(f"No source.json files found under barrage: {barrage_dir}")
    return sources


def _load_training_rows_many(barrage_dirs: tuple[Path, ...], *, contract: MarkerContract) -> pd.DataFrame:
    rows = pd.concat(
        [_load_training_rows(barrage_dir, contract=contract) for barrage_dir in barrage_dirs],
        ignore_index=True,
    )
    duplicates = rows.loc[rows["cell"].duplicated(keep=False), "cell"].astype(str)
    if not duplicates.empty:
        raise ValueError(f"Training barrages contain duplicate cells: {duplicates.iloc[0]}")
    return rows


def _load_training_sources_many(barrage_dirs: tuple[Path, ...]) -> dict[str, OpenSource]:
    sources: dict[str, OpenSource] = {}
    for barrage_dir in barrage_dirs:
        sources.update(_load_training_sources(barrage_dir))
    return sources


def _logreg_fold_scores(
    features: np.ndarray,
    labels: np.ndarray,
    jobs: list[tuple[float, np.ndarray, np.ndarray]],
) -> list[float]:
    scores = []
    # These small fits are much faster without OpenBLAS spawning a full host's threads.
    with threadpool_limits(limits=1):
        for c_value, train_indices, validation_indices in jobs:
            scaler = StandardScaler()
            train = scaler.fit_transform(features[train_indices])
            validation = scaler.transform(features[validation_indices])
            model = LogisticRegression(solver="lbfgs", C=c_value, max_iter=1_000, tol=1e-12)
            model.fit(train, labels[train_indices])
            scores.append(balanced_accuracy_score(labels[validation_indices], model.predict(validation)))
    return scores


def _best_logreg_c(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    cv_folds: int,
    seed: int,
    n_jobs: int = 1,
) -> tuple[float, float, int]:
    """Select C by mean fold balanced accuracy; `n_jobs` processes run single-threaded fits."""
    counts = np.bincount(labels.astype(np.int64))
    nonzero = counts[counts > 0]
    if len(nonzero) != 2 or int(nonzero.min()) < 2:
        raise ValueError(f"Need at least two examples of both classes, got counts={counts.tolist()}")
    folds = min(cv_folds, int(nonzero.min()))
    splits = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(features, labels))
    c_values = [float(10.0**exponent) for exponent in np.arange(-6.0, 0.25, 0.25)]
    jobs = [(c_value, train, validation) for c_value in c_values for train, validation in splits]
    # Strided batches mix cheap strongly regularized fits with slow weakly regularized ones and
    # ship the feature matrix once per batch rather than once per fit.
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_logreg_fold_scores)(features, labels, jobs[start::n_jobs]) for start in range(n_jobs)
    )
    scores = np.empty(len(jobs))
    for start, result in enumerate(results):
        scores[start::n_jobs] = result
    best_c, best_score = 1.0, -1.0
    for index, c_value in enumerate(c_values):
        score = float(np.mean(scores[index * folds : (index + 1) * folds]))
        if score > best_score:
            best_c, best_score = c_value, score
    return best_c, best_score, folds



def _feature_key(target: str, row: pd.Series) -> str:
    payload = (
        f"{target}|{row['cell']}|{int(row['label'])}|{int(row['plane_z'])}|"
        f"{row['centroid_y']}|{row['centroid_x']}"
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _cached_features(
    rows: pd.DataFrame,
    cache: dict[str, Any],
    markers: Sequence[str],
) -> tuple[dict[str, list[np.ndarray | None]], dict[int, tuple[str, ...]]]:
    entries: dict[str, Any] = cache["entries"]
    features = {target: [None] * len(rows) for target in markers}
    missing: dict[int, tuple[str, ...]] = {}
    for row_index, (_, row) in enumerate(rows.iterrows()):
        row_missing: list[str] = []
        for target in markers:
            cached = entries.get(_feature_key(target, row))
            if isinstance(cached, dict) and "feature" in cached:
                features[target][row_index] = np.asarray(cached["feature"], dtype=np.float32)
            else:
                row_missing.append(target)
        if row_missing:
            missing[row_index] = tuple(row_missing)
    return features, missing


def _stack_features(
    features: dict[str, list[np.ndarray | None]],
) -> dict[str, np.ndarray]:
    stacked: dict[str, np.ndarray] = {}
    for target, values in features.items():
        present = [feature for feature in values if feature is not None]
        if len(present) != len(values):
            raise RuntimeError(f"Feature extraction did not populate every {target} row")
        stacked[target] = np.stack(present)
    return stacked


def _extract_missing_features(
    rows: pd.DataFrame,
    *,
    cfg: TrainConfig,
    sources: dict[str, OpenSource],
    model: Any,
    transform: Any,
    device: torch.device,
    cache: dict[str, Any],
    features: dict[str, list[np.ndarray | None]],
    missing: dict[int, tuple[str, ...]],
) -> None:
    """Extract all cache misses by source and target set while preserving row order."""
    pending = rows.iloc[list(missing)].copy()
    pending["_feature_order"] = list(missing)
    pending["_missing_targets"] = list(missing.values())
    entries: dict[str, Any] = cache["entries"]
    cache_path = cfg.feature_cache or (cfg.out_dir / "features.joblib")
    bundle = {
        "crop_size": cfg.crop_size,
        "input_channels": cfg.input_channels,
        "highpass_sigma_px": cfg.highpass_sigma_px,
        "mask_erosion_px": cfg.mask_erosion_px,
    }
    for source_sha256, source_rows in pending.groupby("source_sha256", sort=False):
        source = sources[str(source_sha256)]
        needed_targets = tuple(
            target
            for target in cfg.input_channel_names_by_target
            if any(target in targets for targets in source_rows["_missing_targets"])
        )
        channel_names = tuple(
            dict.fromkeys(
                name
                for target in needed_targets
                for name in cfg.input_channel_names_by_target[target]
            )
        )
        reader = model_crop_reader(
            source,
            cache_gb=cfg.cache_gb,
            channel_names=channel_names,
        )
        try:
            for missing_targets, target_rows in source_rows.groupby(
                "_missing_targets", sort=False
            ):
                targets = tuple(missing_targets)
                target_rows = target_rows.reset_index(drop=True)
                feature_orders = target_rows["_feature_order"].to_numpy(dtype=np.int64)
                image_batches = iter_model_image_batches(
                    sort_cells_for_io(target_rows, reader),
                    bundle=bundle,
                    channel_names_by_target={
                        target: cfg.input_channel_names_by_target[target]
                        for target in targets
                    },
                    reader=reader,
                    batch_size=cfg.batch_size,
                    prefetch_workers=cfg.prefetch_workers,
                    crop_workers=cfg.crop_workers,
                    require_mask=True,
                )
                for pending_images, row_orders in iter_prefetched_batches(image_batches):
                    global_orders = feature_orders[row_orders]
                    for target in targets:
                        batch = prepare_model_batch(
                            pending_images[target],
                            transform=transform,
                            device=device,
                            dtype=feature_dtype(cfg.feature_precision),
                        )
                        output = extract_normalized_features(
                            model,
                            batch,
                            precision=cfg.feature_precision,
                        )
                        for global_order, feature in zip(global_orders, output, strict=True):
                            value = feature.astype(np.float32, copy=False)
                            row = rows.iloc[int(global_order)]
                            entries[_feature_key(target, row)] = {"feature": value}
                            features[target][int(global_order)] = value
                        _dump_joblib_atomic(cache, cache_path)
        finally:
            reader.close()


def _parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train Cell-DINO from a direct-Zarr barrage.")
    parser.add_argument("--barrage-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--target-source-dir",
        type=Path,
        default=None,
        help="Source that the resulting model bundle will be valid for. Defaults to the barrage root.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=None,
        help="Reusable embedding cache path. Defaults to <out-dir>/features.joblib.",
    )
    parser.add_argument("--hub-model", choices=sorted(MODEL_INPUT_CHANNELS), default="cell_dino_hpa_vitl16")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("~/cell_dino_vitl16_pretrain_hpa_sc-d3ab8938.pth").expanduser(),
    )
    parser.add_argument("--crop-size", type=int, default=100)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--crop-eval-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-gb", type=float, default=4.0)
    parser.add_argument("--prefetch-workers", type=int, default=8)
    parser.add_argument("--crop-workers", type=int, default=8)
    parser.add_argument("--highpass-sigma-px", type=float, default=0.0)
    parser.add_argument("--mask-erosion-px", type=int, default=0)
    parser.add_argument(
        "--feature-precision",
        choices=FEATURE_PRECISIONS,
        default="bfloat16",
        help="Precision used for frozen Cell-DINO feature extraction (default: bfloat16).",
    )
    parser.add_argument(
        "--input-channel",
        dest="input_channels",
        action="append",
        default=[],
        metavar="TARGET=CHANNEL",
        help=(
            "Source channel for the next populated model slot of a marker, in flag order. May be "
            "repeated. The default marker contract defaults each marker to brdu and edu; every marker "
            "of any other contract must be given."
        ),
    )
    parser.add_argument(
        "--markers",
        type=Path,
        default=None,
        help="Marker configuration that the barrages and target source must match.",
    )
    args = parser.parse_args()
    if (
        args.batch_size <= 0
        or args.cv_folds < 2
        or args.crop_size <= 0
        or args.cache_gb <= 0
        or args.prefetch_workers <= 0
        or args.crop_workers <= 0
    ):
        parser.error(
            "batch size, crop size, cache size, and worker counts must be positive; "
            "CV folds must be at least 2"
        )
    if not 0.0 < args.test_fraction < 1.0:
        parser.error("--test-fraction must be between 0 and 1")
    if args.highpass_sigma_px < 0:
        parser.error("--highpass-sigma-px must be non-negative")
    if args.mask_erosion_px < 0:
        parser.error("--mask-erosion-px must be non-negative")
    barrage_dirs = tuple(path.expanduser() for path in args.barrage_dir)
    target_source_dir = args.target_source_dir.expanduser() if args.target_source_dir is not None else None
    contract = _training_contract(
        barrage_dirs, target_source_dir, args.markers.expanduser() if args.markers is not None else None
    )
    return TrainConfig(
        barrage_dirs=barrage_dirs,
        target_source_dir=target_source_dir,
        out_dir=args.out_dir.expanduser(),
        feature_cache=args.feature_cache.expanduser() if args.feature_cache is not None else None,
        hub_model=args.hub_model,
        weights=args.weights.expanduser(),
        crop_size=args.crop_size,
        resize_size=args.resize_size,
        crop_eval_size=args.crop_eval_size,
        batch_size=args.batch_size,
        cv_folds=args.cv_folds,
        test_fraction=args.test_fraction,
        limit=args.limit,
        seed=args.seed,
        cache_gb=args.cache_gb,
        prefetch_workers=args.prefetch_workers,
        crop_workers=args.crop_workers,
        highpass_sigma_px=args.highpass_sigma_px,
        mask_erosion_px=args.mask_erosion_px,
        feature_precision=args.feature_precision,
        input_channel_names_by_target=resolve_input_channels(
            args.input_channels, contract, slots=MODEL_INPUT_CHANNELS[args.hub_model]
        ),
        contract=contract,
    )


def main() -> None:
    cfg = _parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    markers = cfg.contract.markers
    rows = _load_training_rows_many(cfg.barrage_dirs, contract=cfg.contract)
    if len(rows) < 8:
        raise ValueError(f"Need at least eight labeled cells, got {len(rows)}")
    rows["_source_order"] = np.arange(len(rows), dtype=np.int64)
    train_rows, validation_rows = split_training_rows(
        rows, markers=markers, limit=cfg.limit, test_fraction=cfg.test_fraction, seed=cfg.seed
    )
    selected_rows = (
        pd.concat([train_rows, validation_rows], ignore_index=True)
        .sort_values("_source_order", kind="stable")
        .reset_index(drop=True)
    )
    positions = pd.Series(
        np.arange(len(selected_rows), dtype=np.int64),
        index=selected_rows["_source_order"].to_numpy(dtype=np.int64),
    )
    train_positions = positions.loc[train_rows["_source_order"]].to_numpy(dtype=np.int64)
    validation_positions = positions.loc[validation_rows["_source_order"]].to_numpy(dtype=np.int64)
    sources = _load_training_sources_many(cfg.barrage_dirs)
    training_source_sha256s = sorted(rows["source_sha256"].astype(str).unique())
    missing_sources = sorted(set(training_source_sha256s) - set(sources))
    if missing_sources:
        raise ValueError(f"Missing source manifests for labeled rows: {missing_sources}")
    target_source = load_source(cfg.target_source_dir or cfg.barrage_dirs[0])
    source_sha256 = str(target_source.manifest["source_sha256"])

    weights_sha256 = file_sha256(cfg.weights)
    dinov2_source = installed_dinov2_source()
    cache_fields = {
        "source_sha256s": training_source_sha256s,
        "weights_sha256": weights_sha256,
        "dinov2_source": dinov2_source,
        "hub_model": cfg.hub_model,
        "crop_size": cfg.crop_size,
        "resize_size": cfg.resize_size,
        "crop_eval_size": cfg.crop_eval_size,
        "highpass_sigma_px": cfg.highpass_sigma_px,
        "mask_erosion_px": cfg.mask_erosion_px,
        "feature_precision": cfg.feature_precision,
        "feature_execution": FEATURE_EXECUTION,
        "input_channel_names_by_target": {
            target: list(channel_names)
            for target, channel_names in cfg.input_channel_names_by_target.items()
        },
    }
    cache_metadata = FEATURE_CACHE.stamp(cache_fields, cfg.contract)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cfg.feature_cache or (cfg.out_dir / "features.joblib")
    cache = joblib.load(cache_path) if cache_path.is_file() else {"metadata": cache_metadata, "entries": {}}
    metadata = cache.get("metadata")
    if (
        not isinstance(metadata, dict)
        or not FEATURE_CACHE.recognizes(metadata)
        or not isinstance(cache.get("entries"), dict)
    ):
        cache = {"metadata": cache_metadata, "entries": {}}
    else:
        recorded, expected = FEATURE_CACHE.normalize(metadata, where=str(cache_path)), cache_metadata
        if "model_source_sha256" in recorded:
            # A legacy cache predates recorded dinov2 sources and, like a legacy bundle, is reused without a source check.
            recorded = {key: value for key, value in recorded.items() if key != "model_source_sha256"}
            expected = {key: value for key, value in cache_metadata.items() if key != "dinov2_source"}
        if recorded != expected:
            cache = {"metadata": cache_metadata, "entries": {}}

    features, missing = _cached_features(selected_rows, cache, markers)
    miss_count = sum(map(len, missing.values()))
    logger.info(
        f"Feature cache rows={len(selected_rows)} "
        f"hits={len(markers) * len(selected_rows) - miss_count} misses={miss_count}"
    )
    if missing:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_cell_dino_model(
            hub_model=cfg.hub_model,
            weights=cfg.weights,
            device=device,
            precision=cfg.feature_precision,
        )
        transform = build_cell_transform(
            resize_size=cfg.resize_size,
            crop_size=cfg.crop_eval_size,
        )
        _extract_missing_features(
            selected_rows,
            cfg=cfg,
            sources=sources,
            model=model,
            transform=transform,
            device=device,
            cache=cache,
            features=features,
            missing=missing,
        )
    all_features = _stack_features(features)
    targets = train_rows[list(markers)].to_numpy(dtype=np.int64)
    validation_targets = validation_rows[list(markers)].to_numpy(dtype=np.int64)
    models: dict[str, LogisticRegression] = {}
    scalers: dict[str, StandardScaler] = {}
    metrics: dict[str, float | int] = {}
    for target_index, target in enumerate(markers):
        train_features = all_features[target][train_positions]
        validation_features = all_features[target][validation_positions]
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_features)
        validation_scaled = scaler.transform(validation_features)
        c_value, cv_score, folds = _best_logreg_c(
            train_features,
            targets[:, target_index],
            cv_folds=cfg.cv_folds,
            seed=cfg.seed,
        )
        with threadpool_limits(limits=1):
            classifier = LogisticRegression(solver="lbfgs", C=c_value, max_iter=1_000, tol=1e-12)
            classifier.fit(train_scaled, targets[:, target_index])
        probabilities = classifier.predict_proba(validation_scaled)[:, 1]
        cutoff, cutoff_score = balanced_accuracy_cutoff(validation_targets[:, target_index], probabilities)
        models[target] = classifier
        scalers[target] = scaler
        metrics.update(
            {
                metric_key(target, "best_c"): c_value,
                metric_key(target, "cv_balanced_accuracy"): cv_score,
                metric_key(target, "cv_folds"): folds,
                cutoff_key(target): cutoff,
                metric_key(target, "validation_balanced_accuracy"): cutoff_score,
            }
        )
    bundle_fields = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_sha256,
        "training_source_sha256s": training_source_sha256s,
        "training_barrage_dirs": [str(path.resolve()) for path in cfg.barrage_dirs],
        "feature_extractor": cfg.hub_model,
        "input_channels": cfg.input_channels,
        "input_channel_names_by_target": {
            target: list(channel_names)
            for target, channel_names in cfg.input_channel_names_by_target.items()
        },
        "weights": str(cfg.weights),
        "weights_sha256": weights_sha256,
        "dinov2_source": dinov2_source,
        "crop_size": cfg.crop_size,
        "resize_size": cfg.resize_size,
        "crop_eval_size": cfg.crop_eval_size,
        "highpass_sigma_px": cfg.highpass_sigma_px,
        "mask_erosion_px": cfg.mask_erosion_px,
        "feature_precision": cfg.feature_precision,
        "feature_execution": FEATURE_EXECUTION,
        **{scaler_key(target): scalers[target] for target in markers},
        **{model_key(target): models[target] for target in markers},
        "train_cells": train_rows["cell"].astype(str).tolist(),
        "validation_cells": validation_rows["cell"].astype(str).tolist(),
        "split_seed": cfg.seed,
        "test_fraction": cfg.test_fraction,
        **metrics,
    }
    bundle = ZARR_MODEL.stamp(bundle_fields, cfg.contract)
    model_path = cfg.out_dir / "model.joblib"
    _dump_joblib_atomic(bundle, model_path)
    logger.info(f"Wrote direct-Zarr Cell-DINO bundle: {model_path}")
    print(f"model_path={model_path}")
    print(f"train_rows={len(train_rows)}")
    print(f"validation_rows={len(validation_rows)}")
    print(f"feature_precision={cfg.feature_precision}")
    print(f"feature_cache={cache_path}")
    for target in markers:
        print(f"{cutoff_key(target)}={bundle[cutoff_key(target)]:.8g}")


if __name__ == "__main__":
    main()
