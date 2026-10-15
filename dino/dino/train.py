from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import zarr
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from dino.artifacts import _dump_joblib_atomic, array_sha256, file_sha256, path_sha256
from dino.artifacts import row_feature_cache_key as _row_feature_cache_key
from dino.training_split import split_training_rows
from dino_markers import DEFAULT_CONTRACT
from dino.workspace_inputs import (
    centered_segmentation_labels,
    cell_id_roi_and_label,
    resolve_workspace_root,
)
from dino.cell_dino import build_cell_transform, require_dinov2_checkout
from fishtools.brdu.model import load_barrage_training_table
from fishtools.io.workspace import Workspace
from fishtools.segment.cell_thumbnail import cell_thumbnail_from_fused


FUSED_ZARR_NAME = "fused.zarr"
C_POWER_RANGE = np.linspace(-6.0, 5.0, 45)
UINT16_TO_UINT8_SCALE = 256.0
FEATURE_CACHE_VERSION = "v3_cell_dino_content_bound"

MODEL_INPUT_CHANNELS = {
    "cell_dino_cp_vits8": 5,
    "cell_dino_hpa_vitl16": 4,
}


@dataclass(frozen=True)
class TrainConfig:
    barrage_dir: Path
    model_out: Path
    feature_cache: Path
    dinov2_repo: Path
    hub_model: str
    input_channels: int
    weights: Path
    codebook: str
    seg_codebook: str
    segmentation_name: str
    crop_size: int
    resize_size: int
    crop_eval_size: int
    batch_size: int
    num_workers: int
    cv_folds: int
    test_fraction: float
    limit: int
    seed: int


def _parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train frozen Cell-DINO logistic regressions on labeled BrdU/EdU barrage cells."
    )
    parser.add_argument("--barrage-dir", type=Path, default=Path("output/brdu_edu_barrage"))
    parser.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help="Output joblib path. Default: <barrage-dir>/brdu_edu_cell_dino_hpa_vitl16_logreg.joblib",
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=None,
        help="Output joblib path for cached Cell-DINO features. Default: <barrage-dir>/cell_dino_hpa_vitl16_feature_cache.joblib",
    )
    parser.add_argument("--dinov2-repo", type=Path, default=Path("~/dinov2").expanduser())
    parser.add_argument(
        "--hub-model",
        choices=sorted(MODEL_INPUT_CHANNELS),
        default="cell_dino_hpa_vitl16",
        help="Local torch.hub Cell-DINO model entrypoint.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("~/cell_dino_vitl16_pretrain_hpa_sc-d3ab8938.pth").expanduser(),
    )
    parser.add_argument("--codebook", default="edu")
    parser.add_argument("--seg-codebook", default="pi")
    parser.add_argument("--segmentation-name", default="output_segmentation-sam_postproc_s1-2-2_v500.zarr")
    parser.add_argument(
        "--crop-size", type=int, default=100, help="Centered crop size used in barrage generation."
    )
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--crop-eval-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--limit", type=int, default=0, help="Optional cap on labeled rows for quick experiments."
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    barrage_dir = Path(args.barrage_dir)
    hub_model = str(args.hub_model)
    model_out = (
        Path(args.model_out)
        if args.model_out is not None
        else (barrage_dir / f"brdu_edu_{hub_model}_logreg.joblib")
    )
    feature_cache = (
        Path(args.feature_cache)
        if args.feature_cache is not None
        else (barrage_dir / f"{hub_model}_feature_cache.joblib")
    )
    return TrainConfig(
        barrage_dir=barrage_dir,
        model_out=model_out,
        feature_cache=feature_cache,
        dinov2_repo=Path(args.dinov2_repo).expanduser(),
        hub_model=hub_model,
        input_channels=MODEL_INPUT_CHANNELS[hub_model],
        weights=Path(args.weights).expanduser(),
        codebook=str(args.codebook),
        seg_codebook=str(args.seg_codebook),
        segmentation_name=str(args.segmentation_name),
        crop_size=int(args.crop_size),
        resize_size=int(args.resize_size),
        crop_eval_size=int(args.crop_eval_size),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        cv_folds=int(args.cv_folds),
        test_fraction=float(args.test_fraction),
        limit=int(args.limit),
        seed=int(args.seed),
    )


def _compose_cell_dino_channels(masked_thumb: np.ndarray, *, mask: np.ndarray) -> np.ndarray:
    """Map masked BrdU/EdU into uint8 and zero-pad to the selected Cell-DINO input width."""

    if masked_thumb.ndim != 3:
        raise ValueError(f"Expected masked_thumb with shape (H, W, C), got {masked_thumb.shape}")
    if masked_thumb.shape[-1] != 2:
        raise ValueError(f"Expected exactly 2 channels (brdu, edu), got {masked_thumb.shape[-1]}")

    output = np.zeros((*masked_thumb.shape[:2], 5), dtype=np.float32)
    for channel_index in range(masked_thumb.shape[-1]):
        channel = masked_thumb[..., channel_index].astype(np.float32)
        scaled = channel / UINT16_TO_UINT8_SCALE
        scaled[~mask] = 0.0
        output[..., channel_index] = scaled

    return output.astype(np.uint8)


def _feature_cache_metadata(
    cfg: TrainConfig,
    *,
    weights_sha256: str,
    model_source_sha256: str,
) -> dict[str, Any]:
    return {
        "feature_cache_version": FEATURE_CACHE_VERSION,
        "feature_extractor": cfg.hub_model,
        "input_channels": cfg.input_channels,
        "weights_sha256": weights_sha256,
        "model_source_sha256": model_source_sha256,
        "codebook": cfg.codebook,
        "seg_codebook": cfg.seg_codebook,
        "segmentation_name": cfg.segmentation_name,
        "crop_size": cfg.crop_size,
        "resize_size": cfg.resize_size,
        "crop_eval_size": cfg.crop_eval_size,
        "uint16_to_uint8_scale": UINT16_TO_UINT8_SCALE,
    }


def _load_feature_cache(cfg: TrainConfig, *, expected_metadata: dict[str, Any]) -> dict[str, Any]:
    if not cfg.feature_cache.exists():
        return {"metadata": expected_metadata, "entries": {}}

    bundle = joblib.load(cfg.feature_cache)
    if bundle.get("metadata") != expected_metadata:
        logger.info(f"Ignoring incompatible feature cache at {cfg.feature_cache}")
        return {"metadata": expected_metadata, "entries": {}}

    entries = bundle.get("entries")
    if not isinstance(entries, dict):
        raise ValueError(f"Expected dict feature cache entries at {cfg.feature_cache}")
    return {"metadata": expected_metadata, "entries": entries}


class BarrageCellDataset(Dataset[tuple[torch.Tensor, str]]):
    """Load barrage-selected cell crops and format them for Cell-DINO ViT-S/8."""

    def __init__(self, rows: pd.DataFrame, *, cfg: TrainConfig) -> None:
        self.rows = rows.reset_index(drop=True).copy()
        self.cfg = cfg
        self.workspace_cache: dict[str, Workspace] = {}
        self.fused_cache: dict[tuple[str, str, str], zarr.Array] = {}
        self.seg_cache: dict[tuple[str, str, str, str], zarr.Array] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        row = self.rows.iloc[index]
        image = self._load_masked_cell_image(row)
        tensor = torch.from_numpy(np.moveaxis(image, -1, 0))
        return tensor, array_sha256(image)

    def _workspace_for_dataset(self, dataset: str) -> Workspace:
        ws = self.workspace_cache.get(dataset)
        if ws is None:
            workspace_root = resolve_workspace_root(dataset)
            ws = Workspace(workspace_root)
            self.workspace_cache[dataset] = ws
        return ws

    def _load_masked_cell_image(self, row: pd.Series) -> np.ndarray:
        dataset = str(row["dataset"])
        roi = str(row["roi"])
        cell = str(row["cell"])
        x_center = int(np.rint(float(row["x"])))
        y_center = int(np.rint(float(row["y"])))
        z_index = int(np.rint(float(row["z"])))

        roi_from_id, label = cell_id_roi_and_label(cell)
        if roi_from_id != roi:
            raise ValueError(f"Cell id roi {roi_from_id!r} != row roi {roi!r}")

        ws = self._workspace_for_dataset(dataset)
        fused_key = (dataset, roi, self.cfg.codebook)
        fused = self.fused_cache.get(fused_key)
        if fused is None:
            fused = zarr.open_array(ws.stitch(roi, self.cfg.codebook) / FUSED_ZARR_NAME, mode="r")
            self.fused_cache[fused_key] = fused

        channel_names_raw = fused.attrs.get("key")
        channel_names = list(channel_names_raw) if isinstance(channel_names_raw, list) else None
        if channel_names is None:
            raise ValueError(f"Missing fused channel names in attrs for dataset={dataset} roi={roi}")
        required_channels = ["brdu", "edu"]
        channel_indices = []
        for channel_name in required_channels:
            if channel_name not in channel_names:
                raise ValueError(
                    f"Required channel {channel_name!r} not found in fused attrs keys={channel_names}"
                )
            channel_indices.append(channel_names.index(channel_name))

        thumb = cell_thumbnail_from_fused(
            fused,
            z_index=z_index,
            x_center=x_center,
            y_center=y_center,
            size=self.cfg.crop_size,
            channels=channel_indices,
        )

        seg_key = (dataset, roi, self.cfg.seg_codebook, self.cfg.segmentation_name)
        seg = self.seg_cache.get(seg_key)
        if seg is None:
            seg = zarr.open_array(
                ws.stitch(roi, self.cfg.seg_codebook) / self.cfg.segmentation_name, mode="r"
            )
            self.seg_cache[seg_key] = seg

        mask = centered_segmentation_labels(
            seg,
            z_index=z_index,
            x_center=x_center,
            y_center=y_center,
            size=self.cfg.crop_size,
        ) == label
        if not mask.any():
            raise ValueError(f"Target cell mask is empty for {cell}")

        masked_thumb = thumb.copy()
        masked_thumb[~mask] = 0
        image = _compose_cell_dino_channels(masked_thumb, mask=mask)
        return image[..., : self.cfg.input_channels]


def _extract_features_with_cache(
    *,
    model: Any,
    rows: pd.DataFrame,
    cfg: TrainConfig,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    cache_bundle: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, bool]:
    transform = build_cell_transform(
        repo=cfg.dinov2_repo,
        resize_size=cfg.resize_size,
        crop_size=cfg.crop_eval_size,
    )
    entries = cache_bundle["entries"]
    cache_keys = [_row_feature_cache_key(row) for _, row in rows.iterrows()]
    loader = DataLoader(
        BarrageCellDataset(rows, cfg=cfg),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    features: list[np.ndarray] = []
    cache_updated = False
    hits = 0
    row_offset = 0
    for images, input_digests in loader:
        batch_keys = cache_keys[row_offset : row_offset + len(input_digests)]
        missing = [
            index
            for index, (cache_key, input_digest) in enumerate(zip(batch_keys, input_digests, strict=True))
            if cache_key not in entries or entries[cache_key].get("input_sha256") != input_digest
        ]
        if missing:
            missing_images = transform(images[missing].float()).to(device, non_blocking=device.type == "cuda")
            with torch.inference_mode():
                missing_features = (
                    torch.nn.functional.normalize(model(missing_images), dim=1, p=2).cpu().numpy()
                )
            for index, feature in zip(missing, missing_features, strict=True):
                entries[batch_keys[index]] = {
                    "feature": feature.astype(np.float32, copy=False),
                    "input_sha256": input_digests[index],
                }
            cache_updated = True
        hits += len(batch_keys) - len(missing)
        features.extend(entries[cache_key]["feature"] for cache_key in batch_keys)
        row_offset += len(batch_keys)

    logger.info(f"Feature cache at {cfg.feature_cache}: hits={hits}, misses={len(rows) - hits}")
    targets = rows.loc[:, ["brdu", "edu"]].to_numpy(dtype=np.int64, copy=True)
    return np.stack(features, axis=0), targets, cache_updated


def _best_logreg_for_target_cv(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    cv_folds: int,
) -> tuple[float, float, int]:
    class_counts = np.bincount(labels.astype(np.int64))
    positive_counts = class_counts[class_counts > 0]
    if len(positive_counts) < 2:
        raise ValueError("Need both classes present to run stratified cross-validation.")

    n_splits = min(cv_folds, int(positive_counts.min()))
    if n_splits < 2:
        raise ValueError(
            f"Need at least 2 rows in each class for stratified CV, got counts={class_counts.tolist()}"
        )

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best_c = 1.0
    best_accuracy = -1.0

    for exponent in C_POWER_RANGE:
        c_value = float(10.0**exponent)
        fold_accuracies: list[float] = []
        for fold_train_idx, fold_val_idx in splitter.split(features, labels):
            scaler = StandardScaler()
            fold_train = scaler.fit_transform(features[fold_train_idx])
            fold_val = scaler.transform(features[fold_val_idx])
            model = LogisticRegression(solver="lbfgs", C=c_value, max_iter=1_000, tol=1e-12)
            model.fit(fold_train, labels[fold_train_idx])
            fold_accuracy = accuracy_score(labels[fold_val_idx], model.predict(fold_val))
            fold_accuracies.append(float(fold_accuracy))
        mean_accuracy = float(np.mean(fold_accuracies))
        if mean_accuracy > best_accuracy:
            best_c = c_value
            best_accuracy = mean_accuracy

    return best_c, best_accuracy, n_splits


def _make_split_tables(train: pd.DataFrame, *, cfg: TrainConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    return split_training_rows(train, markers=DEFAULT_CONTRACT.markers, limit=cfg.limit, test_fraction=cfg.test_fraction, seed=cfg.seed)


def _load_cell_dino_model(*, repo: Path, hub_model: str, weights: Path, device: torch.device) -> Any:
    require_dinov2_checkout(repo)
    model = torch.hub.load(str(repo), hub_model, source="local", pretrained_path=str(weights))
    return model.eval().to(device)


def main() -> None:
    cfg = _parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_table = load_barrage_training_table(barrage_dir=cfg.barrage_dir)
    train_table = train_table.drop_duplicates(subset=["cell"], keep="last").reset_index(drop=True)
    if len(train_table) < 8:
        raise ValueError(f"Need more labeled rows to fit/train split, got {len(train_table)}")

    train_rows, test_rows = _make_split_tables(train_table, cfg=cfg)
    logger.info(
        f"Using labeled barrage rows: train={len(train_rows)} (CV), test={len(test_rows)}, cv_folds={cfg.cv_folds}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_sha256 = file_sha256(cfg.weights)
    model_source_sha256 = path_sha256(cfg.dinov2_repo)
    cache_metadata = _feature_cache_metadata(
        cfg,
        weights_sha256=weights_sha256,
        model_source_sha256=model_source_sha256,
    )
    cache_bundle = _load_feature_cache(cfg, expected_metadata=cache_metadata)
    model = _load_cell_dino_model(
        repo=cfg.dinov2_repo,
        hub_model=cfg.hub_model,
        weights=cfg.weights,
        device=device,
    )

    train_features, train_targets, train_cache_updated = _extract_features_with_cache(
        model=model,
        rows=train_rows,
        cfg=cfg,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=device,
        cache_bundle=cache_bundle,
    )
    test_features, test_targets, test_cache_updated = _extract_features_with_cache(
        model=model,
        rows=test_rows,
        cfg=cfg,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=device,
        cache_bundle=cache_bundle,
    )
    if train_cache_updated or test_cache_updated:
        _dump_joblib_atomic(cache_bundle, cfg.feature_cache)

    brdu_c, brdu_cv_accuracy, brdu_cv_folds = _best_logreg_for_target_cv(
        features=train_features,
        labels=train_targets[:, 0],
        seed=cfg.seed,
        cv_folds=cfg.cv_folds,
    )
    edu_c, edu_cv_accuracy, edu_cv_folds = _best_logreg_for_target_cv(
        features=train_features,
        labels=train_targets[:, 1],
        seed=cfg.seed,
        cv_folds=cfg.cv_folds,
    )

    final_scaler = StandardScaler()
    final_train_scaled = final_scaler.fit_transform(train_features)
    final_test_scaled = final_scaler.transform(test_features)

    brdu_final = LogisticRegression(solver="lbfgs", C=brdu_c, max_iter=1_000, tol=1e-12)
    brdu_final.fit(final_train_scaled, train_targets[:, 0])
    edu_final = LogisticRegression(solver="lbfgs", C=edu_c, max_iter=1_000, tol=1e-12)
    edu_final.fit(final_train_scaled, train_targets[:, 1])

    brdu_test_accuracy = accuracy_score(test_targets[:, 0], brdu_final.predict(final_test_scaled))
    edu_test_accuracy = accuracy_score(test_targets[:, 1], edu_final.predict(final_test_scaled))

    bundle = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_extractor": cfg.hub_model,
        "weights": str(cfg.weights),
        "weights_sha256": weights_sha256,
        "dinov2_repo": str(cfg.dinov2_repo),
        "model_source_sha256": model_source_sha256,
        "input_channels": int(cfg.input_channels),
        "feature_cache": str(cfg.feature_cache),
        "codebook": cfg.codebook,
        "seg_codebook": cfg.seg_codebook,
        "segmentation_name": cfg.segmentation_name,
        "crop_size": cfg.crop_size,
        "resize_size": cfg.resize_size,
        "crop_eval_size": cfg.crop_eval_size,
        "feature_dim": int(final_train_scaled.shape[1]),
        "feature_scaler": final_scaler,
        "cv_folds": int(cfg.cv_folds),
        "brdu_cv_folds": int(brdu_cv_folds),
        "edu_cv_folds": int(edu_cv_folds),
        "train_rows": int(len(train_rows)),
        "val_rows": 0,
        "test_rows": int(len(test_rows)),
        "train_cells": train_rows["cell"].astype(str).tolist(),
        "test_cells": test_rows["cell"].astype(str).tolist(),
        "split_seed": int(cfg.seed),
        "test_fraction": float(cfg.test_fraction),
        "brdu_best_c": float(brdu_c),
        "edu_best_c": float(edu_c),
        "brdu_model": brdu_final,
        "edu_model": edu_final,
    }
    _dump_joblib_atomic(bundle, cfg.model_out)

    print(f"device={device}")
    print(f"model_out={cfg.model_out}")
    print(f"feature_cache={cfg.feature_cache}")
    print(f"train_rows={len(train_rows)}")
    print(f"cv_folds={cfg.cv_folds}")
    print("val_rows=0")
    print(f"test_rows={len(test_rows)}")
    print(f"feature_dim={final_train_scaled.shape[1]}")
    print(f"brdu_best_c={brdu_c:.10g}")
    print(f"edu_best_c={edu_c:.10g}")
    print(f"brdu_cv_accuracy={brdu_cv_accuracy:.6f}")
    print(f"edu_cv_accuracy={edu_cv_accuracy:.6f}")
    print(f"brdu_test_accuracy={brdu_test_accuracy:.6f}")
    print(f"edu_test_accuracy={edu_test_accuracy:.6f}")


if __name__ == "__main__":
    main()
