from __future__ import annotations

import importlib
import inspect
import os
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from dino_markers import bundle_contract
from loguru import logger
from sklearn.metrics import balanced_accuracy_score

from dino.cell_dino import build_cell_transform as build_cell_transform
from dino.pixel_corrections import correct_channels
from dino.zarr_source import (
    CellCropReader,
    OpenSource,
    cell_dino_images,
    cell_dino_mask,
)


MODEL_INPUT_CHANNELS = {
    "cell_dino_cp_vits8": 5,
    "cell_dino_hpa_vitl16": 4,
}
DEFAULT_INPUT_CHANNEL_NAMES = ("brdu", "edu")
FEATURE_PRECISIONS = ("float32", "bfloat16")
FEATURE_EXECUTION = "native_sdpa"


def balanced_accuracy_cutoff(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Choose a probability cutoff, with deterministic ties around 0.5."""

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape or labels.ndim != 1:
        raise ValueError("labels and probabilities must be one-dimensional arrays of equal length")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Both binary classes are required to choose a probability cutoff")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probability cutoffs require finite probabilities")

    candidates = np.unique(probabilities)
    scored = [
        (float(balanced_accuracy_score(labels, probabilities >= cutoff)), float(cutoff))
        for cutoff in candidates
    ]
    best_score = max(score for score, _ in scored)
    cutoff = min(
        (cutoff for score, cutoff in scored if np.isclose(score, best_score)),
        key=lambda value: (abs(value - 0.5), -value),
    )
    return cutoff, best_score


def feature_dtype(precision: str) -> torch.dtype:
    if precision == "float32":
        return torch.float32
    if precision == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported feature precision: {precision}")


def cell_dino_backbone(hub_model: str) -> Any:
    """Resolve a Cell-DINO backbone constructor in the installed dinov2 package."""
    backbones = importlib.import_module("dinov2.hub.cell_dino.backbones")
    available = {
        name: value for name, value in vars(backbones).items() if not name.startswith("_") and inspect.isfunction(value)
    }
    if hub_model not in available:
        raise ValueError(
            f"{hub_model!r} is not a Cell-DINO backbone in {backbones.__name__}; available: {sorted(available)}"
        )
    return available[hub_model]


def load_cell_dino_model(
    *,
    hub_model: str,
    weights: Path,
    device: torch.device,
    precision: str,
) -> Any:
    dtype = feature_dtype(precision)
    if dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support bfloat16")
    attention_module = sys.modules.get("dinov2.layers.attention")
    if attention_module is not None and bool(getattr(attention_module, "XFORMERS_AVAILABLE", False)):
        raise RuntimeError(
            "DINOv2 attention was imported with xFormers enabled; start a fresh process so native "
            "PyTorch SDPA can be selected"
        )
    os.environ["XFORMERS_DISABLED"] = "1"
    model = cell_dino_backbone(hub_model)(pretrained_path=str(weights))
    attention_module = sys.modules.get("dinov2.layers.attention")
    if attention_module is None or bool(getattr(attention_module, "XFORMERS_AVAILABLE", True)):
        raise RuntimeError("Cell-DINO did not initialize with native PyTorch SDPA")
    return model.eval().to(device=device, dtype=dtype)


def extract_normalized_features(
    model: Any,
    batch: torch.Tensor,
    *,
    precision: str,
) -> np.ndarray:
    """Run the frozen encoder at the requested precision and return float32 features."""

    return extract_normalized_feature_tensor(model, batch, precision=precision).float().numpy()


def extract_normalized_feature_tensor(
    model: Any,
    batch: torch.Tensor,
    *,
    precision: str,
) -> torch.Tensor:
    """Return normalized CPU features without changing the encoder's native dtype."""

    expected_dtype = feature_dtype(precision)
    if batch.dtype != expected_dtype:
        raise ValueError(f"Expected {precision} model input, got batch dtype {batch.dtype}")
    with torch.inference_mode():
        output = torch.nn.functional.normalize(model(batch), dim=1, p=2)
    if output.dtype != expected_dtype:
        raise ValueError(f"Expected {precision} model output, got feature dtype {output.dtype}")
    return output.cpu()


def prepare_model_batch(
    images: list[np.ndarray],
    *,
    transform: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Preprocess in float32 on-device, then cast once to the model dtype."""
    batch = torch.from_numpy(np.stack(images)).to(
        device=device,
        dtype=torch.float32,
    )
    return transform(batch).to(dtype=dtype)


def model_crop_reader(
    source: OpenSource,
    *,
    cache_gb: float,
    channel_names: tuple[str, ...] = DEFAULT_INPUT_CHANNEL_NAMES,
    spatial_subtraction: Mapping[str, Any] | None = None,
) -> CellCropReader:
    if spatial_subtraction is not None:
        from dino.spatial_pixels import SpatialPixels

        subtraction = SpatialPixels.from_spec(spatial_subtraction)
        channel_names = tuple(dict.fromkeys((*channel_names, subtraction.reference)))
    missing = sorted(set(channel_names) - set(source.channels))
    if missing:
        raise ValueError(f"Source is missing model input channels: {missing}")
    return CellCropReader(
        labels=source.labels,
        channels={name: source.channels[name] for name in channel_names},
        offset_zyx=source.offset_zyx,
        cache_bytes=int(cache_gb * 1024**3),
    )


def read_model_images(
    reader: CellCropReader,
    row: pd.Series,
    *,
    crop_size: int,
    input_channels: int,
    channel_names_by_target: Mapping[str, tuple[str, ...]],
    highpass_sigma_px: float = 0.0,
    mask_erosion_px: int = 0,
    pixel_correction=None,
    spatial_subtraction=None,
) -> tuple[dict[str, np.ndarray], bool]:
    """Build target-specific model tensors from one source crop."""
    crop = reader.read_cell(
        label=int(row["label"]),
        plane_z=int(row["plane_z"]),
        center_y=int(np.rint(float(row["centroid_y"]))),
        center_x=int(np.rint(float(row["centroid_x"]))),
        size=crop_size,
    )
    if pixel_correction is not None or spatial_subtraction is not None:
        from dataclasses import replace

        origin = (
            int(row["plane_z"]) + reader.offset_zyx[0],
            int(np.rint(float(row["centroid_y"]))) + reader.offset_zyx[1] - crop_size // 2,
            int(np.rint(float(row["centroid_x"]))) + reader.offset_zyx[2] - crop_size // 2,
        )
        channels = correct_channels(
            crop.channels,
            origin,
            reference=spatial_subtraction.reference if spatial_subtraction is not None else None,
            spatial_subtraction=spatial_subtraction,
            pixel_correction=pixel_correction,
        )
        crop = replace(crop, channels=channels)
    return (
        cell_dino_images(
            crop,
            input_channels=input_channels,
            channel_names_by_target=channel_names_by_target,
            highpass_sigma_px=highpass_sigma_px,
            mask_erosion_px=mask_erosion_px,
        ),
        bool(cell_dino_mask(crop, mask_erosion_px=mask_erosion_px).any()),
    )


def sort_cells_for_io(cells: pd.DataFrame, reader: CellCropReader) -> pd.DataFrame:
    """Order cells by every source array's chunk coordinates while retaining row order."""
    ordered = cells.copy()
    ordered["_row_order"] = np.arange(len(ordered), dtype=np.int64)
    plane_z = ordered["plane_z"].to_numpy(dtype=np.int64)
    center_y = np.rint(ordered["centroid_y"]).to_numpy(dtype=np.int64)
    center_x = np.rint(ordered["centroid_x"]).to_numpy(dtype=np.int64)
    sort_columns: list[str] = []
    arrays = [("labels", reader.labels, (0, 0, 0))]
    arrays.extend((name, array, reader.offset_zyx) for name, array in reader.channels.items())
    for index, (_, array, offset) in enumerate(arrays):
        offset_z, offset_y, offset_x = offset
        for axis, values, axis_offset, chunk in zip(
            "zyx",
            (plane_z, center_y, center_x),
            (offset_z, offset_y, offset_x),
            array.chunks,
            strict=True,
        ):
            column = f"_io_{index}_{axis}"
            ordered[column] = (values + axis_offset) // int(chunk)
            sort_columns.append(column)
    return ordered.sort_values([*sort_columns, "cell"], kind="stable").drop(columns=sort_columns)


def iter_model_image_batches(
    ordered: pd.DataFrame,
    *,
    bundle: Mapping[str, Any],
    channel_names_by_target: Mapping[str, tuple[str, ...]],
    reader: CellCropReader,
    batch_size: int,
    prefetch_workers: int,
    crop_workers: int,
    require_mask: bool,
) -> Iterator[tuple[dict[str, list[np.ndarray]], list[int]]]:
    """Read and preprocess spatially ordered crops in bounded batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    targets = tuple(channel_names_by_target)
    if not targets:
        raise ValueError("At least one model target is required")
    pending_images: dict[str, list[np.ndarray]] = {target: [] for target in targets}
    pending_row_orders: list[int] = []
    crop_size = int(bundle["crop_size"])
    input_channels = int(bundle["input_channels"])
    highpass_sigma_px = float(bundle["highpass_sigma_px"])
    mask_erosion_px = int(bundle["mask_erosion_px"])

    pixel_correction = None
    spatial_subtraction = None
    if "spatial_subtraction" in bundle or "pixel_correction" in bundle:
        contract = bundle_contract(bundle, where="model bundle")
    if "spatial_subtraction" in bundle:
        from dino.spatial_pixels import SpatialPixels, require_spatial_subtraction_contract

        require_spatial_subtraction_contract(contract, artifact="model bundle")
        spatial_subtraction = SpatialPixels.from_spec(bundle["spatial_subtraction"])
        if spatial_subtraction.reference not in reader.channels:
            raise ValueError(
                f"Spatial subtraction crop reader requires raw {spatial_subtraction.reference} (405 nm DAPI)"
            )
    if "pixel_correction" in bundle:
        from dino.periodic_pixels import PeriodicPixels, require_periodic_targets

        correction = bundle["pixel_correction"]
        pixel_correction = PeriodicPixels(
            correction["model"],
            correction["model_sha256"],
            tuple(correction["scale_zyx_um"]),
        )
        require_periodic_targets(pixel_correction, contract, artifact="model bundle")

    def prepare_row(row: pd.Series) -> tuple[dict[str, np.ndarray], bool]:
        return read_model_images(
            reader,
            row,
            crop_size=crop_size,
            input_channels=input_channels,
            channel_names_by_target=channel_names_by_target,
            highpass_sigma_px=highpass_sigma_px,
            mask_erosion_px=mask_erosion_px,
            pixel_correction=pixel_correction,
            spatial_subtraction=spatial_subtraction,
        )

    with ThreadPoolExecutor(
        max_workers=crop_workers,
        thread_name_prefix="dino-preprocess",
    ) as crop_executor:
        start = 0
        while start < len(ordered):
            stop = min(start + batch_size, len(ordered))
            while True:
                planned = ordered.iloc[start:stop]
                plan = reader.plan_cells(
                    [
                        (
                            int(row.plane_z),
                            int(np.rint(row.centroid_y)),
                            int(np.rint(row.centroid_x)),
                        )
                        for row in planned.itertuples()
                    ],
                    size=crop_size,
                )
                if reader.plan_fits(plan) or len(planned) == 1:
                    break
                stop = start + max(1, len(planned) // 2)
            reader.prefetch_plan(plan, workers=prefetch_workers)
            rows = [row for _, row in planned.iterrows()]
            for row, (images, has_mask) in zip(
                rows,
                crop_executor.map(prepare_row, rows),
                strict=True,
            ):
                if not has_mask:
                    if require_mask:
                        raise ValueError(f"Target cell mask is empty for {row['cell']}")
                    continue
                for target in targets:
                    pending_images[target].append(images[target])
                pending_row_orders.append(int(row["_row_order"]))
                if len(pending_row_orders) >= batch_size:
                    yield pending_images, pending_row_orders
                    pending_images = {target: [] for target in targets}
                    pending_row_orders = []
            if stop % 10_000 == 0:
                logger.info(f"Prepared model cells={stop}/{len(ordered)}")
            start = stop
    if pending_row_orders:
        yield pending_images, pending_row_orders


def iter_prefetched_batches(
    batches: Iterator[tuple[dict[str, list[np.ndarray]], list[int]]],
) -> Iterator[tuple[dict[str, list[np.ndarray]], list[int]]]:
    """Prepare the next CPU batch while the caller consumes the current batch."""
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="dino-crops") as executor:
        future = executor.submit(next, batches, None)
        while (batch := future.result()) is not None:
            future = executor.submit(next, batches, None)
            yield batch
