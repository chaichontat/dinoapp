from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr


WORKSPACE_ROOT_BASES = (Path("/working"), Path("/working/chaichontat"), Path.home() / "nvme")


def resolve_workspace_root(dataset: str) -> Path:
    for base in WORKSPACE_ROOT_BASES:
        candidate = base / dataset
        if candidate.exists():
            return candidate
    checked = ", ".join(str(base / dataset) for base in WORKSPACE_ROOT_BASES)
    raise FileNotFoundError(f"workspace_root not found for dataset {dataset!r}; checked: {checked}")


def cell_id_roi_and_label(cell: str) -> tuple[str, int]:
    if "|" not in cell:
        raise ValueError(f"Expected cell id '<roi>|<label>' or '<dataset>:<roi>|<label>', got {cell!r}")
    roi_token, label_str = cell.split("|", 1)
    roi = roi_token.rsplit(":", 1)[-1]
    return roi, int(label_str)


def centered_segmentation_labels(
    seg: zarr.Array,
    *,
    z_index: int,
    x_center: int,
    y_center: int,
    size: int,
) -> np.ndarray:
    half = size // 2
    y0 = y_center - half
    x0 = x_center - half

    ys0 = max(0, y0)
    ys1 = min(int(seg.shape[1]), y0 + size)
    xs0 = max(0, x0)
    xs1 = min(int(seg.shape[2]), x0 + size)

    if ys0 < ys1 and xs0 < xs1:
        if seg.ndim == 3:
            crop = np.asarray(seg[z_index, ys0:ys1, xs0:xs1])
        elif seg.ndim == 4:
            crop = np.asarray(seg[z_index, ys0:ys1, xs0:xs1, 0])
        else:
            raise ValueError(f"Unexpected segmentation array ndim={seg.ndim} shape={seg.shape}")
    else:
        crop = np.zeros((0, 0), dtype=np.int32)

    labels_full = np.zeros((size, size), dtype=crop.dtype if crop.size else np.int32)
    if crop.size:
        labels_full[(ys0 - y0) : (ys1 - y0), (xs0 - x0) : (xs1 - x0)] = crop
    return labels_full
