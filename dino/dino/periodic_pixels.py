"""Lazy periodic intensity correction with bounded XY tile reuse across Z."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock

import joblib
import numpy as np

from dino.artifacts import file_sha256

_TILE_SIZE = 960
_TILE_LOCK = Lock()


@lru_cache(maxsize=4)
def _load_field(path: str, sha256: str):
    if file_sha256(Path(path)) != sha256:
        raise ValueError(f"Periodic pixel model hash mismatch: {path}")
    bundle = joblib.load(path)
    if bundle["targets"] != ["brdu", "edu"]:
        raise ValueError("Periodic pixel targets must be brdu, edu")
    return bundle


@lru_cache(maxsize=16)
def _gain_tile(path: str, sha256: str, scale_yx: tuple[float, float], ty: int, tx: int):
    """Evaluate the tensor spline by axis, avoiding a pixels × spline-basis array."""
    bundle = _load_field(path, sha256)
    x = (np.arange(_TILE_SIZE) + tx * _TILE_SIZE) * scale_yx[1]
    y = (np.arange(_TILE_SIZE) + ty * _TILE_SIZE) * scale_yx[0]
    period = bundle["period_xy_um"]
    spline = bundle["periodic_spline"]
    bx = spline.transform(np.column_stack([np.remainder(x / period[0], 1), np.zeros(len(x))]))
    by = spline.transform(np.column_stack([np.zeros(len(y)), np.remainder(y / period[1], 1)]))
    width = bx.shape[1] // 2
    coefficient = bundle["coefficient"].reshape(width, width, 2)
    field = np.einsum("xi,ijt,yj->yxt", bx[:, :width], coefficient, by[:, width:], optimize=True)
    field -= bundle["periodic_center"] @ bundle["coefficient"]
    gain = np.exp(-field).astype(np.float32)
    if not np.isfinite(gain).all() or np.any(gain <= 0):
        raise ValueError("Periodic pixel correction produced invalid gains")
    gain.flags.writeable = False
    return gain


@dataclass(frozen=True)
class PeriodicPixels:
    """Picklable pixel transform; cached arrays remain local to each worker."""

    path: str
    sha256: str
    scale_zyx_um: tuple[float, float, float]

    def __post_init__(self):
        scale = np.asarray(self.scale_zyx_um)
        if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Pixel correction requires positive finite ZYX voxel spacing")

    @property
    def identity(self):
        return dict(
            model=self.path,
            model_sha256=self.sha256,
            scale_zyx_um=list(self.scale_zyx_um),
            operation="float32(raw) * float32(exp(-shared_xy_log_gain))",
            xy_origin_um=[0.0, 0.0],
        )

    def __call__(self, name: str, block: np.ndarray, origin_zyx: tuple[int, int, int]):
        """Multiply raw 2D/3D pixels using their global image-grid coordinates."""
        if name not in ("brdu", "edu"):
            raise ValueError(f"Periodic pixel correction has no target {name!r}")
        if block.ndim not in (2, 3):
            raise ValueError("Periodic pixel correction expects 2D or 3D pixels")
        target = ("brdu", "edu").index(name)
        y0, x0 = origin_zyx[-2:]
        y1, x1 = y0 + block.shape[-2], x0 + block.shape[-1]
        corrected = np.asarray(block, dtype=np.float32).copy()
        for ty in range(y0 // _TILE_SIZE, (y1 - 1) // _TILE_SIZE + 1):
            for tx in range(x0 // _TILE_SIZE, (x1 - 1) // _TILE_SIZE + 1):
                # Serialize cache misses across crop threads; each tile is evaluated once.
                with _TILE_LOCK:
                    gain = _gain_tile(self.path, self.sha256, self.scale_zyx_um[-2:], ty, tx)
                ay, ax = max(y0, ty * _TILE_SIZE), max(x0, tx * _TILE_SIZE)
                by, bx = min(y1, (ty + 1) * _TILE_SIZE), min(x1, (tx + 1) * _TILE_SIZE)
                corrected[..., ay - y0 : by - y0, ax - x0 : bx - x0] *= gain[
                    ay - ty * _TILE_SIZE : by - ty * _TILE_SIZE,
                    ax - tx * _TILE_SIZE : bx - tx * _TILE_SIZE,
                    target,
                ]
        return corrected
