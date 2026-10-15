"""Physical geometry, grouped evaluation, and selection rules for smooth subtraction."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from dino.spatial_subtraction import SpatialSubtraction

GRID = [0.001, 0.01, 0.1]
SEED = 83261


def validate_location(output_dir: Path, prefix: str) -> None:
    if not isinstance(output_dir, Path):
        raise ValueError("output_dir must be a Path")
    if re.fullmatch(r"[a-z][a-z0-9_]*", prefix) is None:
        raise ValueError("prefix must be a lowercase filename identifier")


def metrics(target: np.ndarray, predicted: np.ndarray) -> dict:
    residual = target - predicted
    return {
        "samples": len(target),
        "pinball_loss": float(np.mean(np.maximum(0.05 * residual, -0.95 * residual))),
        "negative_share": float(np.mean(residual < 0)),
        "residual_q05": float(np.quantile(residual, 0.05)),
    }


def outside(model: SpatialSubtraction, xy: np.ndarray) -> np.ndarray:
    return ((xy < model.domain_xy_um[:, 0]) | (xy > model.domain_xy_um[:, 1])).any(axis=1)


def geometry(frame: pd.DataFrame, *, samples_path: Path) -> pd.DataFrame:
    if "x_um" not in frame:
        saved = pd.read_parquet(
            samples_path,
            columns=["z_index", "y_index", "x_index", "x_um", "y_um", "z_um"],
        )
        frame = frame.merge(saved, on=["z_index", "y_index", "x_index"], how="left", validate="one_to_one")
    if not np.isfinite(frame[["x_um", "y_um", "z_um"]].to_numpy()).all():
        raise ValueError("Training/evaluation geometry requires finite physical XYZ")
    frame = frame.copy()
    frame["x_block"] = np.floor(frame.x_um / 250).astype(int)
    frame["y_block"] = np.floor(frame.y_um / 250).astype(int)
    frame["z_band"] = (np.floor(frame.z_um / 100) * 100).astype(int)
    return frame


def inner_splits(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split supplied training rows by whole XY blocks, stratified on Z bands."""
    groups = pd.MultiIndex.from_frame(frame[["x_block", "y_block"]]).factorize()[0]
    return list(
        StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(frame, frame.z_band, groups)
    )


def select_penalty(table: pd.DataFrame) -> tuple[float, dict[str, float]]:
    """Choose by observation-weighted inner loss; exact ties prefer smoothing."""
    scores = {}
    for penalty in GRID:
        rows = table[table.penalty == penalty]
        scores[str(penalty)] = float((rows.pinball_loss * rows.samples).sum() / rows.samples.sum())
    chosen = min(GRID, key=lambda p: (scores[str(p)], -p))
    return chosen, scores
