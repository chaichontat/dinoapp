from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from dino_markers import intensity_column
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class IntensityContext:
    """All-cell 3D intensities and their physical-coordinate kNN index."""

    points_zyx_um: np.ndarray
    means: dict[str, np.ndarray]
    tree: cKDTree
    scale_zyx_um: np.ndarray

    @classmethod
    def load(cls, path: Path, *, scale_zyx_um: np.ndarray, markers: Sequence[str]) -> IntensityContext:
        parquet = pq.ParquetFile(path)
        count = parquet.metadata.num_rows
        points = np.empty((count, 3), dtype=np.float64)
        means = {marker: np.empty(count, dtype=np.float64) for marker in markers}
        columns = [
            "label",
            "centroid_z",
            "centroid_y",
            "centroid_x",
            *(intensity_column(marker) for marker in markers),
        ]
        offset = 0
        for batch in parquet.iter_batches(columns=columns, batch_size=262_144):
            size = batch.num_rows
            labels = batch.column("label").to_numpy(zero_copy_only=False)
            expected = np.arange(offset + 1, offset + size + 1, dtype=labels.dtype)
            if not np.array_equal(labels, expected):
                raise ValueError("Intensity labels must be contiguous and ordered from one")
            for axis, column in enumerate(("centroid_z", "centroid_y", "centroid_x")):
                points[offset : offset + size, axis] = (
                    batch.column(column).to_numpy(zero_copy_only=False) * scale_zyx_um[axis]
                )
            for marker in markers:
                means[marker][offset : offset + size] = batch.column(intensity_column(marker)).to_numpy(
                    zero_copy_only=False
                )
            offset += size
        if not np.isfinite(points).all() or any(
            not np.isfinite(values).all() for values in means.values()
        ):
            raise ValueError("Intensity context contains non-finite values")
        return cls(
            points_zyx_um=points,
            means=means,
            tree=cKDTree(points, compact_nodes=True, balanced_tree=True, copy_data=False),
            scale_zyx_um=scale_zyx_um,
        )

    def nearest_labels(self, points_zyx_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distances, indices = self.tree.query(
            np.asarray(points_zyx_px, dtype=np.float64) * self.scale_zyx_um,
            k=1,
            workers=1,
        )
        return indices.astype(np.int64) + 1, distances

    def neighbor_indices_for_labels(
        self, labels: np.ndarray, *, neighbors: int
    ) -> np.ndarray:
        labels = np.asarray(labels, dtype=np.int64)
        if neighbors <= 0:
            raise ValueError("neighbors must be positive")
        if np.any(labels < 1) or np.any(labels > len(self.points_zyx_um)):
            raise ValueError("Intensity-context labels are outside the available range")
        indices = labels - 1
        _, candidates = self.tree.query(
            self.points_zyx_um[indices], k=neighbors + 1, workers=1
        )
        not_self = candidates != indices[:, None]
        if not np.all(np.count_nonzero(not_self, axis=1) == neighbors):
            raise ValueError("Each intensity kNN query must contain itself exactly once")
        return candidates[not_self].reshape(len(indices), neighbors)

    def features_for_labels(
        self, labels: np.ndarray, *, neighbors: int
    ) -> dict[str, np.ndarray]:
        indices = np.asarray(labels, dtype=np.int64) - 1
        selected = self.neighbor_indices_for_labels(labels, neighbors=neighbors)
        features: dict[str, np.ndarray] = {}
        for target, means in self.means.items():
            target_mean = means[indices]
            local_mean = means[selected].mean(axis=1)
            features[target] = np.column_stack(
                (
                    np.log1p(target_mean),
                    np.log((target_mean + 1.0) / (local_mean + 1.0)),
                )
            )
        return features
