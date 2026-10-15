from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dino.intensity_context import IntensityContext


def test_intensity_context_excludes_self_and_preserves_query_order(tmp_path) -> None:
    path = tmp_path / "props.parquet"
    pd.DataFrame(
        {
            "label": np.arange(1, 5, dtype=np.uint32),
            "centroid_z": [0.0, 0.0, 0.0, 0.0],
            "centroid_y": [0.0, 0.0, 0.0, 0.0],
            "centroid_x": [0.0, 1.0, 2.0, 10.0],
            "intensity_brdu_mean": [10.0, 20.0, 30.0, 40.0],
            "intensity_edu_mean": [100.0, 200.0, 300.0, 400.0],
        }
    ).to_parquet(path, index=False)
    context = IntensityContext.load(path, scale_zyx_um=np.ones(3))

    features = context.features_for_labels(np.array([3, 1]), neighbors=2)

    expected_brdu = np.array(
        [
            [np.log1p(30.0), np.log((30.0 + 1.0) / (15.0 + 1.0))],
            [np.log1p(10.0), np.log((10.0 + 1.0) / (25.0 + 1.0))],
        ]
    )
    np.testing.assert_allclose(features["brdu"], expected_brdu)
    labels, distances = context.nearest_labels(np.array([[0.0, 0.0, 2.1]]))
    np.testing.assert_array_equal(labels, [3])
    np.testing.assert_allclose(distances, [0.1])


def test_intensity_context_rejects_noncontiguous_labels(tmp_path) -> None:
    path = tmp_path / "props.parquet"
    pd.DataFrame(
        {
            "label": np.array([1, 3], dtype=np.uint32),
            "centroid_z": [0.0, 0.0],
            "centroid_y": [0.0, 0.0],
            "centroid_x": [0.0, 1.0],
            "intensity_brdu_mean": [10.0, 20.0],
            "intensity_edu_mean": [100.0, 200.0],
        }
    ).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="contiguous"):
        IntensityContext.load(path, scale_zyx_um=np.ones(3))
