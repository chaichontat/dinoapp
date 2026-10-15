from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dino.intensity_context import IntensityContext
from dino_markers import DEFAULT_CONTRACT


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
    context = IntensityContext.load(path, scale_zyx_um=np.ones(3), markers=DEFAULT_CONTRACT.markers)

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
        IntensityContext.load(path, scale_zyx_um=np.ones(3), markers=DEFAULT_CONTRACT.markers)


def _three_marker_props(path, *, columns) -> None:
    table = {
        "label": np.arange(1, 5, dtype=np.uint32),
        "centroid_z": [0.0] * 4,
        "centroid_y": [0.0] * 4,
        "centroid_x": [0.0, 1.0, 2.0, 10.0],
        "intensity_brdu_mean": [10.0, 20.0, 30.0, 40.0],
        "intensity_edu_mean": [100.0, 200.0, 300.0, 400.0],
        "intensity_pcna_mean": [1.0, 2.0, 3.0, 4.0],
    }
    pd.DataFrame({column: table[column] for column in columns}).to_parquet(path, index=False)


def test_intensity_context_maps_markers_by_column_name_not_position(tmp_path) -> None:
    path = tmp_path / "props.parquet"
    _three_marker_props(
        path,
        columns=[
            "intensity_pcna_mean",
            "intensity_edu_mean",
            "centroid_x",
            "label",
            "intensity_brdu_mean",
            "centroid_y",
            "centroid_z",
        ],
    )

    context = IntensityContext.load(path, scale_zyx_um=np.ones(3), markers=("edu", "brdu", "pcna"))

    assert list(context.means) == ["edu", "brdu", "pcna"]
    np.testing.assert_array_equal(context.means["edu"], [100.0, 200.0, 300.0, 400.0])
    np.testing.assert_array_equal(context.means["brdu"], [10.0, 20.0, 30.0, 40.0])
    np.testing.assert_array_equal(context.means["pcna"], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(context.points_zyx_um[:, 2], [0.0, 1.0, 2.0, 10.0])
    features = context.features_for_labels(np.array([3]), neighbors=2)
    assert list(features) == ["edu", "brdu", "pcna"]
    np.testing.assert_allclose(features["pcna"], [[np.log1p(3.0), np.log(4.0 / 2.5)]])


def test_intensity_context_requires_every_contract_marker_column(tmp_path) -> None:
    path = tmp_path / "props.parquet"
    _three_marker_props(
        path,
        columns=["label", "centroid_z", "centroid_y", "centroid_x", "intensity_brdu_mean", "intensity_edu_mean"],
    )

    with pytest.raises(Exception, match="intensity_pcna_mean"):
        IntensityContext.load(path, scale_zyx_um=np.ones(3), markers=("brdu", "edu", "pcna"))
