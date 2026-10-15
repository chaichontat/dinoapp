from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from dino.integrate import add_dinoinfer_scores


def test_add_dinoinfer_scores_joins_probabilities_and_calls(tmp_path) -> None:
    adata = ad.AnnData(
        X=np.empty((3, 0), dtype=np.float32),
        obs=pd.DataFrame(index=["cell-a", "cell-b", "cell-c"]),
    )
    pd.DataFrame(
        {"brdu_prob": [0.8, 0.2], "edu_prob": [0.1, 0.7]},
        index=["cell-a", "cell-b"],
    ).to_parquet(tmp_path / "a.parquet")
    pd.DataFrame(
        {"brdu_prob": [0.6], "edu_prob": [0.9]},
        index=["cell-c"],
    ).to_parquet(tmp_path / "b.parquet")

    result = add_dinoinfer_scores(adata, tmp_path)

    assert result is adata
    assert result.obs["brdu_prob"].tolist() == pytest.approx([0.8, 0.2, 0.6])
    assert result.obs["edu_prob"].tolist() == pytest.approx([0.1, 0.7, 0.9])
    assert result.obs["brdu_pos"].tolist() == [True, False, True]
    assert result.obs["edu_pos"].tolist() == [False, True, True]


def test_add_dinoinfer_scores_requires_parquet_inputs(tmp_path) -> None:
    adata = ad.AnnData(X=np.empty((1, 0), dtype=np.float32), obs=pd.DataFrame(index=["cell-a"]))

    with pytest.raises(FileNotFoundError, match="No dinoinfer parquet files"):
        add_dinoinfer_scores(adata, tmp_path)


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1, 1.1, "missing"])
def test_invalid_requested_scores_leave_obs_unchanged(tmp_path, value) -> None:
    adata = ad.AnnData(
        X=np.empty((2, 0), dtype=np.float32),
        obs=pd.DataFrame({"brdu_prob": [0.3, 0.4], "edu_pos": [True, True]}, index=["a", "b"]),
    )
    original = adata.obs.copy(deep=True)
    scores = pd.DataFrame({"brdu_prob": [0.8, 0.2], "edu_prob": [0.1, 0.7]}, index=["a", "b"])
    if value == "missing":
        scores = scores.loc[["a"]]
    else:
        scores.loc["b", "edu_prob"] = value
    scores.to_parquet(tmp_path / "scores.parquet")
    with pytest.raises(ValueError, match="finite|missing|probabilit"):
        add_dinoinfer_scores(adata, tmp_path)
    pd.testing.assert_frame_equal(adata.obs, original)


def test_probability_cutoff_and_unrequested_invalid_scores(tmp_path) -> None:
    adata = ad.AnnData(X=np.empty((1, 0)), obs=pd.DataFrame(index=["a"]))
    pd.DataFrame({"brdu_prob": [0.5, np.nan], "edu_prob": [1.0, np.nan]}, index=["a", "unused"]).to_parquet(
        tmp_path / "scores.parquet"
    )
    result = add_dinoinfer_scores(adata, tmp_path)
    assert not result.obs.loc["a", "brdu_pos"]
    assert result.obs.loc["a", "edu_pos"]
