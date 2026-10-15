from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def add_dinoinfer_scores(adata: ad.AnnData, dinoinfer_dir: Path) -> ad.AnnData:
    """Join per-dataset Cell-DINO inference parquets onto ``adata.obs`` by cell ID."""

    parquet_paths = sorted(dinoinfer_dir.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No dinoinfer parquet files found in {dinoinfer_dir}.")

    parts = [pd.read_parquet(path)[["brdu_prob", "edu_prob"]] for path in parquet_paths]
    joined = pd.concat(parts, axis=0, copy=False)
    joined.index = joined.index.astype(str)

    obs = adata.obs.drop(columns=["brdu_prob", "edu_prob"], errors="ignore").join(
        joined,
        how="left",
        validate="one_to_one",
    )
    scores = obs[["brdu_prob", "edu_prob"]].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Requested cells require complete, finite probabilities in [0, 1].")
    obs["edu_pos"] = obs["edu_prob"] > 0.5
    obs["brdu_pos"] = obs["brdu_prob"] > 0.5
    adata.obs = obs
    return adata
