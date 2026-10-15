from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from dino import model_sweep
from fishtools.brdu.features import META_REQUIRED_COLS


def test_model_sweep_exports_to_fresh_destination(tmp_path: Path, monkeypatch) -> None:
    n = 16
    index = pd.DataFrame({"cell": [f"cell-{i}" for i in range(n)]})
    for offset, col in enumerate(META_REQUIRED_COLS):
        index[col] = np.arange(n) + offset + 10
    index.to_csv(tmp_path / "index.csv", index=False)
    pd.DataFrame({"cell": index.cell, "brdu": np.arange(n) % 2, "edu": (np.arange(n) // 2) % 2}).to_csv(
        tmp_path / "labels.csv", index=False
    )
    destination = tmp_path / "fresh" / "sweep"
    cfg = model_sweep.SweepConfig(tmp_path / "index.csv", tmp_path / "labels.csv", destination, 0, 2)
    monkeypatch.setattr(model_sweep, "_parse_args", lambda: cfg)
    monkeypatch.setattr(model_sweep, "_models", lambda seed: {"logreg": LogisticRegression(max_iter=2000)})
    model_sweep.main()
    assert len(pd.read_csv(destination / "model_auc.csv")) == 2
    for target in ("brdu", "edu"):
        assert not pd.read_csv(destination / f"feature_importance_{target}_logreg.csv").empty
    assert (destination / "model_roc_curves.png").is_file()
