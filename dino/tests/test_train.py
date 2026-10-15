from __future__ import annotations

from pathlib import Path

import pytest

from dino import train


@pytest.mark.parametrize("destination", ["cache", "model"])
def test_training_serialization_failure_preserves_existing_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, destination: str
) -> None:
    from types import SimpleNamespace

    import joblib
    import pandas as pd

    cfg = SimpleNamespace(
        seed=0, limit=0, test_fraction=0.25, cv_folds=2,
        barrage_dir=tmp_path, feature_cache=tmp_path / "cache.joblib",
        model_out=tmp_path / "model.joblib", weights=tmp_path / "weights",
        dinov2_repo=tmp_path / "repo", hub_model="test", batch_size=8, num_workers=0,
        input_channels=4, codebook="test", seg_codebook="test", segmentation_name="test",
        crop_size=8, resize_size=8, crop_eval_size=8,
    )
    output = cfg.feature_cache if destination == "cache" else cfg.model_out
    joblib.dump({"previous": destination}, output)
    before = output.read_bytes()
    rows = pd.DataFrame({"cell": [f"cell-{i}" for i in range(16)],
                         "brdu": [0, 0, 1, 1] * 4, "edu": [0, 1, 0, 1] * 4})
    monkeypatch.setattr(train, "_parse_args", lambda: cfg)
    monkeypatch.setattr(train, "load_barrage_training_table", lambda **kwargs: rows)
    monkeypatch.setattr(train, "file_sha256", lambda path: "weights")
    monkeypatch.setattr(train, "path_sha256", lambda path: "source")
    monkeypatch.setattr(train, "_feature_cache_metadata", lambda *args, **kwargs: {})
    monkeypatch.setattr(train, "_load_feature_cache", lambda *args, **kwargs: {"entries": {}})
    monkeypatch.setattr(train, "_load_cell_dino_model", lambda **kwargs: object())

    def extract(*, rows, **kwargs):
        targets = rows[["brdu", "edu"]].to_numpy()
        return targets.astype(float), targets, destination == "cache"

    monkeypatch.setattr(train, "_extract_features_with_cache", extract)
    monkeypatch.setattr(train, "_best_logreg_for_target_cv", lambda **kwargs: (1.0, 1.0, 2))

    def fail_serialization(value, path):
        Path(path).write_bytes(b"partial serialization")
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(train.joblib, "dump", fail_serialization)
    with pytest.raises(RuntimeError, match="serialization failed"):
        train.main()

    assert output.read_bytes() == before
    assert joblib.load(output) == {"previous": destination}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize('label', [2, 9])
def test_training_crop_selects_target_label_and_rejects_empty_target(
    monkeypatch: pytest.MonkeyPatch, label: int
) -> None:
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd
    import zarr

    cfg = SimpleNamespace(codebook='edu', seg_codebook='pi', segmentation_name='seg', crop_size=3, input_channels=4)
    row = pd.Series(dict(dataset='DS', roi='r', cell=f'DS:r|{label}', x=1, y=1, z=0))
    dataset = train.BarrageCellDataset(pd.DataFrame([row]), cfg=cfg)
    dataset.workspace_cache['DS'] = SimpleNamespace()
    fused = zarr.array(np.zeros((1, 3, 3, 2), dtype=np.uint16))
    fused.attrs['key'] = ['brdu', 'edu']
    dataset.fused_cache[('DS', 'r', 'edu')] = fused
    labels = np.array([[0, 2, 0], [2, 2, 3], [0, 3, 0]], dtype=np.uint16)
    dataset.seg_cache[('DS', 'r', 'pi', 'seg')] = zarr.array(labels[None])
    thumb = np.full((3, 3, 2), [256, 512], dtype=np.uint16)
    monkeypatch.setattr(train, 'cell_thumbnail_from_fused', lambda *args, **kwargs: thumb)

    if label == 9:
        with pytest.raises(ValueError, match='Target cell mask is empty for DS:r'):
            dataset._load_masked_cell_image(row)
    else:
        image = dataset._load_masked_cell_image(row)
        expected = np.zeros((3, 3, 4), dtype=np.uint8)
        expected[labels == 2, :2] = [1, 2]
        np.testing.assert_array_equal(image, expected)
