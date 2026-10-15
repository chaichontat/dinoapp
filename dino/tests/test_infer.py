from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import zarr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from dino import infer


class _MeanModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :2].mean(dim=(2, 3)) + 1.0


def _fake_cell_dino_transforms(monkeypatch: pytest.MonkeyPatch) -> None:
    transforms = ModuleType("dinov2.data.cell_dino.transforms")
    monkeypatch.setattr(
        transforms,
        "make_classification_eval_cell_transform",
        lambda *, resize_size, crop_size: torch.nn.Identity(),
        raising=False,
    )
    monkeypatch.setitem(sys.modules, transforms.__name__, transforms)


def _make_group_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cell": "roiA|11",
                "dataset": "DS1",
                "roi": "roiA",
                "x": 5.0,
                "y": 4.0,
                "z": 0.0,
                "_x_center": 5,
                "_y_center": 4,
                "_z_index": 0,
                "_label": 11,
                "_row_order": 5,
            },
            {
                "cell": "roiA|21",
                "dataset": "DS1",
                "roi": "roiA",
                "x": 3.0,
                "y": 8.0,
                "z": 1.0,
                "_x_center": 3,
                "_y_center": 8,
                "_z_index": 1,
                "_label": 21,
                "_row_order": 2,
            },
            {
                "cell": "roiA|12",
                "dataset": "DS1",
                "roi": "roiA",
                "x": 10.0,
                "y": 9.0,
                "z": 0.0,
                "_x_center": 10,
                "_y_center": 9,
                "_z_index": 0,
                "_label": 12,
                "_row_order": 9,
            },
            {
                "cell": "roiA|22",
                "dataset": "DS1",
                "roi": "roiA",
                "x": 9.0,
                "y": 2.0,
                "z": 1.0,
                "_x_center": 9,
                "_y_center": 2,
                "_z_index": 1,
                "_label": 22,
                "_row_order": 1,
            },
        ]
    )


def test_copy_cell_into_batch_image_pads_scales_and_masks_exact_label() -> None:
    brdu = np.array(
        [
            [511, 1023, 1535],
            [2047, 2559, 3071],
            [3583, 4095, 4607],
        ],
        dtype=np.uint16,
    )
    fused_slab = np.stack((brdu, brdu + 256), axis=-1)
    seg_slab = np.array(
        [
            [7, 8, 0],
            [0, 7, 7],
            [7, 8, 7],
        ],
        dtype=np.int32,
    )
    image = np.zeros((5, 4, 4), dtype=np.uint8)

    has_mask = infer._copy_cell_into_batch_image(
        image,
        fused_slab=fused_slab,
        seg_slab=seg_slab,
        x_center=1,
        y_center=1,
        crop_size=4,
        label=7,
    )

    expected = np.array(
        [
            [
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 9, 11],
                [0, 13, 0, 17],
            ],
            [
                [0, 0, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 10, 12],
                [0, 14, 0, 18],
            ],
            [[0, 0, 0, 0]] * 4,
            [[0, 0, 0, 0]] * 4,
            [[0, 0, 0, 0]] * 4,
        ],
        dtype=np.uint8,
    )

    assert has_mask
    assert np.array_equal(image, expected)


def test_slab_batch_loader_emits_expected_pixels_and_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = infer
    _fake_cell_dino_transforms(monkeypatch)
    group = _make_group_rows()

    fused_path = tmp_path / "fused.zarr"
    seg_path = tmp_path / "seg.zarr"
    fused_data = np.empty((2, 12, 13, 2), dtype=np.uint16)
    fused_data[..., 0] = 511
    fused_data[..., 1] = 767
    seg_data = np.zeros((2, 12, 13), dtype=np.int32)
    seg_data[0, 2:6, 3:7] = 11
    seg_data[0, 7:10, 8:12] = 12
    seg_data[1, 6:10, 1:5] = 21
    seg_data[1, 0:4, 7:11] = 22
    zarr.open_array(fused_path, mode="w", shape=fused_data.shape, dtype=fused_data.dtype)[:] = fused_data
    zarr.open_array(seg_path, mode="w", shape=seg_data.shape, dtype=seg_data.dtype)[:] = seg_data

    dataset = module._SlabBatchIterableDataset(
        group,
        fused_path=fused_path,
        seg_path=seg_path,
        channel_indices=(0, 1),
        crop_size=6,
        input_channels=5,
        resize_size=6,
        crop_eval_size=6,
        batch_size=1,
        y_dim=fused_data.shape[1],
        x_dim=fused_data.shape[2],
    )
    loader = module._build_batch_loader(dataset, num_workers=0, device=torch.device("cpu"))

    got_batches = list(loader)
    assert len(got_batches) == 4

    full_mask = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    short_mask = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    for (batch_tensor, batch_meta), (cell, row_order, mask) in zip(
        got_batches,
        [
            ("roiA|11", 5, full_mask),
            ("roiA|12", 9, short_mask),
            ("roiA|21", 2, full_mask),
            ("roiA|22", 1, full_mask),
        ],
        strict=True,
    ):
        expected_image = np.zeros((1, 5, 6, 6), dtype=np.float32)
        expected_image[0, 0] = mask
        expected_image[0, 1] = mask * 2
        assert np.array_equal(batch_tensor.numpy(), expected_image)
        assert batch_meta["cell"] == [cell]
        assert batch_meta["dataset"] == ["DS1"]
        assert batch_meta["roi"] == ["roiA"]
        assert np.array_equal(batch_meta["_row_order"], np.asarray([row_order]))
        assert batch_meta["_has_target_mask"] == [True]


def test_slab_batch_loader_retains_cell_with_empty_target_mask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = infer
    _fake_cell_dino_transforms(monkeypatch)
    group = pd.DataFrame(
        [
            {
                "cell": "roiA|99",
                "dataset": "DS1",
                "roi": "roiA",
                "x": 4.0,
                "y": 4.0,
                "z": 0.0,
                "_x_center": 4,
                "_y_center": 4,
                "_z_index": 0,
                "_label": 99,
                "_row_order": 0,
            }
        ]
    )

    fused_path = tmp_path / "fused.zarr"
    seg_path = tmp_path / "seg.zarr"
    fused_data = np.ones((1, 8, 8, 2), dtype=np.uint16)
    seg_data = np.zeros((1, 8, 8), dtype=np.int32)
    zarr.open_array(fused_path, mode="w", shape=fused_data.shape, dtype=fused_data.dtype)[:] = fused_data
    zarr.open_array(seg_path, mode="w", shape=seg_data.shape, dtype=seg_data.dtype)[:] = seg_data

    dataset = module._SlabBatchIterableDataset(
        group,
        fused_path=fused_path,
        seg_path=seg_path,
        channel_indices=(0, 1),
        crop_size=4,
        input_channels=5,
        resize_size=4,
        crop_eval_size=4,
        batch_size=1,
        y_dim=8,
        x_dim=8,
    )

    batch_tensor, batch_meta = next(iter(dataset))

    assert np.count_nonzero(batch_tensor.numpy()) == 0
    assert batch_meta["cell"] == ["roiA|99"]
    assert batch_meta["_has_target_mask"] == [False]


def test_copy_cell_into_batch_image_keeps_zero_intensity_masked_cell() -> None:
    fused_slab = np.zeros((8, 8, 2), dtype=np.uint16)
    seg_slab = np.zeros((8, 8), dtype=np.int32)
    seg_slab[2:6, 2:6] = 7
    image = np.zeros((5, 4, 4), dtype=np.uint8)

    has_mask = infer._copy_cell_into_batch_image(
        image,
        fused_slab=fused_slab,
        seg_slab=seg_slab,
        x_center=4,
        y_center=4,
        crop_size=4,
        label=7,
    )

    assert has_mask
    assert np.array_equal(image, np.zeros((5, 4, 4), dtype=np.uint8))


def test_resolve_output_path_defaults_to_nvme_dataset_path() -> None:
    module = infer

    cfg = module.InferConfig(
        barrage_dir=Path("output/brdu_edu_barrage"),
        h5ad_path=Path("/tmp/input.h5ad"),
        model_path=Path("output/brdu_edu_barrage/brdu_edu_cell_dino_hpa_vitl16_logreg.joblib"),
        output_path=None,
        fused_zarr=None,
        segmentation_zarr=None,
        weights=None,
        datasets=(),
        rois=(),
        batch_size=128,
        num_workers=4,
        skip_existing=False,
    )
    table = pd.DataFrame({"dataset": ["20251201_JaxA6_Coro6"]})

    got = module._resolve_output_path(cfg, table)

    assert got == Path("~/nvme/dinoinfer/20251201_JaxA6_Coro6-dino-vit16.parquet").expanduser()


@pytest.mark.integration
def test_predict_group_preserves_row_order_with_multiple_workers(tmp_path: Path, monkeypatch: Any) -> None:
    module = infer
    _fake_cell_dino_transforms(monkeypatch)
    workspace_root = tmp_path / "DS1"
    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "ready.DONE").write_text("", encoding="utf-8")
    ws = module.Workspace(workspace_root)
    group = _make_group_rows()

    fused_path = ws.stitch("roiA", "edu") / module.FUSED_ZARR_NAME
    seg_path = ws.stitch("roiA", "pi") / "segmentation.zarr"
    fused_path.parent.mkdir(parents=True, exist_ok=True)
    seg_path.parent.mkdir(parents=True, exist_ok=True)
    fused_data = np.arange(2 * 12 * 13 * 2, dtype=np.uint16).reshape(2, 12, 13, 2)
    seg_data = np.zeros((2, 12, 13), dtype=np.int32)
    seg_data[0, 2:6, 3:7] = 11
    seg_data[0, 7:10, 8:12] = 12
    seg_data[1, 6:10, 1:5] = 21
    seg_data[1, 0:4, 7:11] = 22
    fused = zarr.open_array(fused_path, mode="w", shape=fused_data.shape, dtype=fused_data.dtype)
    fused[:] = fused_data
    fused.attrs["key"] = ["brdu", "edu"]
    zarr.open_array(seg_path, mode="w", shape=seg_data.shape, dtype=seg_data.dtype)[:] = seg_data

    train_features = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )
    scaler = StandardScaler().fit(train_features)
    brdu_model = LogisticRegression(random_state=0, solver="lbfgs").fit(
        train_features, np.asarray([0, 0, 1, 1])
    )
    edu_model = LogisticRegression(random_state=0, solver="lbfgs").fit(
        train_features, np.asarray([1, 0, 1, 0])
    )

    monkeypatch.setattr(
        module,
        "resolve_workspace_root",
        lambda dataset: pytest.fail("explicit image paths must bypass workspace discovery"),
    )
    out = module._predict_group(
        group=group,
        model=_MeanModel(),
        bundle={
            "feature_extractor": "cell_dino_cp_vits8",
            "codebook": "edu",
            "seg_codebook": "pi",
            "segmentation_name": "segmentation.zarr",
            "crop_size": 6,
            "resize_size": 6,
            "crop_eval_size": 6,
            "feature_scaler": scaler,
            "brdu_model": brdu_model,
            "edu_model": edu_model,
        },
        device=torch.device("cpu"),
        batch_size=2,
        num_workers=2,
        fused_path=fused_path,
        seg_path=seg_path,
    )

    assert out["_row_order"].tolist() == [1, 2, 5, 9]
    assert out["cell"].tolist() == ["roiA|22", "roiA|21", "roiA|11", "roiA|12"]
    assert ((out["brdu_prob"] >= 0.0) & (out["brdu_prob"] <= 1.0)).all()
    assert ((out["edu_prob"] >= 0.0) & (out["edu_prob"] <= 1.0)).all()


def test_predict_transformed_batch_sets_nan_without_evaluating_empty_mask() -> None:
    module = infer

    train_features = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], dtype=np.float32
    )
    scaler = StandardScaler().fit(train_features)
    brdu_model = LogisticRegression(random_state=0, solver="lbfgs").fit(
        train_features, np.asarray([0, 0, 1, 1])
    )
    edu_model = LogisticRegression(random_state=0, solver="lbfgs").fit(
        train_features, np.asarray([1, 0, 1, 0])
    )

    class RecordingModel(_MeanModel):
        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes: list[int] = []

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.batch_sizes.append(len(x))
            return super().forward(x)

    model = RecordingModel()
    got = module._predict_transformed_batch(
        batch_meta={
            "cell": ["roiA|1", "roiA|2"],
            "dataset": ["DS1", "DS1"],
            "roi": ["roiA", "roiA"],
            "_row_order": np.asarray([0, 1], dtype=np.int64),
            "_has_target_mask": [True, False],
        },
        batch_images=torch.ones((2, 5, 2, 2), dtype=torch.float32),
        model=model,
        scaler=scaler,
        brdu_model=brdu_model,
        edu_model=edu_model,
        device=torch.device("cpu"),
    )

    assert model.batch_sizes == [1]
    assert np.isfinite(got.loc[0, "brdu_prob"])
    assert np.isfinite(got.loc[0, "edu_prob"])
    assert np.isnan(got.loc[1, "brdu_prob"])
    assert np.isnan(got.loc[1, "edu_prob"])


def test_slab_bounds_clip_to_image_edges() -> None:
    module = infer

    rows = pd.DataFrame(
        [
            {"x": 0.0, "y": 1.0},
            {"x": 9.0, "y": 7.0},
        ]
    )

    assert module._slab_bounds_for_group(rows, crop_size=6, y_dim=8, x_dim=10) == (0, 8, 0, 10)
