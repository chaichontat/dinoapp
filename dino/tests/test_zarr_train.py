from __future__ import annotations

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from dino import zarr_train
from dino.training_split import split_training_rows


def test_training_defaults_to_bfloat16(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dino-zarr train",
            "--barrage-dir",
            str(tmp_path / "input"),
            "--barrage-dir",
            str(tmp_path / "input-2"),
            "--out-dir",
            str(tmp_path / "output"),
        ],
    )

    config = zarr_train._parse_args()

    assert config.barrage_dirs == (tmp_path / "input", tmp_path / "input-2")
    assert config.feature_precision == "bfloat16"
    assert config.feature_cache is None
    assert config.prefetch_workers == 8
    assert config.crop_workers == 8


def test_training_accepts_reusable_feature_cache(monkeypatch, tmp_path) -> None:
    cache = tmp_path / "cache" / "features.joblib"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dino-zarr train",
            "--barrage-dir",
            str(tmp_path / "input"),
            "--out-dir",
            str(tmp_path / "output"),
            "--feature-cache",
            str(cache),
        ],
    )

    config = zarr_train._parse_args()

    assert config.feature_cache == cache


def test_training_rejects_missing_source_before_loading_encoder(monkeypatch, tmp_path) -> None:
    rows = pd.DataFrame(
        {
            "cell": [f"cell-{index}" for index in range(10)],
            "source_sha256": ["missing-source"] * 10,
            "brdu": [0, 1] * 5,
            "edu": [0, 1] * 5,
        }
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["dino-zarr train", "--barrage-dir", str(tmp_path), "--out-dir", str(tmp_path / "output")],
    )
    monkeypatch.setattr(zarr_train, "_load_training_rows_many", lambda paths: rows)
    monkeypatch.setattr(zarr_train, "_load_training_sources_many", lambda paths: {})
    monkeypatch.setattr(
        zarr_train,
        "load_cell_dino_model",
        lambda **kwargs: pytest.fail("missing sources must be rejected before encoder loading"),
    )

    with pytest.raises(ValueError, match="Missing source manifests.*missing-source"):
        zarr_train.main()

    assert not (tmp_path / "output").exists()


def test_logreg_search_bounds_native_threads(monkeypatch) -> None:
    requested_limits: list[int] = []

    def limit_threads(*, limits: int):
        requested_limits.append(limits)
        return nullcontext()

    monkeypatch.setattr(zarr_train, "threadpool_limits", limit_threads)
    features = np.arange(16, dtype=np.float64).reshape(8, 2)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    zarr_train._best_logreg_c(features, labels, cv_folds=2, seed=0)

    assert requested_limits == [1]


def test_logreg_search_does_not_exceed_c_one(monkeypatch) -> None:
    c_values: list[float] = []

    class LogisticModel:
        def __init__(self, *, C: float, **kwargs) -> None:
            c_values.append(C)

        def fit(self, features, labels):
            return self

        def predict(self, features):
            return np.arange(len(features)) % 2

    monkeypatch.setattr(zarr_train, "LogisticRegression", LogisticModel)
    features = np.arange(16, dtype=np.float64).reshape(8, 2)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    zarr_train._best_logreg_c(features, labels, cv_folds=2, seed=0)

    assert min(c_values) == pytest.approx(1e-6)
    assert max(c_values) == pytest.approx(1.0)
    assert len(set(c_values)) == 25


def test_cached_features_reports_only_missing_targets() -> None:
    rows = pd.DataFrame(
        {
            "cell": ["cell-1", "cell-2"],
            "label": [1, 2],
            "plane_z": [3, 4],
            "centroid_y": [5.0, 6.0],
            "centroid_x": [7.0, 8.0],
        }
    )
    cache = {
        "entries": {
            zarr_train._feature_key(target, row): {"feature": np.array([row_index, target_index])}
            for row_index, (_, row) in enumerate(rows.iterrows())
            for target_index, target in enumerate(("brdu", "edu"))
            if (row_index, target) != (1, "edu")
        }
    }

    features, missing = zarr_train._cached_features(rows, cache)

    assert missing == {1: ("edu",)}
    np.testing.assert_array_equal(features["brdu"][0], [0, 0])
    np.testing.assert_array_equal(features["edu"][0], [0, 1])
    np.testing.assert_array_equal(features["brdu"][1], [1, 0])
    assert features["edu"][1] is None


def test_extract_missing_features_reads_targets_together_and_preserves_order(
    monkeypatch, tmp_path
) -> None:
    rows = pd.DataFrame(
        {
            "cell": ["cell-1", "cell-2", "cell-3"],
            "source_sha256": ["source-a"] * 3,
            "label": [1, 2, 3],
            "plane_z": [3, 4, 5],
            "centroid_y": [5.0, 6.0, 7.0],
            "centroid_x": [7.0, 8.0, 9.0],
        }
    )
    features = {
        "brdu": [None, np.array([99], dtype=np.float32), None],
        "edu": [None, None, None],
    }
    missing = {0: ("brdu", "edu"), 1: ("edu",), 2: ("brdu", "edu")}
    closed: list[str] = []
    sources = {
        "source-a": SimpleNamespace(
            manifest={"source_sha256": "source-a"},
            close=lambda: closed.append("source-a"),
        )
    }
    opened_channels: list[tuple[str, ...]] = []

    def open_reader(source, *, cache_gb, channel_names):
        opened_channels.append(channel_names)
        return source

    def sort_for_io(frame, reader):
        return frame.assign(_row_order=np.arange(len(frame))).iloc[::-1]

    batch_targets: list[tuple[str, ...]] = []

    def iter_image_batches(ordered, *, channel_names_by_target, require_mask, **kwargs):
        batch_targets.append(tuple(channel_names_by_target))
        assert require_mask
        row_orders = ordered["_row_order"].astype(int).tolist()
        images = {
            target: [np.full((1, 1, 1), order + offset, dtype=np.uint8) for order in row_orders]
            for target, offset in (("brdu", 10), ("edu", 20))
        }
        yield images, row_orders

    monkeypatch.setattr(zarr_train, "model_crop_reader", open_reader)
    monkeypatch.setattr(zarr_train, "sort_cells_for_io", sort_for_io)
    monkeypatch.setattr(zarr_train, "iter_model_image_batches", iter_image_batches)
    monkeypatch.setattr(zarr_train, "iter_prefetched_batches", lambda batches: batches)
    monkeypatch.setattr(
        zarr_train,
        "prepare_model_batch",
        lambda images, **kwargs: torch.from_numpy(np.stack(images)).float(),
    )
    monkeypatch.setattr(
        zarr_train,
        "extract_normalized_features",
        lambda model, batch, precision: batch[:, 0, 0].numpy(),
    )
    monkeypatch.setattr(zarr_train, "_dump_joblib_atomic", lambda value, path: None)
    cfg = SimpleNamespace(
        batch_size=2,
        cache_gb=1.0,
        crop_size=1,
        input_channels=4,
        highpass_sigma_px=0.0,
        mask_erosion_px=0,
        feature_precision="bfloat16",
        feature_cache=tmp_path / "features.joblib",
        out_dir=tmp_path,
        prefetch_workers=2,
        crop_workers=2,
        input_channel_names_by_target={"brdu": ("brdu",), "edu": ("edu",)},
    )

    zarr_train._extract_missing_features(
        rows,
        cfg=cfg,
        sources=sources,
        model=object(),
        transform=object(),
        device=torch.device("cpu"),
        cache={"entries": {}},
        features=features,
        missing=missing,
    )

    extracted = zarr_train._stack_features(features)
    np.testing.assert_array_equal(extracted["brdu"], [[10], [99], [11]])
    np.testing.assert_array_equal(extracted["edu"], [[20], [20], [21]])
    assert opened_channels == [("brdu", "edu")]
    assert batch_targets == [("brdu", "edu"), ("edu",)]
    assert closed == ["source-a"]

def test_extract_missing_features_checkpoints_target_before_failure(monkeypatch, tmp_path) -> None:
    rows = pd.DataFrame(
        {
            "cell": ["cell-1", "cell-2"],
            "source_sha256": ["source-a", "source-a"],
            "label": [1, 2],
            "plane_z": [0, 0],
            "centroid_y": [5, 6],
            "centroid_x": [7, 8],
        }
    )
    features = {target: [None, None] for target in ("brdu", "edu")}
    missing = {0: ("brdu", "edu"), 1: ("brdu", "edu")}
    sources = {"source-a": SimpleNamespace(manifest={"source_sha256": "source-a"})}

    monkeypatch.setattr(
        zarr_train,
        "model_crop_reader",
        lambda *args, **kwargs: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        zarr_train,
        "sort_cells_for_io",
        lambda frame, reader: frame.assign(_row_order=np.arange(len(frame))),
    )

    image = np.zeros((1, 1, 1), dtype=np.uint8)

    def iter_image_batches(*args, **kwargs):
        yield {"brdu": [image], "edu": [image]}, [0]
        yield {"brdu": [image], "edu": [image]}, [1]

    monkeypatch.setattr(zarr_train, "iter_model_image_batches", iter_image_batches)
    monkeypatch.setattr(zarr_train, "iter_prefetched_batches", lambda batches: batches)
    monkeypatch.setattr(
        zarr_train,
        "prepare_model_batch",
        lambda images, **kwargs: torch.from_numpy(np.stack(images)).float(),
    )
    calls = 0

    def extract(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second target failed")
        return np.ones((1, 3), dtype=np.float32)

    monkeypatch.setattr(zarr_train, "extract_normalized_features", extract)
    cache_path = tmp_path / "features.joblib"
    cache = {"metadata": {"schema": "test"}, "entries": {}}
    cfg = SimpleNamespace(
        batch_size=1,
        cache_gb=1.0,
        crop_size=2,
        input_channels=4,
        highpass_sigma_px=0.0,
        mask_erosion_px=0,
        feature_precision="bfloat16",
        feature_cache=cache_path,
        out_dir=tmp_path,
        prefetch_workers=2,
        crop_workers=2,
        input_channel_names_by_target={"brdu": ("brdu",), "edu": ("edu",)},
    )

    with pytest.raises(RuntimeError, match="second target failed"):
        zarr_train._extract_missing_features(
            rows,
            cfg=cfg,
            sources=sources,
            model=object(),
            transform=lambda batch: batch,
            device=torch.device("cpu"),
            cache=cache,
            features=features,
            missing=missing,
        )

    persisted = joblib.load(cache_path)
    brdu_key = zarr_train._feature_key("brdu", rows.iloc[0])
    edu_key = zarr_train._feature_key("edu", rows.iloc[0])
    assert brdu_key in persisted["entries"]
    assert edu_key not in persisted["entries"]


def test_stack_cached_features_preserves_target_order() -> None:
    rows = pd.DataFrame(
        {
            "cell": ["cell-1"],
            "label": [1],
            "plane_z": [2],
            "centroid_y": [3.0],
            "centroid_x": [4.0],
        }
    )
    cache = {
        "metadata": {"schema": "test"},
        "entries": {
            zarr_train._feature_key(target, rows.iloc[0]): {
                "input_sha256": "legacy",
                "feature": np.array([target_index, target_index + 1], dtype=np.float32),
            }
            for target_index, target in enumerate(("brdu", "edu"))
        },
    }

    features, missing = zarr_train._cached_features(rows, cache)
    stacked = zarr_train._stack_features(features)

    assert missing == {}
    np.testing.assert_array_equal(stacked["brdu"], [[0.0, 1.0]])
    np.testing.assert_array_equal(stacked["edu"], [[1.0, 2.0]])


def test_parallel_logreg_search_matches_serial_search():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(120, 6))
    labels = (features[:, 0] + rng.normal(size=120) > 0).astype(int)
    assert zarr_train._best_logreg_c(features, labels, cv_folds=3, seed=0, n_jobs=3) == zarr_train._best_logreg_c(
        features, labels, cv_folds=3, seed=0)


def test_small_joint_cohort_uses_capacity_aware_split() -> None:
    rows = pd.DataFrame({"cell": [f"cell-{i}" for i in range(8)],
                         "brdu": [0, 0, 1, 1] * 2, "edu": [0, 1, 0, 1] * 2})
    training, validation = split_training_rows(rows, limit=0, test_fraction=0.25, seed=0)

    assert len(training) == 6
    assert len(validation) == 2
    for marker in ("brdu", "edu"):
        assert training[marker].value_counts().min() >= 2
        assert set(validation[marker]) == {0, 1}
    repeated = split_training_rows(rows, limit=0, test_fraction=0.25, seed=0)
    pd.testing.assert_frame_equal(training, repeated[0])
    pd.testing.assert_frame_equal(validation, repeated[1])


@pytest.mark.parametrize("limit", [0, 8])
def test_split_rejects_missing_marker_support(limit) -> None:
    rows = pd.DataFrame({"brdu": [0] * 8 + [1] * 2, "edu": [0, 1] * 5})
    with pytest.raises(ValueError, match="No supported.*stratification.*at least 2"):
        split_training_rows(rows, limit=limit, test_fraction=0.25, seed=0)


@pytest.mark.parametrize("limit, succeeds", [(8, True), (5, False)])
def test_limit_selects_supported_cohort_or_rejects(limit, succeeds) -> None:
    rows = pd.DataFrame({"cell": [f"cell-{i}" for i in range(10)],
                         "brdu": [0, 0, 1, 1] * 2 + [0, 0],
                         "edu": [0, 1, 0, 1] * 2 + [0, 1]})
    if not succeeds:
        with pytest.raises(ValueError, match="No supported.*stratification"):
            split_training_rows(rows, limit=limit, test_fraction=0.25, seed=0)
        return
    training, validation = split_training_rows(rows, limit=limit, test_fraction=0.25, seed=0)
    assert len(training) + len(validation) == limit
    assert set(training.cell).isdisjoint(validation.cell)
    for marker in ("brdu", "edu"):
        assert set(training[marker]) == {0, 1}
        assert training[marker].value_counts().min() >= 2
        assert set(validation[marker]) == {0, 1}


@pytest.mark.parametrize("trainer", ["native", "workspace"])
def test_unsupported_limit_fails_before_feature_extraction(monkeypatch, tmp_path, trainer) -> None:
    from dino import train

    rows = pd.DataFrame({"cell": [f"cell-{i}" for i in range(16)],
                         "brdu": [0, 0, 1, 1] * 4, "edu": [0, 1, 0, 1] * 4})
    cfg = SimpleNamespace(limit=5, test_fraction=0.25, seed=0,
                          barrage_dir=tmp_path, barrage_dirs=(tmp_path,))
    module = zarr_train if trainer == "native" else train
    monkeypatch.setattr(module, "_parse_args", lambda: cfg)
    if trainer == "native":
        monkeypatch.setattr(module, "_load_training_rows_many", lambda paths: rows)
        monkeypatch.setattr(module, "load_cell_dino_model",
                            lambda **kwargs: pytest.fail("unsupported split reached the encoder"))
    else:
        monkeypatch.setattr(module, "load_barrage_training_table", lambda **kwargs: rows)
        monkeypatch.setattr(module, "_load_cell_dino_model",
                            lambda **kwargs: pytest.fail("unsupported split reached the encoder"))
    with pytest.raises(ValueError, match="No supported.*stratification"):
        module.main()
    assert not list(tmp_path.iterdir())
