from __future__ import annotations

import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
import zarr

from dino import zarr_train
from dino.training_split import split_dataset_rows, split_training_rows
from dino.zarr_source import build_source_manifest, write_json_atomic
from dino_markers import (
    DEFAULT_CONTRACT,
    ContractMismatchError,
    MarkerContract,
    MissingContractError,
    parse_config,
    save_config,
)


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
    monkeypatch.setattr(zarr_train, "_load_training_rows_many", lambda paths, *, contract: rows)
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

    features, missing = zarr_train._cached_features(rows, cache, ("brdu", "edu"))

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

    features, missing = zarr_train._cached_features(rows, cache, ("brdu", "edu"))
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
    training, validation = split_training_rows(rows, markers=DEFAULT_CONTRACT.markers, limit=0, test_fraction=0.25, seed=0)

    assert len(training) == 6
    assert len(validation) == 2
    for marker in ("brdu", "edu"):
        assert training[marker].value_counts().min() >= 2
        assert set(validation[marker]) == {0, 1}
    repeated = split_training_rows(rows, markers=DEFAULT_CONTRACT.markers, limit=0, test_fraction=0.25, seed=0)
    pd.testing.assert_frame_equal(training, repeated[0])
    pd.testing.assert_frame_equal(validation, repeated[1])


@pytest.mark.parametrize("limit", [0, 8])
def test_split_rejects_missing_marker_support(limit) -> None:
    rows = pd.DataFrame({"brdu": [0] * 8 + [1] * 2, "edu": [0, 1] * 5})
    with pytest.raises(ValueError, match="No supported.*stratification.*at least 2"):
        split_training_rows(rows, markers=DEFAULT_CONTRACT.markers, limit=limit, test_fraction=0.25, seed=0)


@pytest.mark.parametrize("limit, succeeds", [(8, True), (5, False)])
def test_limit_selects_supported_cohort_or_rejects(limit, succeeds) -> None:
    rows = pd.DataFrame({"cell": [f"cell-{i}" for i in range(10)],
                         "brdu": [0, 0, 1, 1] * 2 + [0, 0],
                         "edu": [0, 1, 0, 1] * 2 + [0, 1]})
    if not succeeds:
        with pytest.raises(ValueError, match="No supported.*stratification"):
            split_training_rows(rows, markers=DEFAULT_CONTRACT.markers, limit=limit, test_fraction=0.25, seed=0)
        return
    training, validation = split_training_rows(rows, markers=DEFAULT_CONTRACT.markers, limit=limit, test_fraction=0.25, seed=0)
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
    cfg = SimpleNamespace(limit=5, test_fraction=0.25, seed=0, contract=DEFAULT_CONTRACT,
                          barrage_dir=tmp_path, barrage_dirs=(tmp_path,))
    module = zarr_train if trainer == "native" else train
    monkeypatch.setattr(module, "_parse_args", lambda: cfg)
    if trainer == "native":
        monkeypatch.setattr(module, "_load_training_rows_many", lambda paths, *, contract: rows)
        monkeypatch.setattr(module, "load_cell_dino_model",
                            lambda **kwargs: pytest.fail("unsupported split reached the encoder"))
    else:
        monkeypatch.setattr(module, "load_barrage_training_table", lambda **kwargs: rows)
        monkeypatch.setattr(module, "_load_cell_dino_model",
                            lambda **kwargs: pytest.fail("unsupported split reached the encoder"))
    with pytest.raises(ValueError, match="No supported.*stratification"):
        module.main()
    assert not list(tmp_path.iterdir())


def _stratified(n: int, strata, *, test_size: int | None = None, train_size: int | None = None, seed: int = 0):
    from sklearn.model_selection import train_test_split

    return train_test_split(
        np.arange(n), test_size=test_size, train_size=train_size, random_state=seed, stratify=np.asarray(strata)
    )


def _eight_combinations(markers: tuple[str, ...]) -> pd.DataFrame:
    """Every 0/1 assignment of three markers twice, with the first marker alternating fastest."""
    grid = [[(index >> bit) & 1 for index in range(8)] * 2 for bit in range(3)]
    return pd.DataFrame({"cell": [f"cell-{i}" for i in range(16)], **dict(zip(markers, grid, strict=True))})


def test_joint_label_joins_markers_in_contract_order() -> None:
    from dino.training_split import joint_label

    rows = pd.DataFrame({"brdu": [1, 0], "edu": [0, 1], "pcna": [1, 1]})

    assert joint_label(rows, ("brdu", "edu")).tolist() == ["1_0", "0_1"]
    assert joint_label(rows, ("pcna", "edu", "brdu")).tolist() == ["1_0_1", "1_1_0"]
    assert joint_label(rows, ("edu",)).tolist() == ["0", "1"]


def test_three_marker_split_falls_back_to_markers_in_contract_order() -> None:
    markers = ("pcna", "edu", "brdu")
    rows = _eight_combinations(markers)

    training, validation = split_training_rows(rows, markers=markers, limit=0, test_fraction=0.25, seed=0)

    # Eight joint strata exceed the four validation rows, so the first marker stratifies the split.
    keep, held_out = _stratified(16, rows["pcna"].astype(str), train_size=12)
    assert training["cell"].tolist() == rows["cell"].iloc[np.sort(keep)].tolist()
    assert validation["cell"].tolist() == rows["cell"].iloc[np.sort(held_out)].tolist()
    for marker in markers:
        assert set(validation[marker]) == {0, 1}
        assert training[marker].value_counts().min() >= 2


def test_three_marker_split_requires_support_for_every_marker() -> None:
    markers = ("pcna", "edu", "brdu")
    rows = _eight_combinations(markers)
    rows.loc[rows["brdu"].eq(1), "brdu"] = 0
    rows.loc[0, "brdu"] = 1

    with pytest.raises(ValueError, match="No supported joint/per-marker stratification.*at least 2"):
        split_training_rows(rows, markers=markers, limit=0, test_fraction=0.25, seed=0)


def test_single_marker_split_has_no_duplicate_candidate() -> None:
    rows = pd.DataFrame({"cell": [f"cell-{i}" for i in range(12)], "pcna": [0, 1] * 6})

    training, validation = split_training_rows(rows, markers=("pcna",), limit=0, test_fraction=0.25, seed=0)

    keep, held_out = _stratified(12, rows["pcna"].astype(str), train_size=9)
    assert training["cell"].tolist() == rows["cell"].iloc[np.sort(keep)].tolist()
    assert validation["cell"].tolist() == rows["cell"].iloc[np.sort(held_out)].tolist()


def _two_dataset_rows(per_dataset: int = 24) -> tuple[pd.DataFrame, np.ndarray]:
    index = np.arange(per_dataset)
    rows = pd.DataFrame(
        {"brdu": np.tile(index % 2, 2), "edu": np.tile(index // 2 % 2, 2), "pcna": np.tile(index // 4 % 2, 2)}
    )
    return rows, np.repeat([0, 1], per_dataset)


def test_dataset_split_reproduces_the_joint_train_default_stratification() -> None:
    rows, dataset_index = _two_dataset_rows()

    train, validation = split_dataset_rows(
        rows, markers=DEFAULT_CONTRACT.markers, dataset_index=dataset_index, datasets=2, test_size=12, seed=7
    )

    expected_train, expected_validation = _stratified(
        len(rows), dataset_index * 4 + rows["brdu"] + 2 * rows["edu"], test_size=12, seed=7
    )
    np.testing.assert_array_equal(train, np.sort(expected_train))
    np.testing.assert_array_equal(validation, np.sort(expected_validation))


def test_dataset_split_orders_joint_bits_by_contract_markers() -> None:
    rows, dataset_index = _two_dataset_rows()
    markers = ("pcna", "brdu", "edu")

    train, validation = split_dataset_rows(
        rows, markers=markers, dataset_index=dataset_index, datasets=2, test_size=16, seed=1
    )

    strata = dataset_index * 8 + rows["pcna"] + 2 * rows["brdu"] + 4 * rows["edu"]
    expected_train, expected_validation = _stratified(len(rows), strata, test_size=16, seed=1)
    np.testing.assert_array_equal(train, np.sort(expected_train))
    np.testing.assert_array_equal(validation, np.sort(expected_validation))


def test_dataset_split_falls_back_to_each_marker_in_order_when_joint_strata_are_infeasible() -> None:
    rows, dataset_index = _two_dataset_rows()
    rows.loc[(dataset_index == 0) & rows["brdu"].eq(1) & rows["edu"].eq(1), "edu"] = 0
    rows.loc[3, "edu"] = 1  # a single dataset-0 row with both markers positive

    train, validation = split_dataset_rows(
        rows, markers=DEFAULT_CONTRACT.markers, dataset_index=dataset_index, datasets=2, test_size=12, seed=7
    )

    expected_train, expected_validation = _stratified(len(rows), dataset_index * 2 + rows["brdu"], test_size=12, seed=7)
    np.testing.assert_array_equal(train, np.sort(expected_train))
    np.testing.assert_array_equal(validation, np.sort(expected_validation))


def test_dataset_split_rejects_a_dataset_without_two_rows_of_each_class() -> None:
    rows, dataset_index = _two_dataset_rows()
    rows.loc[dataset_index == 1, "pcna"] = 0
    rows.loc[24, "pcna"] = 1

    with pytest.raises(ValueError, match=r"markers \['brdu', 'edu', 'pcna'\].*2 datasets"):
        split_dataset_rows(
            rows, markers=("brdu", "edu", "pcna"), dataset_index=dataset_index, datasets=2, test_size=12, seed=7
        )


# --- Marker contract: input channels, contract resolution, and three-marker training. ---

CUSTOM = MarkerContract("dapi", ("edu", "brdu", "pcna"))
LEGACY_MODEL = Path(__file__).parent / "fixtures" / "legacy_model"


def _config(contract: MarkerContract):
    return parse_config(
        {
            "version": 1,
            "reference": {"key": contract.reference, "label": contract.reference.upper()},
            "markers": [
                {"key": marker, "label": marker.upper(), "color": "#ff0000"} for marker in contract.markers
            ],
            "presentation": {
                "crop_panels": list(contract.channel_keys),
                "negative_color": [0.25, 0.25, 0.25],
                "uncalled_color": "#999999",
            },
        },
        source="test",
    )


def _declare(directory: Path, contract: MarkerContract) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    save_config(_config(contract), directory)
    return directory


def _train_argv(monkeypatch, tmp_path, *args: str, barrages=None) -> None:
    barrages = barrages or [tmp_path / "barrage"]
    argv = ["dino-zarr train", "--out-dir", str(tmp_path / "out")]
    for barrage in barrages:
        argv += ["--barrage-dir", str(barrage)]
    monkeypatch.setattr(sys, "argv", [*argv, *args])


def test_default_contract_keeps_the_default_input_channels(monkeypatch, tmp_path) -> None:
    _train_argv(monkeypatch, tmp_path)

    config = zarr_train._parse_args()

    assert config.contract == DEFAULT_CONTRACT
    assert config.input_channel_names_by_target == {"brdu": ("brdu", "edu"), "edu": ("brdu", "edu")}


def test_input_channels_follow_flag_order_and_default_the_unnamed_target(monkeypatch, tmp_path) -> None:
    _train_argv(monkeypatch, tmp_path, "--input-channel", "brdu=pi", "--input-channel", "brdu=brdu")

    config = zarr_train._parse_args()

    assert config.input_channel_names_by_target == {"brdu": ("pi", "brdu"), "edu": ("brdu", "edu")}


@pytest.mark.parametrize(
    "values, match",
    [
        (["brdu"], "TARGET=CHANNEL"),
        (["=brdu"], "TARGET=CHANNEL"),
        (["edu="], "TARGET=CHANNEL"),
        (["pcna=pcna"], "target 'pcna' is not a marker"),
        (["brdu=pcna"], "channel 'pcna' is not a channel"),
        (["brdu=pi", "brdu=pi"], "must be unique"),
        (["edu=edu", "edu=pi", "edu=brdu"], "2 slots"),
    ],
)
def test_input_channels_reject_malformed_values(values, match) -> None:
    with pytest.raises(ValueError, match=match):
        zarr_train.resolve_input_channels(values, DEFAULT_CONTRACT, slots=2)


def test_custom_contract_requires_input_channels_for_every_marker() -> None:
    values = ["edu=edu", "brdu=brdu", "brdu=dapi"]

    with pytest.raises(ValueError, match=r"missing \['pcna'\]"):
        zarr_train.resolve_input_channels(values, CUSTOM, slots=4)
    resolved = zarr_train.resolve_input_channels([*values, "pcna=dapi", "pcna=pcna"], CUSTOM, slots=4)
    assert resolved == {"edu": ("edu",), "brdu": ("brdu", "dapi"), "pcna": ("dapi", "pcna")}
    assert list(resolved) == list(CUSTOM.markers)


def test_training_takes_its_contract_from_the_barrage(monkeypatch, tmp_path) -> None:
    _declare(tmp_path / "barrage", CUSTOM)
    _train_argv(monkeypatch, tmp_path, *(f"--input-channel={m}={m}" for m in CUSTOM.markers))

    config = zarr_train._parse_args()

    assert config.contract == CUSTOM
    assert config.input_channel_names_by_target == {marker: (marker,) for marker in CUSTOM.markers}


def test_training_rejects_barrages_with_reordered_markers(monkeypatch, tmp_path) -> None:
    _declare(tmp_path / "first", MarkerContract("pi", ("brdu", "edu")))
    _declare(tmp_path / "second", MarkerContract("pi", ("edu", "brdu")))
    _train_argv(monkeypatch, tmp_path, barrages=[tmp_path / "first", tmp_path / "second"])
    monkeypatch.setattr(
        zarr_train, "load_cell_dino_model", lambda **kwargs: pytest.fail("a contract mismatch reached the encoder")
    )

    with pytest.raises(ContractMismatchError, match="marker order is part of the contract"):
        zarr_train.main()
    assert not (tmp_path / "out").exists()


def test_training_rejects_a_target_source_with_another_contract(monkeypatch, tmp_path) -> None:
    _declare(tmp_path / "barrage", MarkerContract("pi", ("brdu", "edu")))
    _declare(tmp_path / "source", MarkerContract("pi", ("edu", "brdu")))
    _train_argv(monkeypatch, tmp_path, "--target-source-dir", str(tmp_path / "source"))

    with pytest.raises(ContractMismatchError, match="source"):
        zarr_train._parse_args()


def test_markers_option_asserts_the_artifact_contract(monkeypatch, tmp_path) -> None:
    _declare(tmp_path / "barrage", MarkerContract("pi", ("brdu", "edu")))
    reordered = _declare(tmp_path / "supplied", MarkerContract("pi", ("edu", "brdu"))) / "markers.json"
    _train_argv(monkeypatch, tmp_path, "--markers", str(reordered))

    with pytest.raises(ContractMismatchError, match="--markers requires reference 'pi', markers \\['edu', 'brdu'\\]"):
        zarr_train._parse_args()
    matching = tmp_path / "barrage" / "markers.json"
    _train_argv(monkeypatch, tmp_path, "--markers", str(matching))
    assert zarr_train._parse_args().contract == DEFAULT_CONTRACT


class _SumEncoder(torch.nn.Module):
    """Slot sums plus a constant component, as the characterization stub does."""

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        sums = batch[:, :2].sum(dim=(2, 3))
        return torch.cat([sums, torch.full_like(sums[:, :1], 300.0)], dim=1)


def _patch_encoder(monkeypatch, factory) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(zarr_train, "load_cell_dino_model", lambda **kwargs: factory())
    monkeypatch.setattr(zarr_train, "build_cell_transform", lambda **kwargs: torch.nn.Identity())


def _fail_encoder():
    pytest.fail("a reusable cache must not load the encoder")


def _three_marker_barrage(tmp_path) -> Path:
    """Thirty-two 2x2 cells, four of each assignment of three markers, as a barrage and its source."""
    barrage = _declare(tmp_path / "barrage", CUSTOM)
    rng = np.random.default_rng(11)
    cells = np.arange(32)
    calls = {marker: (cells // 2**bit) % 2 for bit, marker in enumerate(CUSTOM.markers)}
    labels = zarr.open_array(tmp_path / "labels.zarr", mode="w", shape=(1, 16, 32), chunks=(1, 16, 32), dtype="u4")
    pixels = np.zeros((1, 16, 32), dtype=np.uint32)
    for cell in cells:
        row, column = divmod(int(cell), 8)
        pixels[0, 4 * row + 1 : 4 * row + 3, 4 * column + 1 : 4 * column + 3] = cell + 1
    labels[:] = pixels
    channel_paths = {}
    for key in CUSTOM.channel_keys:
        array = zarr.open_array(tmp_path / f"{key}.zarr", mode="w", shape=(1, 18, 34), chunks=(1, 18, 34), dtype="u2")
        base = np.full((1, 18, 34), 500.0)
        if key in calls:
            for cell in cells:
                row, column = divmod(int(cell), 8)
                base[0, 4 * row + 2 : 4 * row + 4, 4 * column + 2 : 4 * column + 4] += 2000.0 * calls[key][cell]
        array[:] = (base + rng.uniform(0, 100, base.shape)).astype(np.uint16)
        channel_paths[key] = tmp_path / f"{key}.zarr"
    manifest = build_source_manifest(
        labels_path=tmp_path / "labels.zarr",
        channel_paths=channel_paths,
        offset_zyx=(0, 1, 1),
        required_channels=CUSTOM.channel_keys,
        dataset="ds",
        roi="roi",
    )
    write_json_atomic(manifest, barrage / "source.json")
    index = pd.DataFrame(
        {
            "cell": [f"roi|{cell + 1}" for cell in cells],
            "label": cells + 1,
            "centroid_z": 0.0,
            "centroid_y": [4.0 * (cell // 8) + 1.5 for cell in cells],
            "centroid_x": [4.0 * (cell % 8) + 1.5 for cell in cells],
            "plane_z": 0,
            "source_sha256": manifest["source_sha256"],
        }
    )
    index.to_csv(barrage / "index.csv", index=False)
    pd.DataFrame({"cell": index["cell"], **calls}).to_csv(barrage / "labels.csv", index=False)
    (tmp_path / "encoder").mkdir()
    (tmp_path / "encoder" / "config.json").write_text("{}")
    (tmp_path / "weights.pth").write_text("weights")
    return barrage


def _train_three_markers(monkeypatch, tmp_path, *, out: str, cache: Path | None = None) -> None:
    argv = [
        "dino-zarr train",
        "--barrage-dir", str(tmp_path / "barrage"),
        "--out-dir", str(tmp_path / out),
        "--dinov2-repo", str(tmp_path / "encoder"),
        "--weights", str(tmp_path / "weights.pth"),
        "--crop-size", "4", "--resize-size", "4", "--crop-eval-size", "4",
        "--batch-size", "4", "--test-fraction", "0.25", "--cache-gb", "0.001",
        "--prefetch-workers", "1", "--crop-workers", "1",
    ]
    for marker in CUSTOM.markers:
        argv += ["--input-channel", f"{marker}={marker}", "--input-channel", f"{marker}=dapi"]
    if cache is not None:
        argv += ["--feature-cache", str(cache)]
    monkeypatch.setattr(sys, "argv", argv)
    zarr_train.main()


def test_three_marker_training_writes_a_v7_bundle_and_v3_cache(monkeypatch, tmp_path) -> None:
    _three_marker_barrage(tmp_path)
    _patch_encoder(monkeypatch, lambda: _SumEncoder().eval())

    _train_three_markers(monkeypatch, tmp_path, out="out")

    bundle = joblib.load(tmp_path / "out" / "model.joblib")
    cache = joblib.load(tmp_path / "out" / "features.joblib")
    assert bundle["schema"] == "dino-zarr-model-v7"
    assert bundle["marker_contract"] == CUSTOM.to_json()
    assert bundle["input_channel_names_by_target"] == {m: [m, "dapi"] for m in CUSTOM.markers}
    for marker in CUSTOM.markers:
        for suffix in ("feature_scaler", "model", "cutoff", "best_c", "cv_balanced_accuracy", "cv_folds",
                       "validation_balanced_accuracy"):
            assert f"{marker}_{suffix}" in bundle, (marker, suffix)
    assert not any(key.startswith(("pi_", "dapi_")) for key in bundle)
    assert len(bundle["train_cells"]) + len(bundle["validation_cells"]) == 32
    assert list(bundle).index("edu_feature_scaler") < list(bundle).index("edu_model")
    assert cache["metadata"]["schema"] == "dino-zarr-features-v3"
    assert cache["metadata"]["marker_contract"] == CUSTOM.to_json()
    assert cache["metadata"]["input_channel_names_by_target"] == {m: [m, "dapi"] for m in CUSTOM.markers}
    assert len(cache["entries"]) == 3 * 32


def test_three_marker_cache_is_reused_without_the_encoder(monkeypatch, tmp_path) -> None:
    _three_marker_barrage(tmp_path)
    _patch_encoder(monkeypatch, lambda: _SumEncoder().eval())
    _train_three_markers(monkeypatch, tmp_path, out="first")
    cache = tmp_path / "first" / "features.joblib"
    before = cache.read_bytes()
    _patch_encoder(monkeypatch, _fail_encoder)

    _train_three_markers(monkeypatch, tmp_path, out="second", cache=cache)

    assert cache.read_bytes() == before
    first, second = (joblib.load(tmp_path / name / "model.joblib") for name in ("first", "second"))
    assert second["edu_cutoff"] == first["edu_cutoff"]


def test_current_cache_without_its_contract_fails_naming_the_cache(monkeypatch, tmp_path) -> None:
    _three_marker_barrage(tmp_path)
    cache = tmp_path / "cache.joblib"
    joblib.dump({"metadata": {"schema": "dino-zarr-features-v3"}, "entries": {}}, cache)
    _patch_encoder(monkeypatch, _fail_encoder)

    with pytest.raises(MissingContractError, match="cache.joblib"):
        _train_three_markers(monkeypatch, tmp_path, out="out", cache=cache)


def test_changed_input_channels_reset_a_recognized_cache(monkeypatch, tmp_path) -> None:
    _three_marker_barrage(tmp_path)
    _patch_encoder(monkeypatch, lambda: _SumEncoder().eval())
    _train_three_markers(monkeypatch, tmp_path, out="first")
    cache = tmp_path / "first" / "features.joblib"
    stale = joblib.load(cache)
    stale["metadata"]["input_channel_names_by_target"]["edu"] = ["dapi", "edu"]
    joblib.dump(stale, cache)

    _train_three_markers(monkeypatch, tmp_path, out="second", cache=cache)

    refreshed = joblib.load(cache)
    assert refreshed["metadata"]["input_channel_names_by_target"]["edu"] == ["edu", "dapi"]
    assert len(refreshed["entries"]) == 3 * 32


def test_an_extended_legacy_cache_keeps_its_v2_metadata(monkeypatch, tmp_path) -> None:
    root = tmp_path / "legacy_model"
    shutil.copytree(LEGACY_MODEL, root)
    monkeypatch.chdir(root)
    cache_path = root / "model" / "features.joblib"
    cache = joblib.load(cache_path)
    legacy_metadata = dict(cache["metadata"])
    dropped = sorted(cache["entries"])[:5]
    for key in dropped:
        del cache["entries"][key]
    joblib.dump(cache, cache_path)
    _patch_encoder(monkeypatch, lambda: _SumEncoder().eval())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dino-zarr train", "--barrage-dir", ".", "--out-dir", "retrain", "--feature-cache", str(cache_path),
            "--dinov2-repo", "encoder", "--weights", "encoder/weights.pth", "--crop-size", "4",
            "--resize-size", "4", "--crop-eval-size", "4", "--batch-size", "4", "--test-fraction", "0.25",
            "--cache-gb", "0.001", "--prefetch-workers", "1", "--crop-workers", "1",
        ],
    )

    zarr_train.main()

    extended = joblib.load(cache_path)
    assert extended["metadata"] == legacy_metadata
    assert extended["metadata"]["schema"] == "dino-zarr-features-v2"
    assert "marker_contract" not in extended["metadata"]
    assert set(extended["entries"]) >= set(dropped)
    assert joblib.load(root / "retrain" / "model.joblib")["schema"] == "dino-zarr-model-v7"
