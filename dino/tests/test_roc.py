from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from dino import roc


def test_evaluate_targets_writes_expected_auc_outputs(tmp_path: Path) -> None:
    module = roc

    labels = pd.DataFrame(
        {
            "cell": ["ds1:r1|1", "ds1:r1|2", "ds1:r1|3", "ds1:r1|4", "missing|5"],
            "brdu": [0, 0, 1, 1, 1],
            "edu": [0, 1, 0, 1, 0],
        }
    )
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)

    dinoinfer_dir = tmp_path / "dinoinfer"
    dinoinfer_dir.mkdir()
    scores = pd.DataFrame(
        {
            "cell": ["ds1:r1|1", "ds1:r1|2", "ds1:r1|3", "ds1:r1|4"],
            "dataset": ["ds1"] * 4,
            "roi": ["r1"] * 4,
            "brdu_prob": [0.1, 0.2, 0.8, 0.9],
            "edu_prob": [0.1, 0.8, 0.2, 0.9],
        }
    ).set_index("cell")
    scores.to_parquet(dinoinfer_dir / "ds1-dino-vit16.parquet")

    matched, total_labels, split_description = module._load_matched_labels_and_scores(
        labels_csv=labels_path,
        dinoinfer_dir=dinoinfer_dir,
        bundle_path=None,
        split="all",
    )
    assert total_labels == 5
    assert matched["cell"].tolist() == ["ds1:r1|1", "ds1:r1|2", "ds1:r1|3", "ds1:r1|4"]
    assert split_description == "All labeled barrage cells (n=5)"

    summary, curves = module._evaluate_targets(matched, total_labels=total_labels)
    summary_by_target = summary.set_index("target")
    assert summary_by_target.loc["brdu", "auc"] == pytest.approx(1.0)
    assert summary_by_target.loc["edu", "auc"] == pytest.approx(1.0)

    out_png = tmp_path / "roc.png"
    module._plot_roc_curves(
        selected_split="test",
        selected_summary=summary,
        selected_curves=curves,
        all_summary=summary,
        all_curves=curves,
        examples=pd.DataFrame(
            [
                {
                    "channel": channel,
                    "status": status,
                    "image": np.zeros((80, 80, 4), dtype=np.uint8),
                }
                for channel in ("brdu", "edu")
                for status in ("Positive", "Negative")
                for _ in range(16)
            ]
        ),
        example_crop_size=80,
        example_scale_bar_um=5.0,
        out_png=out_png,
    )
    assert out_png.exists()


def test_channel_montage_places_negative_and_positive_examples_side_by_side() -> None:
    tile = np.zeros((2, 3, 4), dtype=np.uint8)
    examples = pd.DataFrame(
        [
            {"channel": "brdu", "status": status, "image": tile.copy()}
            for status in ("Negative", "Positive")
            for _ in range(16)
        ]
    )

    montage = roc._make_channel_montage(examples, channel="brdu", gap=2, group_gap=4)

    assert montage.shape == (14, 40, 4)
    assert np.all(montage[:, 18:22] == 255)


@pytest.mark.parametrize("identity", [None, "another-model", "matching"])
def test_roc_checks_prediction_model_identity(tmp_path: Path, identity: str | None) -> None:
    from dino.artifacts import file_sha256, write_parquet_atomic

    labels_path = tmp_path / "labels.csv"
    pd.DataFrame({"cell": ["a", "b"], "brdu": [0, 1], "edu": [1, 0]}).to_csv(labels_path, index=False)
    bundle_path = tmp_path / "bundle.joblib"
    joblib.dump({}, bundle_path)
    scores = pd.DataFrame({"brdu_prob": [0.1, 0.9], "edu_prob": [0.9, 0.1]}, index=["a", "b"])
    provenance = {}
    if identity is not None:
        provenance["model_sha256"] = file_sha256(bundle_path) if identity == "matching" else identity
    write_parquet_atomic(scores, tmp_path / "scores" / "a.parquet", provenance=provenance)
    with nullcontext() if identity == "matching" else pytest.raises(ValueError, match="model_sha256"):
        matched, total, _ = roc._load_matched_labels_and_scores(
            labels_csv=labels_path,
            dinoinfer_dir=tmp_path / "scores",
            bundle_path=bundle_path,
            split="all",
        )
        assert total == len(matched) == 2


@pytest.mark.parametrize("split", ["train", "test"])
def test_saved_split_membership_survives_label_reordering(tmp_path: Path, split: str) -> None:
    labels_path = tmp_path / "labels.csv"
    labels = pd.DataFrame({"cell": ["a", "b", "c", "d", "outside"], "brdu": [0, 1, 0, 1, 0], "edu": [1, 0, 1, 0, 1]})
    bundle_path = tmp_path / "bundle.joblib"
    bundle = {"train_cells": ["c", "a"], "test_cells": ["d", "b"]}
    joblib.dump(bundle, bundle_path)
    for frame in (labels, labels.iloc[::-1]):
        frame.to_csv(labels_path, index=False)
        selected, description = roc._select_labels_for_split(
            labels_csv=labels_path,
            bundle_path=bundle_path,
            split=split,
        )
        assert selected.cell.tolist() == bundle[f"{split}_cells"]
        assert description == f"Saved DINO {split} split (train=2, test=2)"


@pytest.mark.parametrize(
    "membership",
    [
        {},
        {"train_cells": ["a", "a"], "test_cells": ["b"]},
        {"train_cells": ["a"], "test_cells": ["a", "b"]},
        {"train_cells": ["a"], "test_cells": ["missing"]},
    ],
)
def test_heldout_split_requires_valid_saved_cell_ids(tmp_path: Path, membership: dict) -> None:
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame({"cell": ["a", "b"], "brdu": [0, 1], "edu": [1, 0]}).to_csv(labels_path, index=False)
    bundle_path = tmp_path / "bundle.joblib"
    index = pd.DataFrame(
        {
            "cell": ["a", "b"],
            "dataset": ["ds", "ds"],
            "roi": ["r", "r"],
            "x": [0, 1],
            "y": [0, 1],
            "z": [0, 1],
        }
    )
    index.to_csv(tmp_path / "index.csv", index=False)
    cache_path = tmp_path / "cache.joblib"
    joblib.dump({"entries": {roc._row_feature_cache_key(row): {} for _, row in index.iterrows()}}, cache_path)
    joblib.dump(
        {"feature_cache": str(cache_path), "train_rows": 1, "test_rows": 1, **membership}, bundle_path
    )
    with pytest.raises(ValueError, match="saved|duplicate|overlap|missing"):
        roc._select_labels_for_split(
            labels_csv=labels_path,
            bundle_path=bundle_path,
            split="test",
        )


def test_legacy_bundle_still_supports_cache_evaluation(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        {
            "cell": ["a", "b"],
            "brdu": [0, 1],
            "edu": [1, 0],
            "dataset": ["ds", "ds"],
            "roi": ["r", "r"],
            "x": [0, 1],
            "y": [0, 1],
            "z": [0, 1],
        }
    )
    rows[["cell", "brdu", "edu"]].to_csv(tmp_path / "labels.csv", index=False)
    rows.to_csv(tmp_path / "index.csv", index=False)
    cache_path = tmp_path / "cache.joblib"
    joblib.dump({"entries": {roc._row_feature_cache_key(rows.iloc[1]): {}}}, cache_path)
    bundle_path = tmp_path / "bundle.joblib"
    joblib.dump({"feature_cache": str(cache_path)}, bundle_path)
    selected, _ = roc._select_labels_for_split(
        labels_csv=tmp_path / "labels.csv",
        bundle_path=bundle_path,
        split="cache",
    )
    assert selected.cell.tolist() == ["b"]
