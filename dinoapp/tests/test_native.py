"""Behavior checks for native field sampling, picking, and barrage edits."""

import io
import json
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from dinoapp.native import (
    AnnotationFiles,
    FieldSource,
    combined_uncertainty,
    load_prediction_pointer,
    pick_cell,
    prefetch_brdu_crop,
    prefetch_edu_crop,
    render_brdu_crop,
    render_edu_crop,
    sample_center,
    toggle_brdu,
    toggle_edu,
    training_gallery,
    training_miscalls,
)
from dinoapp.native_app import new_gallery_state, update_gallery_card
from dinoapp.label_store import LabelConflictError, load_labels, update_label
from dino.zarr_source import CellCrop


@pytest.mark.parametrize(
    "adapter,prefix",
    [
        ("native", "full-run2"),
        ("prod3", "prod3-realigned-v1"),
        ("prod3_2", "235-4-edu-workflow-recovered"),
        ("prod2_2", "234-2-514561638"),
    ],
)
def test_field_overlays_preserve_adapter_annotation_namespace(adapter, prefix):
    source = object.__new__(import_module(f"dinoapp.{adapter}").FieldSource)
    source.calls = np.array([4, 2, 1], dtype=np.uint8)
    source.uncertainty = np.array([np.nan, 2.5, np.nan], dtype=np.float32)
    field = SimpleNamespace(ids=np.array([0, 1, 2, 5, 8], dtype=np.uint32))
    annotations = SimpleNamespace(
        values={
            "foreign|2": (0, 0),
            f"{prefix}|1": (1, 1),
            f"{prefix}|8": (0, 1),
        }
    )

    assert source.cell_id(1) == f"{prefix}|1"
    np.testing.assert_array_equal(source.field_calls(field, annotations), [4, 3, 1, 4, 1])
    np.testing.assert_allclose(
        source.field_uncertainty(field), [np.nan, 2.5, np.nan, np.nan, np.nan],
        equal_nan=True,
    )


def test_prediction_pointer_binds_model_and_complete_manifest(tmp_path: Path):
    model_dir = tmp_path / "logistic-pixel-v2"
    prediction_dir = model_dir / "cortex-pred" / "predictions"
    prediction_dir.mkdir(parents=True)
    (prediction_dir.parent / "complete.json").write_text(
        json.dumps({"model_sha256": "model-hash", "cells": 42})
    )
    pointer = tmp_path / "current.json"
    pointer.write_text(
        json.dumps(
            {
                "artifact": "dino-review-current-v1",
                "model_dir": str(model_dir),
                "prediction_dir": str(prediction_dir),
                "model_sha256": "model-hash",
                "cells": 42,
            }
        )
    )

    model_version, actual_dir, manifest_path, contract = load_prediction_pointer(pointer)

    assert model_version == "logistic-pixel-v2"
    assert actual_dir == prediction_dir
    assert manifest_path == prediction_dir.parent / "complete.json"
    assert contract["cells"] == 42


def test_seeded_fields_repeat_and_stay_on_the_boundary():
    segments = np.array([[[100.0, 100], [100, 400]], [[100, 400], [400, 400]]])
    center = sample_center(segments, seed=17, position=0)
    np.testing.assert_array_equal(center, sample_center(segments, seed=17, position=0))
    assert center[0] == 100 or center[1] == 400
    assert not np.array_equal(center, sample_center(segments, seed=18, position=0))
    assert not np.array_equal(center, sample_center(segments, seed=17, position=1))


def test_click_selects_the_segment_under_the_cursor_not_a_nearby_centroid():
    mask = np.array([[0, 10, 10], [20, 20, 0]], dtype=np.uint32)
    assert pick_cell(mask, 0.5, 0.1) == 10
    assert pick_cell(mask, 0.1, 0.9) == 20
    assert pick_cell(mask, 0.01, 0.01) is None
    for x, y in [(1, 0.2), (-0.01, 0.3), (0.2, 1), (float("nan"), 0.5)]:
        assert pick_cell(mask, x, y) is None


def test_combined_uncertainty_uses_the_closest_classifier_threshold():
    p_brdu = np.array([0.4, 0.2, 0.6])
    p_edu = np.array([0.1, 0.55, 0.9])

    result = combined_uncertainty(p_brdu, p_edu, brdu_threshold=0.5, edu_threshold=0.5)

    expected = np.maximum(1 / np.abs(p_brdu - 0.5), 1 / np.abs(p_edu - 0.5))
    np.testing.assert_allclose(result, expected)


def test_existing_labels_return_to_their_barrage_and_new_labels_go_to_default(tmp_path):
    old = tmp_path / "old" / "labels.csv"
    default = tmp_path / "pixel" / "labels.csv"
    update_label(old, "full-run2|10", (1, 0))
    files = AnnotationFiles((old, default), default)
    assert files.destination("full-run2|10") == old
    assert files.destination("full-run2|20") == default
    edit = files.assign("full-run2|10", (0, 1), expected=(1, 0))
    assert load_labels(old).iloc[0].edu == 1
    files.assign("full-run2|20", (1, 1), expected=None)
    assert load_labels(default).cell.tolist() == ["full-run2|20"]
    files.undo(edit)
    assert load_labels(old).iloc[0].brdu == 1
    assert load_labels(old).iloc[0].edu == 0
    assert load_labels(default).cell.tolist() == ["full-run2|20"]


def test_new_label_can_be_undone_and_stale_edits_are_rejected(tmp_path):
    path = tmp_path / "labels.csv"
    files = AnnotationFiles((path,), path)
    edit = files.assign("full-run2|5", (1, 1), expected=None)
    files.undo(edit)
    assert load_labels(path).empty
    edit = files.assign("full-run2|5", (0, 1), expected=None)
    update_label(path, "full-run2|5", (1, 0))
    with pytest.raises(LabelConflictError):
        files.undo(edit)
    assert load_labels(path).iloc[0][["brdu", "edu"]].tolist() == [1, 0]


def test_duplicate_tranche_ownership_fails_without_writing(tmp_path):
    paths = (tmp_path / "one.csv", tmp_path / "two.csv")
    for p in paths:
        update_label(p, "full-run2|1", (0, 1))
    original = [p.read_bytes() for p in paths]
    with pytest.raises(ValueError, match="multiple"):
        AnnotationFiles(paths, paths[-1])
    assert original == [p.read_bytes() for p in paths]


def test_training_miscalls_use_live_labels_and_requested_split():
    predictions = pd.DataFrame(
        {
            "cell": ["run|1", "run|2", "run|3"],
            "label": [1, 2, 3],
            "tranche": ["pixel"] * 3,
            "split": ["training", "training", "validation"],
            "p_brdu": [0.9, 0.2, 0.1],
            "pred_brdu": [True, False, False],
            "p_edu": [0.1, 0.8, 0.9],
            "pred_edu": [False, True, True],
        }
    )
    live = {"run|1": (0, 0), "run|2": (0, 1), "run|3": (0, 0)}

    result = training_miscalls(
        predictions, live, split="training", target="either", identity=(0, 0)
    )

    assert result.cell.tolist() == ["run|1"]
    assert result.iloc[0][["saved_brdu", "saved_edu"]].tolist() == [0, 0]
    assert result.iloc[0].audit_score == pytest.approx(0.9)
    assert result.iloc[0].audit_marker == "brdu"
    corrected = {**live, "run|1": (1, 0)}
    assert training_miscalls(
        predictions,
        corrected,
        split="training",
        target="either",
        identity=(0, 0),
    ).empty
    edu = training_miscalls(
        predictions, live, split="validation", target="edu", identity=(0, 0)
    )
    assert edu.cell.tolist() == ["run|3"]
    assert edu.audit_marker.tolist() == ["edu"]


def test_training_miscalls_show_each_active_marker_disagreement():
    predictions = pd.DataFrame(
        {
            "cell": ["run|1"],
            "label": [1],
            "tranche": ["pixel"],
            "split": ["training"],
            "p_brdu": [0.7],
            "pred_brdu": [True],
            "p_edu": [0.95],
            "pred_edu": [True],
        }
    )

    result = training_miscalls(
        predictions,
        {"run|1": (0, 0)},
        split="training",
        target="either",
        identity=(0, 0),
    )

    assert result.cell.tolist() == ["run|1", "run|1"]
    assert result.audit_marker.tolist() == ["edu", "brdu"]
    assert result.audit_score.tolist() == pytest.approx([0.95, 0.7])


def test_training_miscalls_are_subsetted_by_current_saved_identity():
    predictions = pd.DataFrame(
        {
            "cell": ["run|1", "run|2"],
            "label": [1, 2],
            "tranche": ["pixel", "pixel"],
            "split": ["training", "training"],
            "p_brdu": [0.9, 0.9],
            "pred_brdu": [True, True],
            "p_edu": [0.1, 0.1],
            "pred_edu": [False, False],
        }
    )
    live = {"run|1": (0, 0), "run|2": (0, 1)}

    result = training_miscalls(
        predictions, live, split="all", target="either", identity=(0, 1)
    )

    assert result.cell.tolist() == ["run|2", "run|2"]
    assert result.audit_marker.tolist() == ["brdu", "edu"]
    assert result[["saved_brdu", "saved_edu"]].values.tolist() == [[0, 1], [0, 1]]


@pytest.mark.parametrize("identity", [(-1, 0), (0, 2), (0,), None])
def test_training_miscalls_reject_broader_saved_identity_filters(identity):
    predictions = pd.DataFrame(
        {
            "cell": ["run|1"],
            "label": [1],
            "tranche": ["pixel"],
            "split": ["training"],
            "p_brdu": [0.9],
            "pred_brdu": [True],
            "p_edu": [0.1],
            "pred_edu": [False],
        }
    )

    with pytest.raises(ValueError, match="identity"):
        training_miscalls(
            predictions,
            {"run|1": (0, 0)},
            split="all",
            target="either",
            identity=identity,
        )


def test_training_gallery_uses_live_identity_and_requested_split():
    predictions = pd.DataFrame(
        {
            "cell": ["run|1", "run|2", "run|3", "run|4"],
            "tranche": ["cortex", "cortex", "pixel", "pixel"],
            "split": ["training", "training", "validation", "training"],
        }
    )
    live = {
        "run|1": (0, 0),
        "run|2": (0, 1),
        "run|3": (1, 0),
        "run|4": (1, 1),
        "run|5": (0, 0),
    }

    neither = training_gallery(predictions, live, split="training", identity=(0, 0))
    assert neither.cell.tolist() == ["run|1"]
    assert neither.iloc[0][["saved_brdu", "saved_edu"]].tolist() == [0, 0]
    assert training_gallery(
        predictions, live, split="all", identity=(1, 0)
    ).cell.tolist() == ["run|3"]

    corrected = {**live, "run|1": (0, 1)}
    assert training_gallery(
        predictions, corrected, split="training", identity=(0, 0)
    ).empty
    assert training_gallery(
        predictions, corrected, split="training", identity=(0, 1)
    ).cell.tolist() == ["run|1", "run|2"]


@pytest.mark.parametrize("identity", [(-1, 0), (0, 2), (0,), None])
def test_training_gallery_rejects_broader_identity_filters(identity):
    predictions = pd.DataFrame(
        {"cell": ["run|1"], "tranche": ["pixel"], "split": ["training"]}
    )

    with pytest.raises(ValueError, match="identity"):
        training_gallery(
            predictions, {"run|1": (0, 0)}, split="training", identity=identity
        )


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ((0, 0), (0, 1)),
        ((0, 1), (0, 0)),
        ((1, 0), (1, 1)),
        ((1, 1), (1, 0)),
    ],
)
def test_toggle_edu_preserves_brdu(before, after):
    assert toggle_edu(before) == after


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ((0, 0), (1, 0)),
        ((0, 1), (1, 1)),
        ((1, 0), (0, 0)),
        ((1, 1), (0, 1)),
    ],
)
def test_toggle_brdu_preserves_edu(before, after):
    assert toggle_brdu(before) == after


def test_gallery_card_state_changes_without_replacing_its_loaded_crop():
    image = "data:image/png;base64,resident"
    card = {
        "kind": "toggle",
        "marker": "edu",
        "cell": "run|7",
        "label": "Cell 7",
        "image": image,
    }

    update_gallery_card(card, (1, 1), (1, 0))

    assert card["image"] is image
    assert card["state"] == "BrdU+ / EdU+"
    assert card["expected"] == [1, 1]
    assert card["value"] == [1, 0]
    assert card["changed"] is True


def test_brdu_gallery_card_state_changes_without_replacing_its_loaded_crop():
    image = "data:image/png;base64,resident"
    card = {
        "kind": "toggle",
        "marker": "brdu",
        "cell": "run|7",
        "label": "Cell 7",
        "image": image,
    }

    update_gallery_card(card, (1, 0), (0, 0))

    assert card["image"] is image
    assert card["state"] == "BrdU+ / EdU−"
    assert card["expected"] == [1, 0]
    assert card["value"] == [0, 0]
    assert card["changed"] is True


def test_miscall_gallery_card_flips_only_its_marker():
    image = "data:image/png;base64,resident"
    card = {
        "kind": "miscall",
        "cell": "run|7",
        "label": "Cell 7",
        "image": image,
        "marker": "brdu",
    }

    update_gallery_card(card, (0, 1), (0, 1))

    assert card["image"] is image
    assert card["state"] == "BrdU− / EdU+"
    assert card["value"] == [1, 1]
    assert card["disabled"] is False
    assert card["action"] == "Set BrdU+"


def test_miscall_gallery_keeps_both_marker_cards_for_one_cell():
    rows = pd.DataFrame(
        {
            "cell": ["run|7", "run|7"],
            "audit_marker": ["brdu", "edu"],
        }
    )

    gallery = new_gallery_state(
        rows,
        SimpleNamespace(values={"run|7": (0, 0)}),
        kind="miscall",
        crop_kind="audit_marker",
    )

    assert gallery["item_ids"] == ["run|7:brdu", "run|7:edu"]
    assert gallery["rows"].cell.tolist() == ["run|7", "run|7"]


def test_render_edu_crop_is_one_square_panel_and_ignores_other_channels():
    labels = np.zeros((8, 8), dtype=np.uint32)
    labels[2:6, 2:6] = 7
    mask = labels == 7
    crop = CellCrop(
        labels=labels,
        mask=mask,
        neighbor_mask=np.zeros_like(mask),
        channels={
            "edu": np.arange(64, dtype=np.uint16).reshape(8, 8),
            "pi": np.empty((0, 0)),
            "brdu": np.empty((0, 0)),
        },
    )

    result = Image.open(io.BytesIO(render_edu_crop(crop, display_max=63)))

    assert result.mode == "RGB"
    assert result.size == (8, 8)
    np.testing.assert_array_equal(np.asarray(result)[2, 2], [0, 255, 255])


def test_render_brdu_crop_is_one_square_panel_and_ignores_other_channels():
    labels = np.zeros((8, 8), dtype=np.uint32)
    labels[2:6, 2:6] = 7
    mask = labels == 7
    crop = CellCrop(
        labels=labels,
        mask=mask,
        neighbor_mask=np.zeros_like(mask),
        channels={
            "brdu": np.arange(64, dtype=np.uint16).reshape(8, 8),
            "pi": np.empty((0, 0)),
            "edu": np.empty((0, 0)),
        },
    )

    result = Image.open(io.BytesIO(render_brdu_crop(crop, display_max=63)))

    assert result.mode == "RGB"
    assert result.size == (8, 8)
    np.testing.assert_array_equal(np.asarray(result)[2, 2], [0, 255, 255])


@pytest.mark.parametrize(
    ("prefetch", "expected"),
    [
        (prefetch_edu_crop, ("edu", 20)),
        (prefetch_brdu_crop, ("brdu", 20)),
    ],
)
def test_prefetch_populates_the_requested_crop_cache(prefetch, expected):
    class Source:
        def __init__(self):
            self.calls = []

        def edu_cell_crop(self, label):
            self.calls.append(("edu", label))

        def brdu_cell_crop(self, label):
            self.calls.append(("brdu", label))

    source = Source()

    prefetch(source, "full-run2|20")

    assert source.calls == [expected]


@pytest.mark.parametrize(
    "crop_method",
    [FieldSource.edu_cell_crop, FieldSource.brdu_cell_crop],
)
def test_gallery_crop_cache_retains_current_and_prefetched_pages(
    crop_method, monkeypatch
):
    labels = np.zeros((8, 8), dtype=np.uint32)
    labels[2:6, 2:6] = 7
    mask = labels == 7
    crop = CellCrop(
        labels=labels,
        mask=mask,
        neighbor_mask=np.zeros_like(mask),
        channels={
            "pi": np.ones((8, 8), dtype=np.uint16),
            "edu": np.ones((8, 8), dtype=np.uint16),
            "brdu": np.ones((8, 8), dtype=np.uint16),
        },
    )
    monkeypatch.setattr(
        "dino.crop_rendering.render_crop",
        lambda crop, *, cell, display_max: Image.new("RGB", crop.mask.shape),
    )

    class Source:
        def __init__(self):
            self.reads = np.zeros(48, dtype=np.uint8)

        def _read_cell_crop(self, label):
            self.reads[label] += 1
            return crop, 0, {"display_max": {"pi": 1, "edu": 1, "brdu": 1}}

    source = Source()
    crop_method.cache_clear()
    try:
        for label in range(48):
            crop_method(source, label)
        for label in range(24):
            crop_method(source, label)

        np.testing.assert_array_equal(source.reads, np.ones(48, dtype=np.uint8))
    finally:
        crop_method.cache_clear()


def test_prediction_pointer_binds_independent_review_manifest(tmp_path):
    from dino.artifacts import file_sha256
    model = tmp_path / 'latest'
    cortex = model / 'cortex-pred'
    cortex.mkdir(parents=True)
    (cortex / 'complete.json').write_text(json.dumps({'model_sha256': 'latest-sha', 'cells': 100}))
    review = model / 'labeled-review/complete.json'
    review.parent.mkdir()
    review.write_text(json.dumps({'model_sha256': 'latest-sha'}))
    pointer = tmp_path / 'current.json'
    pointer.write_text(json.dumps({'artifact': 'dino-review-current-v1', 'model_dir': str(model),
                                   'prediction_dir': str(cortex / 'predictions'),
                                   'model_sha256': 'latest-sha', 'cells': 100,
                                   'review_manifest': str(review), 'review_manifest_sha256': file_sha256(review)}))
    _, _, _, contract = load_prediction_pointer(pointer)
    assert contract['review_manifest'] == str(review)
    review.write_text(json.dumps({'model_sha256': 'different'}))
    with pytest.raises(ValueError, match='labeled review manifest'):
        load_prediction_pointer(pointer)
    record = json.loads(pointer.read_text())
    record['review_manifest_sha256'] = file_sha256(review)
    pointer.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='different classifier'):
        load_prediction_pointer(pointer)
