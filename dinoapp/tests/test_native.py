"""Behavior checks for native field sampling, picking, and barrage edits."""

import base64
import io
import json
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from dino_markers import (
    DEFAULT_CONTRACT,
    CORTEX_RESCORE,
    INFERENCE,
    ContractMismatchError,
    MarkerContract,
    UnsupportedArtifactError,
    config_to_json,
    default_config,
    parse_config,
)
from PIL import Image
from skimage.segmentation import find_boundaries
from test_characterization_review import build_corrected_barrage, build_native_root

from dinoapp.native import (
    AnnotationFiles,
    Field,
    FieldSource,
    classifier_caption,
    code_colors,
    load_model_calls,
    load_prediction_pointer,
    pick_cell,
    prediction_contract,
    prefetch_marker_crop,
    render_field,
    render_marker_crop,
    sample_center,
    training_gallery,
    training_miscalls,
)
from dinoapp.label_store import LabelConflictError, load_labels, update_label
from dino.zarr_source import CellCrop
from dinoapp.native_app import new_gallery_state, update_gallery_card

CONFIG3 = parse_config(
    {
        "version": 1,
        "reference": {"key": "dapi", "label": "DAPI"},
        "markers": [
            {"key": "edu", "label": "EdU", "color": "#00ff00"},
            {"key": "brdu", "label": "BrdU", "color": "#ff00ff"},
            {"key": "pcna", "label": "PCNA", "color": "#0000ff"},
        ],
        "presentation": {
            "crop_panels": ["dapi", "pcna", "edu", "brdu"],
            "negative_color": [0.25, 0.25, 0.25],
            "uncalled_color": "#999999",
        },
    },
    source="test",
)
REORDERED = parse_config(
    {
        "version": 1,
        "reference": {"key": "pi", "label": "PI"},
        "markers": [
            {"key": "edu", "label": "EdU", "color": "#00ff00"},
            {"key": "brdu", "label": "BrdU", "color": "#ff00ff"},
        ],
        "presentation": {
            "crop_panels": ["pi", "edu", "brdu"],
            "negative_color": [0.25, 0.25, 0.25],
            "uncalled_color": "#999999",
        },
    },
    source="test",
)


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
    source.config = default_config()
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


def test_existing_labels_return_to_their_barrage_and_new_labels_go_to_default(tmp_path):
    old = tmp_path / "old" / "labels.csv"
    default = tmp_path / "pixel" / "labels.csv"
    update_label(old, "full-run2|10", (1, 0), contract=DEFAULT_CONTRACT)
    files = AnnotationFiles((old, default), default, contract=DEFAULT_CONTRACT)
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
    files = AnnotationFiles((path,), path, contract=DEFAULT_CONTRACT)
    edit = files.assign("full-run2|5", (1, 1), expected=None)
    files.undo(edit)
    assert load_labels(path).empty
    edit = files.assign("full-run2|5", (0, 1), expected=None)
    update_label(path, "full-run2|5", (1, 0), contract=DEFAULT_CONTRACT)
    with pytest.raises(LabelConflictError):
        files.undo(edit)
    assert load_labels(path).iloc[0][["brdu", "edu"]].tolist() == [1, 0]


def test_duplicate_tranche_ownership_fails_without_writing(tmp_path):
    paths = (tmp_path / "one.csv", tmp_path / "two.csv")
    for p in paths:
        update_label(p, "full-run2|1", (0, 1), contract=DEFAULT_CONTRACT)
    original = [p.read_bytes() for p in paths]
    with pytest.raises(ValueError, match="multiple"):
        AnnotationFiles(paths, paths[-1], contract=DEFAULT_CONTRACT)
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
        predictions, live, contract=DEFAULT_CONTRACT, split="training", target=None, identity=(0, 0)
    )

    assert result.cell.tolist() == ["run|1"]
    assert result.iloc[0][["saved_brdu", "saved_edu"]].tolist() == [0, 0]
    assert result.iloc[0].audit_score == pytest.approx(0.9)
    assert result.iloc[0].audit_marker == "brdu"
    corrected = {**live, "run|1": (1, 0)}
    assert training_miscalls(
        predictions,
        corrected,
        contract=DEFAULT_CONTRACT, split="training",
        target=None,
        identity=(0, 0),
    ).empty
    edu = training_miscalls(
        predictions, live, contract=DEFAULT_CONTRACT, split="validation", target="edu", identity=(0, 0)
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
        contract=DEFAULT_CONTRACT, split="training",
        target=None,
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
        predictions, live, contract=DEFAULT_CONTRACT, split="all", target=None, identity=(0, 1)
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
            contract=DEFAULT_CONTRACT, split="all",
            target=None,
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

    neither = training_gallery(predictions, live, contract=DEFAULT_CONTRACT, split="training", identity=(0, 0))
    assert neither.cell.tolist() == ["run|1"]
    assert neither.iloc[0][["saved_brdu", "saved_edu"]].tolist() == [0, 0]
    assert training_gallery(
        predictions, live, contract=DEFAULT_CONTRACT, split="all", identity=(1, 0)
    ).cell.tolist() == ["run|3"]

    corrected = {**live, "run|1": (0, 1)}
    assert training_gallery(
        predictions, corrected, contract=DEFAULT_CONTRACT, split="training", identity=(0, 0)
    ).empty
    assert training_gallery(
        predictions, corrected, contract=DEFAULT_CONTRACT, split="training", identity=(0, 1)
    ).cell.tolist() == ["run|1", "run|2"]


@pytest.mark.parametrize("identity", [(-1, 0), (0, 2), (0,), None])
def test_training_gallery_rejects_broader_identity_filters(identity):
    predictions = pd.DataFrame(
        {"cell": ["run|1"], "tranche": ["pixel"], "split": ["training"]}
    )

    with pytest.raises(ValueError, match="identity"):
        training_gallery(
            predictions, {"run|1": (0, 0)}, contract=DEFAULT_CONTRACT, split="training", identity=identity
        )


def test_gallery_card_state_changes_without_replacing_its_loaded_crop():
    image = "data:image/png;base64,resident"
    card = {
        "kind": "toggle",
        "marker": "edu",
        "cell": "run|7",
        "label": "Cell 7",
        "image": image,
    }

    update_gallery_card(card, (1, 1), (1, 0), config=default_config())

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

    update_gallery_card(card, (1, 0), (0, 0), config=default_config())

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

    update_gallery_card(card, (0, 1), (0, 1), config=default_config())

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


def _marker_crop(marker: str) -> CellCrop:
    labels = np.zeros((8, 8), dtype=np.uint32)
    labels[2:6, 2:6] = 7
    mask = labels == 7
    channels = {key: np.empty((0, 0)) for key in ("pi", "edu", "brdu", "pcna")}
    channels[marker] = np.arange(64, dtype=np.uint16).reshape(8, 8)
    return CellCrop(labels=labels, mask=mask, neighbor_mask=np.zeros_like(mask), channels=channels)


def test_render_marker_crop_is_one_square_panel_and_ignores_other_channels():
    result = Image.open(io.BytesIO(render_marker_crop(_marker_crop("pcna"), "pcna", 63, label="PCNA")))

    assert result.mode == "RGB"
    assert result.size == (8, 8)
    np.testing.assert_array_equal(np.asarray(result)[2, 2], [0, 255, 255])


def test_render_marker_crop_errors_name_the_marker_label():
    crop = _marker_crop("pcna")
    with pytest.raises(ValueError, match="PCNA channel/mask shape mismatch"):
        render_marker_crop(crop, "edu", 63, label="PCNA")
    with pytest.raises(ValueError, match="PCNA display maximum"):
        render_marker_crop(crop, "pcna", 0.0, label="PCNA")


def test_prefetch_populates_the_requested_crop_cache():
    class Source:
        def __init__(self):
            self.calls = []

        def marker_crop(self, label, marker):
            self.calls.append((label, marker))

    source = Source()

    prefetch_marker_crop(source, "full-run2|20", "pcna")

    assert source.calls == [(20, "pcna")]


def test_gallery_crop_cache_retains_current_and_prefetched_pages(monkeypatch):
    crop = _marker_crop("edu")
    crop.channels.update(brdu=np.ones((8, 8), dtype=np.uint16), edu=np.ones((8, 8), dtype=np.uint16))

    class Source:
        config = default_config()

        def __init__(self):
            self.reads = np.zeros(48, dtype=np.uint8)

        def _read_cell_crop(self, label):
            self.reads[label] += 1
            return crop, 0, {"display_max": {"pi": 1, "edu": 1, "brdu": 1}}

    source = Source()
    FieldSource.marker_crop.cache_clear()
    try:
        for label in range(48):
            for marker in ("edu", "brdu"):
                FieldSource.marker_crop(source, label, marker)
        for label in range(24):
            for marker in ("edu", "brdu"):
                FieldSource.marker_crop(source, label, marker)

        np.testing.assert_array_equal(source.reads, np.full(48, 2, dtype=np.uint8))
    finally:
        FieldSource.marker_crop.cache_clear()


def test_marker_crop_rejects_the_reference_and_unknown_markers():
    class Source:
        config = default_config()

    source = Source()
    for marker in ("pi", "dapi", "combined"):
        with pytest.raises(ValueError, match="Unexpected crop marker"):
            FieldSource.marker_crop(source, 1, marker)


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


def _predictions(markers, labels=(1, 2, 4, 5)):
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"label": np.asarray(labels, dtype=np.int64)})
    frame["cell"] = [f"run|{label}" for label in labels]
    for marker in markers:
        probability = rng.uniform(0.05, 0.95, len(labels))
        frame[f"p_{marker}"] = probability
        frame[marker] = probability >= 0.5
    return frame


def test_model_calls_encode_the_first_marker_as_the_most_significant_bit():
    frame = _prod2_predictions()

    model = load_model_calls(
        frame, dataset="Prod2", contract=DEFAULT_CONTRACT, cutoffs={"brdu": 0.5, "edu": 0.5},
        cell_prefix="full-run2",
    )

    assert model.calls.dtype == np.uint8
    np.testing.assert_array_equal(model.calls, [4, 0, 1, 4, 2, 3])
    assert model.uncertainty.dtype == np.float32
    expected = 1 / np.minimum(np.abs(frame.p_brdu - 0.5), np.abs(frame.p_edu - 0.5))
    np.testing.assert_allclose(model.uncertainty[[1, 2, 4, 5]], expected, rtol=1e-6)
    assert np.isnan(model.uncertainty[[0, 3]]).all()
    assert model.uncertainty_log_vmax == pytest.approx(float(np.quantile(np.log1p(expected), 0.999)))


def test_model_calls_for_three_markers_follow_contract_order_and_uncalled_code():
    contract = CONFIG3.contract
    frame = _predictions(contract.markers, labels=(2, 3, 6, 7))
    frame["edu"] = [True, False, False, True]
    frame["brdu"] = [False, True, False, True]
    frame["pcna"] = [True, True, False, False]
    for marker in contract.markers:
        frame[f"p_{marker}"] = np.where(frame[marker], 0.8, 0.1)

    model = load_model_calls(
        frame, dataset="Test", contract=contract, cutoffs=dict.fromkeys(contract.markers, 0.5),
        cell_prefix="run",
    )

    assert model.calls.dtype == np.uint8
    np.testing.assert_array_equal(model.calls, [8, 8, 5, 3, 8, 8, 0, 6])
    np.testing.assert_allclose(model.uncertainty[[2, 3, 6, 7]], [1 / 0.3, 1 / 0.3, 1 / 0.4, 1 / 0.3], rtol=1e-6)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda f: f.drop(columns=["p_pcna"]), "missing columns"),
        (lambda f: f.assign(label=[1, 1, 4, 5]), "unique and positive"),
        (lambda f: f.assign(label=[0, 2, 4, 5], cell=["run|0", "run|2", "run|4", "run|5"]), "unique and positive"),
        (lambda f: f.assign(cell=["run|1", "run|2", "run|4", "run|6"]), "cell identities"),
        (lambda f: f.assign(pcna=[True, None, True, False]), "complete and binary"),
        (lambda f: f.assign(pcna=[1, 2, 0, 1]), "complete and binary"),
        (lambda f: f.assign(p_pcna=[0.1, 1.2, 0.3, 0.4]), "finite and within"),
        (lambda f: f.assign(p_pcna=[0.1, np.nan, 0.3, 0.4]), "finite and within"),
        (lambda f: f.iloc[:0], "empty"),
    ],
)
def test_model_calls_reject_invalid_frames_with_the_dataset_name(mutate, message):
    contract = CONFIG3.contract
    frame = mutate(_predictions(contract.markers))

    with pytest.raises(ValueError, match=rf"^Prod9.*{message}"):
        load_model_calls(
            frame, dataset="Prod9", contract=contract, cutoffs=dict.fromkeys(contract.markers, 0.5),
            cell_prefix="run",
        )


def test_model_calls_require_cutoffs_for_exactly_the_contract_markers():
    frame = _predictions(DEFAULT_CONTRACT.markers)
    for cutoffs in ({"brdu": 0.5}, {"brdu": 0.5, "edu": 0.5, "pcna": 0.5}):
        with pytest.raises(ValueError, match="same nonempty markers"):
            load_model_calls(frame, dataset="Prod9", contract=DEFAULT_CONTRACT, cutoffs=cutoffs, cell_prefix="run")


def test_model_calls_without_a_finite_score_need_a_given_display_scale():
    frame = _predictions(DEFAULT_CONTRACT.markers)
    for marker in DEFAULT_CONTRACT.markers:
        frame[f"p_{marker}"] = 0.5
    arguments = dict(dataset="Prod9", contract=DEFAULT_CONTRACT, cutoffs={"brdu": 0.5, "edu": 0.5}, cell_prefix="run")

    with pytest.raises(ValueError, match="^Prod9.*No finite model uncertainty scores"):
        load_model_calls(frame, **arguments)

    model = load_model_calls(frame, uncertainty_log_vmax=4.5, **arguments)
    assert model.uncertainty_log_vmax == 4.5
    assert np.isinf(model.uncertainty[frame.label]).all()


def test_prediction_contract_reads_unversioned_manifests_as_the_default():
    assert prediction_contract({"model_sha256": "m"}, where="m.json") == DEFAULT_CONTRACT
    with pytest.raises(UnsupportedArtifactError, match="m.json: unversioned prediction manifest carries marker_contract"):
        prediction_contract({"marker_contract": DEFAULT_CONTRACT.to_json()}, where="m.json")


def test_prediction_contract_reads_inference_and_rescore_versions():
    reordered = MarkerContract("pi", ("edu", "brdu"))
    assert prediction_contract({"artifact": "dino-zarr-infer-v2"}, where="m.json") == DEFAULT_CONTRACT
    assert prediction_contract({"artifact": "dino-cortex-rescore-v1"}, where="m.json") == DEFAULT_CONTRACT
    for kind in (INFERENCE, CORTEX_RESCORE):
        assert prediction_contract(kind.stamp({}, reordered), where="m.json") == reordered
        assert prediction_contract(kind.stamp({}, CONFIG3.contract), where="m.json") == CONFIG3.contract


@pytest.mark.parametrize("artifact", [None, "dino-zarr-infer-v1"])
def test_prediction_contract_rejects_other_versions(artifact):
    with pytest.raises(UnsupportedArtifactError):
        prediction_contract({"artifact": artifact}, where="m.json")


def test_classifier_caption_lists_each_marker_cutoff_in_contract_order():
    assert classifier_caption("abcdef012345", default_config(), {"brdu": 0.25, "edu": 0.75}) == (
        "Classifier abcdef012345 · BrdU cutoff 0.2500000000 · EdU cutoff 0.7500000000"
    )
    assert classifier_caption("abc", CONFIG3, {"edu": 0.1, "brdu": 0.2, "pcna": 0.3}) == (
        "Classifier abc · EdU cutoff 0.1000000000 · BrdU cutoff 0.2000000000 · PCNA cutoff 0.3000000000"
    )


def test_code_colors_blend_marker_colors_per_distinct_code():
    codes = np.array([[0, 3], [5, 7], [8, 3]], dtype=np.uint8)

    colors = code_colors(CONFIG3, codes)

    assert colors.shape == (3, 2, 3) and colors.dtype == np.float64
    np.testing.assert_array_equal(colors[0, 0], [0.25, 0.25, 0.25])
    np.testing.assert_array_equal(colors[0, 1], [1, 0, 1])  # BrdU magenta + PCNA blue
    np.testing.assert_array_equal(colors[1, 0], [0, 1, 1])  # EdU green + PCNA blue
    np.testing.assert_array_equal(colors[1, 1], [1, 1, 1])  # every marker, clipped
    np.testing.assert_array_equal(colors[2, 0], [0.6, 0.6, 0.6])
    np.testing.assert_array_equal(colors[2, 1], colors[0, 1])


def _field(images, mask, *, borders=None) -> Field:
    ids, inverse = np.unique(mask, return_inverse=True)
    return Field(
        images=images, mask=mask, ids=ids, inverse=inverse.reshape(mask.shape),
        borders=np.zeros(mask.shape, dtype=bool) if borders is None else borders,
        origin_xy_um=np.zeros(2), scale_xy_um=np.array([10.0, 10.0]), center_xy_um=np.zeros(2),
        z_um=0.0, z_index=0, seed=None, position=None, contours=[], boundary_info={},
    )


def _render(field, calls, config, mode, contrast, **kwargs) -> np.ndarray:
    url = render_field(
        field, calls, config=config, mode=mode, selected=None, border_alpha=kwargs.pop("border_alpha", 0.0),
        boundary_alpha=0.0, contrast=contrast, **kwargs,
    )
    prefix = "data:image/png;base64,"
    assert url.startswith(prefix)
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(url[len(prefix):]))).convert("RGB"))


def _three_marker_images(shape=(16, 16)):
    return {
        "dapi": np.full(shape, 0.3), "edu": np.full(shape, 0.5),
        "brdu": np.full(shape, 0.8), "pcna": np.full(shape, 0.7),
    }


CONTRAST3 = dict.fromkeys(("dapi", "edu", "brdu", "pcna"), (0.0, 1.0))


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("Cell identities", (0.8, 0.5, 1.0)),  # blue clips: BrdU magenta 0.8 + PCNA blue 0.7
        ("dapi", (0.3, 0.3, 0.3)),
        ("pcna", (0.0, 0.0, 0.7)),
    ],
)
def test_three_marker_field_composite_clips_and_views_each_channel(mode, expected):
    field = _field(_three_marker_images(), np.zeros((16, 16), dtype=np.uint32))

    pixels = _render(field, np.array([8], dtype=np.uint8), CONFIG3, mode, CONTRAST3)

    np.testing.assert_allclose(pixels[:8, :8].reshape(-1, 3), np.tile(np.array(expected) * 255, (64, 1)), atol=1)


def test_field_composite_normalizes_with_each_channel_contrast():
    images = _three_marker_images()
    contrast = {**CONTRAST3, "edu": (0.25, 0.75), "pcna": (0.7, 0.9)}

    pixels = _render(_field(images, np.zeros((16, 16), dtype=np.uint32)), np.array([8], dtype=np.uint8),
                     CONFIG3, "Cell identities", contrast)

    np.testing.assert_allclose(pixels[0, 0], np.array([0.8, 0.5, 0.8]) * 255, atol=1)


def test_identity_borders_take_each_codes_blended_color():
    mask = np.zeros((16, 16), dtype=np.uint32)
    mask[3:12, 3:12] = 1
    borders = find_boundaries(mask, mode="inner") & (mask > 0)
    field = _field(_three_marker_images(), mask, borders=borders)

    pixels = _render(field, np.array([8, 5], dtype=np.uint8), CONFIG3, "Cell identities", CONTRAST3,
                     border_alpha=1.0)

    np.testing.assert_allclose(pixels[3, 6], [0, 255, 255], atol=1)  # EdU + PCNA
    np.testing.assert_allclose(pixels[7, 7], np.array([0.8, 0.5, 1.0]) * 255, atol=1)


def test_field_views_reject_unknown_modes_and_invalid_contrast_by_label():
    field = _field(_three_marker_images(), np.zeros((16, 16), dtype=np.uint32))
    calls = np.array([8], dtype=np.uint8)

    with pytest.raises(ValueError, match="Unknown field view"):
        _render(field, calls, CONFIG3, "BrdU", CONTRAST3)
    with pytest.raises(ValueError, match="Invalid contrast limits for PCNA"):
        _render(field, calls, CONFIG3, "Cell identities", {**CONTRAST3, "pcna": (1.0, 1.0)})


def test_model_uncertainty_view_colors_scored_cells_and_requires_a_scale():
    mask = np.zeros((16, 16), dtype=np.uint32)
    mask[3:12, 3:12] = 1
    field = _field(_three_marker_images(), mask)
    calls = np.array([8, 5], dtype=np.uint8)

    pixels = _render(field, calls, CONFIG3, "Model uncertainty", CONTRAST3,
                     uncertainty=np.array([np.nan, 3.0], dtype=np.float32), uncertainty_log_vmax=np.log1p(3.0))

    np.testing.assert_allclose(pixels[7, 7], np.array([0.993248, 0.906157, 0.143936]) * 255, atol=1)
    np.testing.assert_allclose(pixels[0, 0], np.array([0.8, 0.5, 1.0]) * 255, atol=1)
    with pytest.raises(ValueError, match="display maximum"):
        _render(field, calls, CONFIG3, "Model uncertainty", CONTRAST3, uncertainty=np.array([np.nan, 3.0]))


def test_field_calls_encode_saved_labels_for_every_marker_and_fill_the_uncalled_code():
    source = object.__new__(FieldSource)
    source.config = CONFIG3
    source.calls = np.array([8, 6, 1], dtype=np.uint8)
    field = SimpleNamespace(ids=np.array([0, 1, 2, 9], dtype=np.uint32))
    annotations = SimpleNamespace(values={"full-run2|2": (1, 0, 1), "full-run2|9": (0, 1, 1)})

    calls = source.field_calls(field, annotations)

    assert calls.dtype == np.uint8
    np.testing.assert_array_equal(calls, [8, 6, 5, 3])


def test_annotation_directories_name_the_directories_each_adapter_annotates(tmp_path):
    from dinoapp import prod2_2, prod3, prod3_2

    assert FieldSource.annotation_directories(tmp_path) == tuple(
        tmp_path / name for name in ("barrage-cortex", "barrage-vz-svz", "barrage-pixel")
    )
    assert FieldSource.annotation_directories()[-1].name == "barrage-pixel"
    assert prod3.FieldSource.annotation_directories() == (
        prod3.REVIEW_ROOT / "barrage-cortex-v1", prod3.REVIEW_ROOT / "barrage-vz-svz-v1",
    )
    assert prod3_2.FieldSource.annotation_directories(tmp_path) == (tmp_path / "dino-prod3-2" / "barrage-cortex-v1",)
    assert prod3_2.FieldSource.annotation_directories() == (prod3_2.REVIEW_ROOT / "barrage-cortex-v1",)
    assert prod2_2.FieldSource.annotation_directories(tmp_path) == (tmp_path / "dino-prod2-2" / "barrage-cortex-v1",)
    assert prod2_2.FieldSource.annotation_directories() == (prod2_2.REVIEW_ROOT / "barrage-cortex-v1",)


def _prod2_predictions():
    return pd.DataFrame({
        "label": np.array([1, 2, 4, 5], dtype=np.int64),
        "cell": [f"full-run2|{label}" for label in (1, 2, 4, 5)],
        "brdu": [False, False, True, True], "edu": [False, True, False, True],
        "p_brdu": [0.1, 0.2, 0.8, 0.9], "p_edu": [0.3, 0.7, 0.2, 0.6],
    })


def test_prod2_adapter_reads_its_calls_config_contrast_and_caption(tmp_path):
    root = build_native_root(tmp_path, _prod2_predictions())

    source = FieldSource(root)

    assert source.config == default_config()
    np.testing.assert_array_equal(source.calls, [4, 0, 1, 4, 2, 3])
    assert source.uncertainty_log_vmax > 0
    assert source.contrast == {"pi": (0.0, 1.0), "brdu": (0.0, 1.0), "edu": (0.0, 1.0)}
    assert source.annotation_paths == tuple(
        root / name / "labels.csv" for name in ("barrage-cortex", "barrage-vz-svz", "barrage-pixel")
    )
    assert source.classifier_caption == (
        f"Classifier {'a' * 12} · BrdU cutoff 0.5000000000 · EdU cutoff 0.5000000000"
    )
    assert source.training_predictions.cell.tolist() == _prod2_predictions().cell.tolist()


def _write_markers(root: Path, config) -> None:
    (root / "barrage-pixel").mkdir(parents=True, exist_ok=True)
    (root / "barrage-pixel" / "markers.json").write_text(json.dumps(config_to_json(config)))


def test_prod2_adapter_rejects_an_annotation_directory_with_other_channels(tmp_path):
    root = build_native_root(tmp_path, _prod2_predictions())
    _write_markers(root, CONFIG3)

    with pytest.raises(ValueError, match="do not match the marker configuration"):
        FieldSource(root)


def test_prod2_adapter_rejects_reordered_annotations_against_its_unversioned_manifest(tmp_path):
    root = build_native_root(tmp_path, _prod2_predictions())
    _write_markers(root, REORDERED)

    with pytest.raises(ContractMismatchError, match="marker order is part of the contract"):
        FieldSource(root)


def test_prod2_adapter_rejects_a_prediction_manifest_with_another_contract(tmp_path):
    root = build_native_root(tmp_path, _prod2_predictions())
    manifest = tmp_path / "model-v1" / "cortex-pred" / "complete.json"
    record = INFERENCE.stamp(json.loads(manifest.read_text()), MarkerContract("pi", ("edu", "brdu")))
    manifest.write_text(json.dumps(record))

    with pytest.raises(ContractMismatchError):
        FieldSource(root)


def test_prod2_adapter_accepts_a_versioned_manifest_of_its_contract(tmp_path):
    root = build_native_root(tmp_path, _prod2_predictions())
    manifest = tmp_path / "model-v1" / "cortex-pred" / "complete.json"
    manifest.write_text(json.dumps(CORTEX_RESCORE.stamp(json.loads(manifest.read_text()), DEFAULT_CONTRACT)))

    assert FieldSource(root).config.contract == DEFAULT_CONTRACT


def test_prod3_marker_crop_renders_one_marker_with_its_display_maximum_and_label():
    from dinoapp import prod3

    crop = _marker_crop("edu")
    source = object.__new__(prod3.FieldSource)
    source.config = default_config()
    source.crop_contract = {"display_max": {"pi": 1.0, "edu": 63.0, "brdu": 63.0}}
    source._read_cell_crop = lambda label: (crop, 3)

    image, plane = source.marker_crop(5, "edu")

    assert plane == 3
    assert image == render_marker_crop(crop, "edu", 63.0, label="EdU")
    for marker in ("pi", "dapi", "combined"):
        with pytest.raises(ValueError, match="Unexpected crop marker"):
            source.marker_crop(5, marker)


def test_corrected_barrage_rejects_a_display_source_of_another_contract(tmp_path, monkeypatch):
    from dinoapp.corrected_barrage import CorrectedBarrageSource

    root, _ = build_corrected_barrage(tmp_path, monkeypatch)
    (tmp_path / "corrected-source" / "markers.json").write_text(json.dumps(config_to_json(REORDERED)))

    with pytest.raises(ContractMismatchError, match="marker order is part of the contract"):
        CorrectedBarrageSource(root)
