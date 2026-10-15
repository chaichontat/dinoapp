"""Connected views resolve bounded identities without changing review rosters."""

import numpy as np
import pandas as pd
import pytest

from dinoapp.prod3 import FieldSource
from dinoapp.sources import NativeSource


@pytest.fixture
def native_source(tmp_path):
    reader = object.__new__(FieldSource)
    reader.props_path = tmp_path / "cells.parquet"
    pd.DataFrame(
        dict(label=[10, 20], plane_z=[0, 1], centroid_y=[4.0, 8.0], centroid_x=[5.0, 9.0])
    ).to_parquet(reader.props_path)
    reader.shape = (2, 20, 20)
    reader.origin = np.array([100.0, 200.0, 300.0])
    reader.scale = np.array([2.0, 3.0, 4.0])
    reader.training_predictions = pd.DataFrame(
        dict(cell=["prod3-realigned-v1|10"], split=["training"], tranche=["cortex"])
    )
    reader.annotation_paths = (tmp_path / "labels.csv",)
    reader.default_annotation_path = reader.annotation_paths[0]
    reader.cell_crop = lambda label: (str(label).encode(), int(reader.cell_geometry(label).plane_z))
    reader.edu_cell_crop = reader.brdu_cell_crop = reader.cell_crop
    return NativeSource(reader, label_root=tmp_path)


def test_valid_cell_outside_gallery_resolves_without_expanding_roster(native_source):
    assert native_source.crop("prod3-realigned-v1|20", "combined").detail == "Z 102.0 µm"
    assert native_source.review_rows(dataset="All datasets").cell.tolist() == ["prod3-realigned-v1|10"]


@pytest.mark.parametrize("cell", ["other|20", "prod3-realigned-v1|020", "prod3-realigned-v1|0", "prod3-realigned-v1|+20", "prod3-realigned-v1|20|20"])
def test_wrong_native_identity_is_rejected_before_geometry_read(native_source, monkeypatch, cell):
    def forbidden(label):
        pytest.fail("invalid identity reached geometry")

    monkeypatch.setattr(native_source.field_source, "cell_geometry", forbidden)
    with pytest.raises(ValueError, match="identity"):
        native_source.crop(cell, "combined")


@pytest.mark.parametrize("labels", [[], [20, 20]])
def test_missing_or_duplicate_geometry_is_rejected(native_source, labels):
    reader = native_source.field_source
    pd.DataFrame(
        dict(
            label=labels,
            plane_z=[1] * len(labels),
            centroid_y=[8.0] * len(labels),
            centroid_x=[9.0] * len(labels),
        )
    ).to_parquet(reader.props_path)
    with pytest.raises(ValueError, match="one saved crop geometry"):
        native_source.crop("prod3-realigned-v1|20", "combined")


def test_geometry_read_is_filtered_to_requested_label(native_source, monkeypatch):
    import dinoapp.prod3 as prod3

    original = prod3.pq.read_table
    reads = []

    def read(path, **kwargs):
        reads.append(kwargs)
        return original(path, **kwargs)

    monkeypatch.setattr(prod3.pq, "read_table", read)
    native_source.field_source.cell_geometry(20)
    assert reads == [
        dict(columns=["label", "plane_z", "centroid_y", "centroid_x"], filters=[("label", "=", 20)])
    ]


@pytest.fixture
def workspace_review(tmp_path, monkeypatch):
    import io
    from types import SimpleNamespace
    from PIL import Image
    from streamlit.testing.v1 import AppTest
    from dinoapp import app, native_app

    reader = object.__new__(FieldSource)
    reader.shape = (3, 64, 64)
    reader.origin = np.array([100.0, 0.0, 0.0])
    reader.scale = np.array([2.0, 10.0, 10.0])
    reader.default_z = 100.0
    reader.props_path = tmp_path / "cells.parquet"
    pd.DataFrame(
        dict(label=[1, 2], plane_z=[0, 1], centroid_y=[27.0, 36.0], centroid_x=[27.0, 36.0])
    ).to_parquet(reader.props_path)
    reader.labels = np.zeros(reader.shape, dtype=np.uint32)
    reader.labels[:, 25:29, 25:29] = 1
    reader.labels[:, 34:38, 34:38] = 2
    reader.arrays = {
        name: np.full(reader.shape, value, dtype=np.uint16)
        for name, value in (("561", 1000), ("BrdU", 2000), ("EdU", 3000))
    }
    reader.display_pixels = SimpleNamespace(field_images=lambda images, selection: images)
    reader.contours_at = lambda zi: [np.array([[[280.0, 280.0], [300.0, 300.0]]])] * 2
    reader._field_boundary_info = lambda labels: {}
    reader.training_predictions = pd.DataFrame(
        dict(
            cell=["prod3-realigned-v1|1"],
            label=[1],
            split=["training"],
            tranche=["cortex"],
            p_brdu=[0.1],
            p_edu=[0.2],
            pred_brdu=[False],
            pred_edu=[False],
        )
    )
    reader.annotation_paths = (tmp_path / "labels.csv",)
    reader.default_annotation_path = reader.annotation_paths[0]
    reader.model_version = "model-1"
    reader.classifier_caption = "Classifier model-1 · BrdU cutoff 0.5 · EdU cutoff 0.5"
    reader.calls = np.array([4, 0, 3], dtype=np.uint8)
    reader.uncertainty = np.array([np.nan, 1.0, 2.0])
    reader.uncertainty_log_vmax = 3.0
    reader.contrast = np.array([[0.0, 5000.0]] * 3)

    def crop(label):
        row = reader.cell_geometry(label)
        output = io.BytesIO()
        Image.new("RGB", (8, 8), (label * 40, 0, 0)).save(output, format="PNG")
        return output.getvalue(), int(row.plane_z)

    reader.cell_crop = reader.edu_cell_crop = reader.brdu_cell_crop = crop
    monkeypatch.setattr(native_app, "get_source", lambda cls: reader)
    monkeypatch.setattr(
        native_app, "get_field", lambda cls, seed, position, z, width: reader.field(seed, position, z, width)
    )
    monkeypatch.setenv("DINO_REVIEW_SOURCE", "Native fields")
    monkeypatch.setenv("DINO_NATIVE_DATASET", "Prod3")
    monkeypatch.setenv("DINO_REVIEW_LABEL_ROOT", str(tmp_path))
    app.st.cache_resource.clear()
    review = AppTest.from_file(app.__file__).run(timeout=20)
    assert not review.exception
    assert context_of(review)["workspace"]["review_source"].label_root == tmp_path
    return review, reader, tmp_path


def choose(review, label, value):
    next(widget for widget in review.selectbox if widget.label == label).select(value).run()
    assert not review.exception


def press(review, label):
    next(widget for widget in review.button if widget.label == label).click().run()
    assert not review.exception


def context_of(review):
    (context,) = review.session_state.review_contexts.values()
    return context


def test_field_queue_crop_location_round_trip_has_no_annotation_side_effects(workspace_review):
    review, reader, root = workspace_review
    next(widget for widget in review.number_input if widget.label == "Position").set_value(5).run()
    context = context_of(review)
    original = context["workspace"]["active_field"][1]
    press(review, "Review cells in this field")
    context = context_of(review)
    queue = context["workspace"]["queue"]
    assert queue["cells"] == ["prod3-realigned-v1|1", "prod3-realigned-v1|2"]
    assert queue["position"] == 0
    assert reader.training_predictions.cell.tolist() == ["prod3-realigned-v1|1"]
    press(review, "Next")
    assert queue["position"] == 1
    press(review, "Show in field")
    context = context_of(review)
    located = context["workspace"]["located_field"][1]
    assert located.z_index == 1
    assert located.seed is None and located.position is None
    assert context["native"]["selection"] == {"label": 2, "expected": None}
    assert context["widgets"]["position"] == 5
    press(review, "Return to cell review")
    assert context_of(review)["workspace"]["queue"]["position"] == 1
    choose(review, "Review scope", "Dataset cells")
    assert (
        next(widget for widget in review.selectbox if widget.label == "Review view").value == "Spatial field"
    )
    press(review, "Return to sampled fields")
    returned = context_of(review)["workspace"]["active_field"][1]
    np.testing.assert_array_equal(returned.origin_xy_um, original.origin_xy_um)
    np.testing.assert_array_equal(returned.mask, original.mask)
    assert returned.seed == original.seed and returned.position == 5
    assert not context_of(review)["history"]
    assert not (root / "labels.csv").exists()
    assert not list(root.rglob("changes-*.jsonl"))


def test_located_cell_navigation_does_not_confirm_but_fresh_field_click_does(workspace_review):
    import json
    from streamlit.components.v2.bidi_component.main import _make_trigger_id

    review, reader, root = workspace_review
    press(review, "Review cells in this field")
    press(review, "Next")
    press(review, "Show in field")
    context = context_of(review)
    field = context["workspace"]["located_field"][1]
    yy, xx = np.nonzero(field.mask == 2)
    states = review._tree.get_widget_states()
    component_id = review.get("bidi_component")[0].proto.id
    states.widgets.add(id=component_id, json_value="{}")
    event = states.widgets.add(id=_make_trigger_id(component_id, "events"))
    event.json_trigger_value = json.dumps(
        [
            {
                "event": "clicked",
                "value": {
                    "x": float((xx.mean() + 0.5) / field.mask.shape[1]),
                    "y": float((yy.mean() + 0.5) / field.mask.shape[0]),
                    "token": context["event_token"],
                },
            }
        ]
    )
    review._run(states)
    assert not review.exception
    assert (root / "labels.csv").read_text().strip().endswith("prod3-realigned-v1|2,1,1")
    assert len(context_of(review)["history"]) == 1
    review.run()
    assert len(context_of(review)["history"]) == 1


def test_scope_return_restores_view_without_rebuilding_gallery(workspace_review):
    review, reader, root = workspace_review
    (root / "labels.csv").write_text("cell,brdu,edu\nprod3-realigned-v1|1,0,0\n")
    choose(review, "Review view", "EdU correction gallery")
    (gallery,) = context_of(review)["galleries"].values()
    queue_id = gallery["queue_id"]
    press(review, "Review cell")
    press(review, "Return to previous view")
    assert (
        next(widget for widget in review.selectbox if widget.label == "Review view").value
        == "EdU correction gallery"
    )
    assert next(iter(context_of(review)["galleries"].values()))["queue_id"] == queue_id
    choose(review, "Review scope", "Captured cells")
    choose(review, "Review scope", "Dataset cells")
    assert (
        next(widget for widget in review.selectbox if widget.label == "Review view").value
        == "EdU correction gallery"
    )
    assert not context_of(review)["history"]


@pytest.mark.parametrize("adapter", ["prod2", "prod3", "prod3_2", "prod2_2"])
def test_native_viewport_loader_preserves_geometry_and_adapter_pixels(tmp_path, adapter):
    from dinoapp import native, prod3, prod3_2, prod2_2

    cls = {
        "prod2": native.FieldSource,
        "prod3": prod3.FieldSource,
        "prod3_2": prod3_2.FieldSource,
        "prod2_2": prod2_2.FieldSource,
    }[adapter]
    reader = object.__new__(cls)
    reader.shape = (2, 24, 24)
    reader.origin, reader.scale = np.array([100.0, 200.0, 300.0]), np.array([2.0, 3.0, 4.0])
    reader.arrays = {
        name: np.arange(np.prod(reader.shape)).reshape(reader.shape)
        for name in (reader.anatomy_channel if adapter != "prod2" else "561", "EdU", "BrdU")
    }
    reader.labels = np.ones(reader.shape, dtype=np.uint16)
    reader.contours_at = lambda zi: []
    seen = []
    if adapter == "prod2":
        reader.assignment_paths = (tmp_path / "a.parquet", tmp_path / "b.parquet")
        for path in reader.assignment_paths:
            pd.DataFrame(dict(label=[1], r_vz=[1.0], cutoff_um=[2.0], near_ventricle=[True])).to_parquet(path)
    else:
        from types import SimpleNamespace

        def correct(images, selection):
            seen.append(selection)
            return {name: image + 5 for name, image in images.items()}

        reader.display_pixels = SimpleNamespace(field_images=correct)
        reader._field_boundary_info = lambda labels: {}
    field = reader.field_at([340.0, 230.0], 1, 20.0)
    np.testing.assert_array_equal(field.origin_xy_um, [328.0, 218.0])
    assert field.seed is None and field.position is None
    expected = reader.arrays["EdU"][1, 6:14, 7:13]
    np.testing.assert_array_equal(field.images["EdU"], expected if adapter == "prod2" else expected + 5)
    if adapter != "prod2":
        assert seen == [(1, slice(6, 14), slice(7, 13))]


def test_captured_assignment_rejects_label_changed_since_render(workspace_review):
    from dinoapp.label_store import update_label
    review, reader, root = workspace_review
    path = root / "labels.csv"
    update_label(path, "prod3-realigned-v1|1", (0, 0))
    press(review, "Review cells in this field")
    button = next(widget for widget in review.button if widget.label == "BrdU− / EdU+")
    update_label(path, "prod3-realigned-v1|1", (1, 1), expected=(0, 0))
    before = path.read_bytes()
    button.click().run()
    assert not review.exception
    assert path.read_bytes() == before
    assert not context_of(review)["history"]
    assert not list(root.rglob("changes-*.jsonl"))
