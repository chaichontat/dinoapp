"""Connected views resolve bounded identities without changing review rosters."""

import numpy as np
import pandas as pd
import pytest
from dino_markers import DEFAULT_CONTRACT, default_config

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
    reader.config = default_config()
    reader.cell_crop = lambda label: (str(label).encode(), int(reader.cell_geometry(label).plane_z))
    reader.marker_crop = lambda label, marker: reader.cell_crop(label)
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


def build_reader(reader, tmp_path):
    import io
    from types import SimpleNamespace
    from PIL import Image

    reader.config = default_config()
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
    reader.contrast = {"pi": (0.0, 5000.0), "brdu": (0.0, 5000.0), "edu": (0.0, 5000.0)}

    def crop(label):
        row = reader.cell_geometry(label)
        output = io.BytesIO()
        Image.new("RGB", (8, 8), (label * 40, 0, 0)).save(output, format="PNG")
        return output.getvalue(), int(row.plane_z)

    reader.cell_crop = crop
    reader.marker_crop = lambda label, marker: crop(label)
    return reader


@pytest.fixture
def workspace_review(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from dinoapp import app, native_app

    reader = build_reader(object.__new__(FieldSource), tmp_path)
    monkeypatch.setattr(FieldSource, "annotation_directories", classmethod(lambda cls, root=None: (tmp_path,)), raising=False)
    monkeypatch.setattr(native_app, "get_source", lambda cls, marker_revision: reader)
    monkeypatch.setattr(
        native_app,
        "get_field",
        lambda cls, marker_revision, seed, position, z, width: reader.field(seed, position, z, width),
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
        for name in reader.raw_channels
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
    np.testing.assert_array_equal(field.images["edu"], expected if adapter == "prod2" else expected + 5)
    if adapter != "prod2":
        assert seen == [(1, slice(6, 14), slice(7, 13))]


def test_captured_assignment_rejects_label_changed_since_render(workspace_review):
    from dinoapp.label_store import update_label
    review, reader, root = workspace_review
    path = root / "labels.csv"
    update_label(path, "prod3-realigned-v1|1", (0, 0), contract=DEFAULT_CONTRACT)
    press(review, "Review cells in this field")
    next(widget for widget in review.segmented_control if widget.label == "EdU").set_value(1).run()
    assert not review.exception
    save = next(widget for widget in review.button if widget.label == "Save and next")
    assert not save.disabled
    update_label(path, "prod3-realigned-v1|1", (1, 1), contract=DEFAULT_CONTRACT, expected=(0, 0))
    before = path.read_bytes()
    save.click().run()
    assert not review.exception
    assert path.read_bytes() == before
    assert not context_of(review)["history"]
    assert not list(root.rglob("changes-*.jsonl"))


THREE_MARKERS = {
    "version": 1,
    "reference": {"key": "dapi", "label": "DAPI"},
    "markers": [
        {"key": "edu", "label": "EdU", "color": "#00ff00"},
        {"key": "brdu", "label": "BrdU", "color": "#ff00ff"},
        {"key": "pcna", "label": "PCNA", "color": "#00ffff"},
    ],
    "presentation": {
        "crop_panels": ["dapi", "edu", "brdu", "pcna"],
        "negative_color": [0.25, 0.25, 0.25],
        "uncalled_color": "#999999",
    },
}
THREE_MARKER_CELLS = ("tri|1", "tri|2", "tri|3")


def write_three_marker_barrage(root, labels=None):
    import json
    from PIL import Image

    for index, cell in enumerate(THREE_MARKER_CELLS, start=1):
        for stem in ("combined", "edu", "brdu", "pcna"):
            Image.new("RGB", (8, 8), (index * 40, 0, 0)).save(root / f"{cell.split('|')[1]}_{stem}.png")
    rows = ["cell,rank,combined_image_path,edu_image_path,brdu_image_path,pcna_image_path"]
    for index, cell in enumerate(THREE_MARKER_CELLS, start=1):
        paths = ",".join(f"{index}_{stem}.png" for stem in ("combined", "edu", "brdu", "pcna"))
        rows.append(f"{cell},{index},{paths}")
    (root / "index.csv").write_text("\n".join(rows) + "\n")
    (root / "markers.json").write_text(json.dumps(THREE_MARKERS))
    if labels is not None:
        (root / "labels.csv").write_text("cell,edu,brdu,pcna\n" + "".join(f"{row}\n" for row in labels))


@pytest.fixture
def three_marker_barrage(tmp_path, monkeypatch):
    from dinoapp import app

    monkeypatch.setenv("DINO_REVIEW_SOURCE", "Barrage")
    monkeypatch.setenv("DINO_REVIEW_BARRAGE_DIR", str(tmp_path))
    app.st.cache_resource.clear()

    def open_review(labels=None):
        from streamlit.testing.v1 import AppTest

        write_three_marker_barrage(tmp_path, labels)
        review = AppTest.from_file(app.__file__).run(timeout=20)
        assert not review.exception
        return review

    return tmp_path, open_review


def set_marker(review, label, value):
    next(widget for widget in review.segmented_control if widget.label == label).set_value(value).run()
    assert not review.exception


def test_cell_review_offers_one_control_per_marker_and_starts_unset(three_marker_barrage):
    root, open_review = three_marker_barrage
    review = open_review()
    choose(review, "Review view", "Cell review")
    assert [widget.label for widget in review.segmented_control] == ["EdU", "BrdU", "PCNA"]
    assert [widget.value for widget in review.segmented_control] == [None, None, None]
    save = next(widget for widget in review.button if widget.label == "Save and next")
    assert save.disabled
    assert "No saved call" in [element.value for element in review.main.markdown]
    assert not (root / "labels.csv").exists()


def test_save_and_next_waits_for_every_marker_then_writes_one_row_and_advances(three_marker_barrage):
    root, open_review = three_marker_barrage
    review = open_review()
    choose(review, "Review view", "Cell review")
    set_marker(review, "EdU", 1)
    set_marker(review, "BrdU", 0)
    assert next(widget for widget in review.button if widget.label == "Save and next").disabled
    set_marker(review, "PCNA", 1)
    save = next(widget for widget in review.button if widget.label == "Save and next")
    assert not save.disabled
    assert not (root / "labels.csv").exists()
    save.click().run()
    assert not review.exception
    assert (root / "labels.csv").read_text() == "cell,edu,brdu,pcna\ntri|1,1,0,1\n"
    context = context_of(review)
    assert context["history"][-1].value == (1, 0, 1)
    assert context["drafts"] == {"tri|2": {"expected": None, "values": {"edu": None, "brdu": None, "pcna": None}}}
    (queue,) = context["cell_queues"].values()
    assert queue["position"] == 1
    assert "**Cell 2 of 3**" in [element.value for element in review.main.markdown]
    assert [widget.value for widget in review.segmented_control] == [None, None, None]
    journal_path, = (root / "cell-review").glob("changes-*.jsonl")
    import json
    record = json.loads(journal_path.read_text().splitlines()[0])
    assert (record["cell"], record["previous"], record["value"], record["action"]) == ("tri|1", None, [1, 0, 1], "assign")


def test_unsaved_cells_exclude_saved_all_negative_and_saved_cells_filter_per_marker(three_marker_barrage):
    root, open_review = three_marker_barrage
    review = open_review(labels=["tri|1,0,0,0", "tri|2,1,0,1"])
    choose(review, "Review view", "Cell review")
    (queue,) = context_of(review)["cell_queues"].values()
    assert queue["cells"] == ["tri|3"]
    choose(review, "Cells", "All cells")
    assert [q["cells"] for q in context_of(review)["cell_queues"].values()][-1] == ["tri|1", "tri|2", "tri|3"]
    choose(review, "Cells", "Saved")
    assert [widget.label for widget in review.sidebar.selectbox][-3:] == ["EdU", "BrdU", "PCNA"]
    assert [q["cells"] for q in context_of(review)["cell_queues"].values()][-1] == ["tri|1", "tri|2"]
    choose(review, "EdU", "EdU+")
    assert [q["cells"] for q in context_of(review)["cell_queues"].values()][-1] == ["tri|2"]
    choose(review, "BrdU", "BrdU−")
    choose(review, "PCNA", "PCNA−")
    assert [q["cells"] for q in context_of(review)["cell_queues"].values()][-1] == []
    choose(review, "PCNA", "PCNA+")
    assert [q["cells"] for q in context_of(review)["cell_queues"].values()][-1] == ["tri|2"]
    assert "No saved call" not in [element.value for element in review.main.markdown]
    assert "EdU+ / BrdU− / PCNA+" in [element.value for element in review.main.markdown]


def send_flip(review, payload):
    import json
    from streamlit.components.v2.bidi_component.main import _make_trigger_id

    states = review._tree.get_widget_states()
    component_id = review.get("bidi_component")[0].proto.id
    states.widgets.add(id=component_id, json_value="{}")
    event = states.widgets.add(id=_make_trigger_id(component_id, "events"))
    event.json_trigger_value = json.dumps([{"event": "flipped", "value": payload}])
    review._run(states)


def test_gallery_views_and_per_marker_filters_follow_the_config_and_a_card_toggles_one_marker(three_marker_barrage):
    root, open_review = three_marker_barrage
    review = open_review(labels=["tri|1,0,0,0", "tri|2,1,0,1", "tri|3,1,1,0"])
    views = next(widget for widget in review.sidebar.selectbox if widget.label == "Review view").options
    assert views == ["Cell review", "EdU correction gallery", "BrdU correction gallery", "PCNA correction gallery"]
    choose(review, "Review view", "PCNA correction gallery")
    (gallery,) = context_of(review)["galleries"].values()
    assert gallery["cells"] == ["tri|1", "tri|2", "tri|3"]
    assert [widget.label for widget in review.sidebar.selectbox][-3:] == ["EdU", "BrdU", "PCNA"]
    choose(review, "EdU", "EdU+")
    choose(review, "BrdU", "BrdU−")
    context = context_of(review)
    gallery = next(g for key, g in context["galleries"].items() if key[-1] == (1, 0, None))
    assert gallery["cells"] == ["tri|2"]
    assert gallery["marker"] == "pcna"
    page = gallery["pages"][(gallery["queue_id"], 0)]
    (card,) = page
    assert (card["state"], card["action"], card["expected"], card["value"]) == (
        "EdU+ / BrdU− / PCNA+", "Set PCNA−", [1, 0, 1], [1, 0, 0],
    )
    send_flip(review, {
        "itemId": "tri|2", "cell": "tri|2", "expected": card["expected"], "value": card["value"],
        "token": context["event_token"],
    })
    assert not review.exception
    assert (root / "labels.csv").read_text() == "cell,edu,brdu,pcna\ntri|1,0,0,0\ntri|2,1,0,0\ntri|3,1,1,0\n"
    assert context_of(review)["history"][-1].previous == (1, 0, 1)
    assert "EdU+ / BrdU− / PCNA−" in context_of(review)["notice"]


def test_gallery_rejects_a_flip_that_changes_more_than_its_marker(three_marker_barrage):
    root, open_review = three_marker_barrage
    review = open_review(labels=["tri|2,1,0,1"])
    choose(review, "Review view", "BrdU correction gallery")
    before = (root / "labels.csv").read_bytes()
    send_flip(review, {
        "itemId": "tri|2", "cell": "tri|2", "expected": [1, 0, 1], "value": [1, 1, 0],
        "token": context_of(review)["event_token"],
    })
    assert not review.exception
    assert (root / "labels.csv").read_bytes() == before
    assert "Ignored an invalid gallery edit" in context_of(review)["notice"]


def test_changing_only_a_label_rebuilds_the_source_and_survives_a_stale_filter(three_marker_barrage):
    import json
    import os

    root, open_review = three_marker_barrage
    review = open_review(labels=["tri|1,0,0,0"])
    choose(review, "Review view", "EdU correction gallery")
    choose(review, "EdU", "EdU+")
    markers = json.loads(json.dumps(THREE_MARKERS))
    markers["markers"][0]["label"] = "EdU label"
    stat = (root / "markers.json").stat()
    (root / "markers.json").write_text(json.dumps(markers))
    os.utime(root / "markers.json", ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    review.run()
    assert not review.exception
    views = next(widget for widget in review.sidebar.selectbox if widget.label == "Review view").options
    assert views == ["Cell review", "EdU label correction gallery", "BrdU correction gallery", "PCNA correction gallery"]
    choose(review, "Review view", "EdU label correction gallery")
    assert [widget.label for widget in review.sidebar.selectbox][-3:] == ["EdU label", "BrdU", "PCNA"]
    assert review.sidebar.selectbox[-3].value == "Any"
    choose(review, "Review view", "Cell review")
    assert [widget.label for widget in review.segmented_control] == ["EdU label", "BrdU", "PCNA"]


def test_stale_session_write_shows_the_mismatch_and_leaves_labels_unchanged(three_marker_barrage, monkeypatch):
    import json
    from dinoapp import review_session

    root, open_review = three_marker_barrage
    review = open_review(labels=["tri|1,0,0,0"])
    choose(review, "Review view", "Cell review")
    choose(review, "Cells", "All cells")
    for label, value in (("EdU", 1), ("BrdU", 1), ("PCNA", 1)):
        set_marker(review, label, value)
    reordered = json.loads(json.dumps(THREE_MARKERS))
    reordered["markers"].reverse()
    persist = review_session.save_assignment

    def stale_save(annotations, state_dir, context, cell, value, **kwargs):
        (root / "markers.json").write_text(json.dumps(reordered))
        return persist(annotations, state_dir, context, cell, value, **kwargs)

    monkeypatch.setattr(review_session, "save_assignment", stale_save)
    before = (root / "labels.csv").read_bytes()
    next(widget for widget in review.button if widget.label == "Save and next").click().run()
    assert not review.exception
    assert (root / "labels.csv").read_bytes() == before
    assert not context_of(review)["history"]
    assert any("Marker contract mismatch" in element.value for element in review.info)
    assert not list((root / "cell-review").glob("changes-*.jsonl"))


def test_annotation_marker_change_rebuilds_the_source_and_the_sampled_field(tmp_path, monkeypatch):
    import json
    import os
    import streamlit as st
    from dino_markers import config_to_json, resolve_directory
    from streamlit.testing.v1 import AppTest
    from dinoapp import app, prod3

    built = {"sources": 0, "fields": 0}

    class Reader(FieldSource):
        def __init__(self):
            built["sources"] += 1
            build_reader(self, tmp_path)
            self.config = resolve_directory(tmp_path)

        @classmethod
        def annotation_directories(cls, root=None):
            return (tmp_path,)

        def field(self, *args):
            built["fields"] += 1
            return super().field(*args)

    monkeypatch.setattr(prod3, "FieldSource", Reader)
    monkeypatch.setenv("DINO_REVIEW_SOURCE", "Native fields")
    monkeypatch.setenv("DINO_NATIVE_DATASET", "Prod3")
    monkeypatch.setenv("DINO_REVIEW_LABEL_ROOT", str(tmp_path))
    app.st.cache_resource.clear()
    st.cache_data.clear()
    (tmp_path / "labels.csv").write_text("cell,brdu,edu\nprod3-realigned-v1|1,0,0\n")
    review = AppTest.from_file(app.__file__).run(timeout=20)
    assert not review.exception
    assert built == {"sources": 1, "fields": 1}
    review.run()
    assert not review.exception
    assert built == {"sources": 1, "fields": 1}

    markers = config_to_json(default_config())
    markers["markers"][1]["label"] = "EdU label"
    markers["presentation"]["crop_panels"] = ["pi", "edu", "brdu"]
    stat = (tmp_path / "labels.csv").stat()
    (tmp_path / "markers.json").write_text(json.dumps(markers))
    os.utime(tmp_path / "markers.json", ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    review.run()
    assert not review.exception
    assert built == {"sources": 2, "fields": 2}
    assert [widget.label for widget in review.segmented_control] == ["EdU label", "BrdU"]
    review.run()
    assert built == {"sources": 2, "fields": 2}
