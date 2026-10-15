from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from dino_markers import (
    DEFAULT_CONTRACT,
    ContractMismatchError,
    MarkerConfig,
    default_config,
    parse_config,
    save_config,
)

from dinoapp import app
from dinoapp.app import (
    gallery_page_crops,
    get_native_modules,
    new_gallery_state,
    prefetch_gallery_page,
    update_gallery_card,
)
from dinoapp.label_store import LabelConflictError, load_labels, update_label
from dinoapp.sources import AnnotationFiles, BarrageSource, NativeSource


def three_marker_config() -> MarkerConfig:
    return parse_config(
        {
            "version": 1,
            "reference": {"key": "dapi", "label": "DAPI"},
            "markers": [
                {"key": "edu", "label": "EdU", "color": "#00ff00"},
                {"key": "brdu", "label": "BrdU", "color": "#ff00ff"},
                {"key": "pcna", "label": "PCNA", "color": "#00ffff"},
            ],
            "presentation": {
                "crop_panels": ["dapi", "pcna", "edu", "brdu"],
                "negative_color": [0.25, 0.25, 0.25],
                "uncalled_color": "#999999",
            },
        },
        source="test",
    )


def write_three_marker_barrage(root: Path, *, views: tuple[str, ...] = ("combined", "edu", "brdu", "pcna")) -> None:
    root.mkdir(parents=True, exist_ok=True)
    save_config(three_marker_config(), root)
    rows = []
    for rank, cell in enumerate(("t:1|1", "t:1|2", "t:1|3")):
        paths = {}
        for view in views:
            path = Path(f"{view}-{rank}.png")
            (root / path).write_bytes(f"{view}-{cell}".encode())
            paths[f"{view}_image_path"] = str(path)
        rows.append({"rank": rank, "cell": cell, **paths})
    pd.DataFrame(rows).to_csv(root / "index.csv", index=False)
    (root / "labels.csv").write_text("cell,edu,brdu,pcna\nt:1|1,0,0,0\nt:1|2,1,0,1\nt:1|3,1,1,1\n")


def write_barrage(root: Path) -> None:
    for directory in ("images-brdu", "images-edu", "images-combined"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, (cell, dataset) in enumerate((("one:1|10", "one"), ("two:2|20", "two"))):
        paths = {}
        for marker in ("brdu", "edu", "combined"):
            path = Path(f"images-{marker}") / f"{rank}.png"
            (root / path).write_bytes(f"{marker}-{cell}".encode())
            paths[f"{marker}_image_path"] = str(path)
        rows.append(
            {
                "rank": rank,
                "cell": cell,
                "dataset": dataset,
                "roi": rank + 1,
                "label": (rank + 1) * 10,
                **paths,
            }
        )
    pd.DataFrame(rows).to_csv(root / "index.csv", index=False)
    pd.DataFrame(
        {
            "cell": ["one:1|10", "two:2|20", "outside:9|90"],
            "brdu": [0, 0, 1],
            "edu": [0, 1, 1],
        }
    ).to_csv(root / "labels.csv", index=False)


def test_barrage_source_filters_live_labels_without_dropping_external_labels(tmp_path: Path):
    write_barrage(tmp_path)
    source = BarrageSource(tmp_path)
    annotations = AnnotationFiles(
        source.annotation_paths, source.default_annotation_path, contract=source.config.contract
    )

    rows = source.correction_rows(
        annotations.values,
        identity=(0, 0),
        subset="All labeled",
        dataset="All datasets",
    )

    assert rows.cell.tolist() == ["one:1|10"]
    assert source.datasets == ("All datasets", "one", "two")
    assert source.review_rows(dataset="two").cell.tolist() == ["two:2|20"]
    crop = source.crop("one:1|10", "edu")
    assert crop.image == b"edu-one:1|10"
    assert crop.title == "Cell 10"
    assert crop.detail == "one · ROI 1 · rank 1"
    annotations.assign("one:1|10", (1, 0), expected=(0, 0))
    assert "outside:9|90" in load_labels(tmp_path / "labels.csv").cell.tolist()


def test_barrage_source_rejects_crop_paths_outside_root(tmp_path: Path):
    write_barrage(tmp_path)
    index = pd.read_csv(tmp_path / "index.csv")
    index.loc[0, "edu_image_path"] = "../escape.png"
    index.to_csv(tmp_path / "index.csv", index=False)

    with pytest.raises(ValueError, match="escapes"):
        BarrageSource(tmp_path).crop("one:1|10", "edu")


def test_barrage_crop_title_uses_the_written_cell_identity(tmp_path: Path):
    write_barrage(tmp_path)
    index = pd.read_csv(tmp_path / "index.csv")
    index.loc[0, "label"] = 999
    index.to_csv(tmp_path / "index.csv", index=False)

    crop = BarrageSource(tmp_path).crop("one:1|10", "edu")

    assert crop.title == "Cell 10"


def test_annotation_undo_uses_compare_and_set(tmp_path: Path):
    path = tmp_path / "labels.csv"
    update_label(path, "cell|1", (0, 0), contract=DEFAULT_CONTRACT)
    annotations = AnnotationFiles((path,), path, contract=DEFAULT_CONTRACT)
    edit = annotations.assign("cell|1", (1, 0), expected=(0, 0))
    update_label(path, "cell|1", (0, 1), contract=DEFAULT_CONTRACT, expected=(1, 0))

    with pytest.raises(LabelConflictError):
        annotations.undo(edit)


def test_native_adapter_keeps_numeric_parsing_inside_source():
    predictions = pd.DataFrame(
        {
            "cell": ["full-run2|10", "full-run2|20"],
            "split": ["training", "validation"],
            "tranche": ["cortex", "pixel"],
        }
    )
    calls: list[tuple[str, int]] = []

    def crop(marker: str):
        def read(label: int):
            calls.append((marker, label))
            return marker.encode(), 3

        return read

    annotation_paths = tuple(
        Path("/tmp/native-labels") / name / "labels.csv"
        for name in ("barrage-cortex", "barrage-vz-svz", "barrage-pixel")
    )
    field_source = SimpleNamespace(
        annotation_paths=annotation_paths,
        default_annotation_path=annotation_paths[-1],
        display_name="Prod2 native",
        cell_id=lambda label: f"full-run2|{label}",
        gallery_predictions=predictions,
        model_version="model-1",
        origin=[100.0],
        scale=[2.0],
        config=default_config(),
        marker_crop=lambda label, marker: crop(marker)(label),
        cell_crop=crop("combined"),
    )
    source = NativeSource(field_source, label_root=Path("/tmp/native-labels"))
    rows = source.correction_rows(
        {"full-run2|10": (0, 0), "full-run2|20": (1, 0)},
        identity=(0, 0),
        subset="Training",
        dataset="All datasets",
    )

    assert rows.cell.tolist() == ["full-run2|10"]
    result = source.crop("full-run2|10", "edu")
    assert calls == [("edu", 10)]
    assert result.detail == "Training · cortex · Z 106.0 µm"


def test_native_adapter_uses_dataset_annotation_ownership(tmp_path: Path):
    paths = (tmp_path / "cortex.csv", tmp_path / "vz_svz.csv")
    update_label(paths[0], "prod3|10", (0, 0), contract=DEFAULT_CONTRACT)
    field_source = SimpleNamespace(
        annotation_paths=paths,
        default_annotation_path=paths[-1],
        display_name="Prod3 native",
        config=default_config(),
        gallery_predictions=pd.DataFrame({"cell": ["prod3|10", "prod3|20"]}),
        model_version="model-3",
    )

    source = NativeSource(field_source, label_root=tmp_path / "review-state")

    assert source.annotation_paths == paths
    assert source.default_annotation_path == paths[-1]
    assert source.display_name == "Prod3 native"
    annotations = AnnotationFiles(
        source.annotation_paths, source.default_annotation_path, contract=source.config.contract
    )
    annotations.assign("prod3|10", (1, 0), expected=(0, 0))
    annotations.assign("prod3|20", (0, 1), expected=None)
    assert load_labels(paths[0])[["cell", "brdu", "edu"]].values.tolist() == [["prod3|10", 1, 0]]
    assert load_labels(paths[1])[["cell", "brdu", "edu"]].values.tolist() == [["prod3|20", 0, 1]]
    assert not (tmp_path / "review-state").exists()


def test_gallery_queue_is_stable_and_cards_change_only_one_marker(tmp_path: Path):
    write_barrage(tmp_path)
    source = BarrageSource(tmp_path)
    annotations = AnnotationFiles(
        source.annotation_paths, source.default_annotation_path, contract=source.config.contract
    )
    gallery = new_gallery_state(
        source,
        annotations,
        marker="edu",
        subset="All labeled",
        dataset="All datasets",
        identity=(0, 0),
    )
    annotations.assign("one:1|10", (1, 0), expected=(0, 0))

    assert gallery["cells"] == ["one:1|10"]
    card = {"cell": "one:1|10", "cellLabel": "Cell 10", "marker": "edu"}
    update_gallery_card(card, (1, 0), gallery["initial"]["one:1|10"], config=source.config)
    assert card["expected"] == [1, 0]
    assert card["value"] == [1, 1]
    assert card["changed"] is True


def test_gallery_prefetch_reuses_next_page_crops(monkeypatch: pytest.MonkeyPatch):
    class ImmediateExecutor:
        def submit(self, function, *args):
            future = Future()
            future.set_result(function(*args))
            return future

    class Source:
        source_id = "test"

        def __init__(self):
            self.calls: list[tuple[str, str]] = []

        def crop(self, cell: str, marker: str):
            self.calls.append((cell, marker))
            return SimpleNamespace(image=cell.encode(), title=cell, detail=marker)

    source = Source()
    cells = [f"cell|{index}" for index in range(app.GALLERY_PAGE_SIZE + 2)]
    gallery = {
        "queue_id": "queue",
        "marker": "edu",
        "cells": cells,
        "pages": {},
        "prefetches": {},
    }
    monkeypatch.setattr(app, "gallery_prefetch_executor", lambda: ImmediateExecutor())

    prefetch_gallery_page(source, gallery, 1)
    prefetched_calls = list(source.calls)
    crops = gallery_page_crops(source, gallery, 1)

    assert prefetched_calls == [("cell|24", "edu"), ("cell|25", "edu")]
    assert source.calls == prefetched_calls
    assert [crop.title for crop in crops] == ["cell|24", "cell|25"]


def test_gallery_rejects_disagreement_between_item_and_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class Rerun(Exception):
        pass

    class Column:
        def button(self, *args, **kwargs):
            return False

        def write(self, *args, **kwargs):
            return None

        def info(self, *args, **kwargs):
            return None

    cards = [
        {
            "itemId": cell,
            "cell": cell,
            "marker": "edu",
            "expected": [0, 0],
            "value": [0, 1],
        }
        for cell in ("cell|1", "cell|2")
    ]
    gallery = {
        "marker": "edu",
        "cells": ["cell|1", "cell|2"],
        "page": 0,
        "queue_id": "queue",
        "pages": {("queue", 0): cards},
        "prefetches": {},
        "initial": {"cell|1": (0, 0), "cell|2": (0, 0)},
    }
    context = app.get_context("event-test")
    event = {
        "itemId": "cell|1",
        "cell": "cell|2",
        "expected": [0, 0],
        "value": [0, 1],
        "token": "source:queue:0",
    }
    result = SimpleNamespace(needsFull=None, flipped=event)
    saved: list[str] = []

    def component(**kwargs):
        return result

    monkeypatch.setattr(app, "gallery_picker", lambda: component)
    monkeypatch.setattr(app, "prefetch_gallery_page", lambda *args: None)
    monkeypatch.setattr(app, "save_assignment", lambda *args, **kwargs: saved.append("saved"))
    monkeypatch.setattr(app.st, "title", lambda *args, **kwargs: None)
    monkeypatch.setattr(app.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(app.st, "columns", lambda widths, **kwargs: tuple(Column() for _ in widths))
    monkeypatch.setattr(app.st, "rerun", lambda: (_ for _ in ()).throw(Rerun()))

    with pytest.raises(Rerun):
        app.render_gallery(
            SimpleNamespace(source_id="source", crop_markers=("edu",), config=default_config()),
            SimpleNamespace(values={"cell|1": (0, 0), "cell|2": (0, 0)}),
            context,
            gallery,
            state_dir=tmp_path,
        )

    assert saved == []
    assert context["notice"] == (
        "That cell changed after this gallery was rendered; no edit was saved."
    )


def test_native_loader_uses_packaged_modules(monkeypatch):
    adapter = "prod3_2"
    imports = []

    def load(name):
        imports.append(name)
        return SimpleNamespace(__name__=name)

    monkeypatch.setattr(app.importlib, "import_module", load)
    get_native_modules.clear()
    try:
        native, native_app = get_native_modules(adapter)
        assert native.__name__ == app.ADAPTERS[adapter]
        assert native_app.__name__ == "dinoapp.native_app"
        assert imports == [app.ADAPTERS[adapter], "dinoapp.native_app"]
    finally:
        get_native_modules.clear()


def test_launch_starts_universal_streamlit_app(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[list[str], bool]] = []
    monkeypatch.setattr(app.sys, "argv", ["dinoapp", "--server.port", "8502"])
    monkeypatch.setattr(
        app.subprocess,
        "run",
        lambda command, check: calls.append((command, check)),
    )

    app.launch()

    assert calls == [
        (
            [
                app.sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app.Path(app.__file__).resolve()),
                "--server.port",
                "8502",
            ],
            True,
        )
    ]


@pytest.mark.parametrize('marker', ['edu', 'brdu'])
@pytest.mark.parametrize('empty_reason', ['source', 'annotations', 'split', 'identity'])
def test_native_empty_gallery_preserves_row_schema(tmp_path, marker, empty_reason):
    columns = ['cell', 'label', 'tranche', 'split', 'p_brdu', 'pred_brdu', 'p_edu', 'pred_edu']
    frame = pd.DataFrame([['prod3|10', 10, 'vz_svz', 'training', .2, False, .7, True]], columns=columns)
    if empty_reason == 'source':
        frame = pd.DataFrame(columns=columns)  # The live Prod3 adapter's original schema.
    path = tmp_path / 'labels.csv'
    pd.DataFrame([{'cell': 'prod3|10', 'brdu': 0, 'edu': 1}]).to_csv(path, index=False)
    annotations = AnnotationFiles((path,), path, contract=DEFAULT_CONTRACT)
    if empty_reason == 'annotations':
        annotations.values.clear()
    field_source = SimpleNamespace(annotation_paths=(path,), default_annotation_path=path,
                                   display_name='Prod3 native', gallery_predictions=frame, config=default_config(),
                                   model_version='baseline')
    source = NativeSource(field_source, label_root=tmp_path)
    subset = 'Validation' if empty_reason == 'split' else 'All labeled'
    identity = (1, 0) if empty_reason == 'identity' else (0, 1)
    rows = source.correction_rows(annotations.values, identity=identity, subset=subset, dataset='All datasets')
    assert rows.empty
    assert rows.columns.tolist() == columns
    gallery = new_gallery_state(source, annotations, marker=marker, subset=subset,
                                dataset='All datasets', identity=identity)
    assert gallery['cells'] == []
    assert gallery['initial'] == {}


def test_native_target_identity_tracks_annotation_ownership_not_review_revisions(tmp_path):
    paths = (tmp_path / 'cortex' / 'labels.csv', tmp_path / 'pixel' / 'labels.csv')
    predictions = pd.DataFrame({'cell': ['prod3|1'], 'split': ['training'], 'tranche': ['cortex']})

    def reader(model, pixels, owners=paths):
        return SimpleNamespace(annotation_paths=owners, default_annotation_path=owners[-1],
                               gallery_predictions=predictions, display_name='Native', config=default_config(),
                               model_version=model, pixel_version=pixels)

    first = NativeSource(reader('model-1', 'display-1'), label_root=tmp_path / 'session-one')
    second = NativeSource(reader('model-2', 'display-2'), label_root=tmp_path / 'session-two')
    separate = NativeSource(reader('model-1', 'display-1', (tmp_path / 'other' / 'labels.csv',)),
                            label_root=tmp_path / 'session-one')
    assert first.target_id == second.target_id
    assert first.source_id != second.source_id
    assert first.target_id != separate.target_id


def test_barrage_revision_does_not_replace_its_annotation_target(tmp_path):
    write_barrage(tmp_path)
    first = BarrageSource(tmp_path)
    (tmp_path / 'complete.json').write_text('{"revision": 2}')
    second = BarrageSource(tmp_path)
    assert first.target_id == second.target_id
    assert first.source_id != second.source_id
    assert first.annotation_paths == second.annotation_paths


def test_hand_made_three_marker_barrage_follows_display_markers_and_marker_image_paths(tmp_path: Path):
    write_three_marker_barrage(tmp_path)
    source = BarrageSource(tmp_path)

    assert source.config == three_marker_config()
    assert source.crop_markers == ("combined", "pcna", "edu", "brdu")
    for marker in ("combined", "edu", "brdu", "pcna"):
        assert source.crop("t:1|2", marker).image == f"{marker}-t:1|2".encode()
    for marker in ("dapi", "pi", "other"):
        with pytest.raises(ValueError, match="Unexpected crop marker"):
            source.crop("t:1|2", marker)


def test_three_marker_barrage_lists_only_views_with_an_image_column(tmp_path: Path):
    write_three_marker_barrage(tmp_path, views=("combined", "pcna"))

    assert BarrageSource(tmp_path).crop_markers == ("combined", "pcna")


def test_three_marker_annotations_hold_triples_and_filter_per_marker(tmp_path: Path):
    write_three_marker_barrage(tmp_path)
    source = BarrageSource(tmp_path)
    annotations = AnnotationFiles(
        source.annotation_paths, source.default_annotation_path, contract=source.config.contract
    )
    assert annotations.values == {"t:1|1": (0, 0, 0), "t:1|2": (1, 0, 1), "t:1|3": (1, 1, 1)}

    def cells(identity):
        rows = source.correction_rows(
            annotations.values, identity=identity, subset="All labeled", dataset="All datasets"
        )
        return rows.cell.tolist()

    assert cells((1, 0, 1)) == ["t:1|2"]
    assert cells((None, None, None)) == ["t:1|1", "t:1|2", "t:1|3"]
    assert cells((1, None, 1)) == ["t:1|2", "t:1|3"]
    assert cells((None, 0, None)) == ["t:1|1", "t:1|2"]
    assert cells((None, None, 0)) == ["t:1|1"]
    with pytest.raises(ValueError, match="one entry per marker"):
        cells((0, 1))


def test_annotation_files_reject_a_directory_whose_contract_differs(tmp_path: Path):
    write_three_marker_barrage(tmp_path / "three")
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "labels.csv").write_text("cell,brdu,edu\nx|1,0,1\n")
    three = three_marker_config().contract

    with pytest.raises(ContractMismatchError, match="marker order is part of the contract"):
        AnnotationFiles((tmp_path / "three" / "labels.csv",), tmp_path / "three" / "labels.csv", contract=DEFAULT_CONTRACT)
    with pytest.raises(ContractMismatchError):
        AnnotationFiles((plain / "labels.csv",), plain / "labels.csv", contract=three)
    with pytest.raises(ContractMismatchError):
        AnnotationFiles(
            (plain / "labels.csv", tmp_path / "three" / "labels.csv"), plain / "labels.csv", contract=DEFAULT_CONTRACT
        )


def test_native_adapter_dispatches_crops_by_marker_through_marker_crop(tmp_path: Path):
    calls: list[tuple[str, int, str | None]] = []
    config = three_marker_config()
    path = tmp_path / "labels.csv"
    field_source = SimpleNamespace(
        annotation_paths=(path,),
        default_annotation_path=path,
        display_name="Three marker",
        config=config,
        cell_id=lambda label: f"three|{label}",
        gallery_predictions=pd.DataFrame(
            {"cell": ["three|10"], "split": ["training"], "tranche": ["cortex"]}
        ),
        model_version="model-1",
        origin=[100.0],
        scale=[2.0],
        cell_crop=lambda label: (calls.append(("combined", label, None)) or b"c", 3),
        marker_crop=lambda label, marker: (calls.append(("marker", label, marker)) or marker.encode(), 3),
    )
    source = NativeSource(field_source, label_root=tmp_path)

    assert source.config is config
    assert source.crop_markers == ("combined", "pcna", "edu", "brdu")
    assert source.crop("three|10", "combined").image == b"c"
    assert source.crop("three|10", "pcna").image == b"pcna"
    assert calls == [("combined", 10, None), ("marker", 10, "pcna")]
    for marker in ("dapi", "other"):
        with pytest.raises(ValueError, match="Unexpected crop marker"):
            source.crop("three|10", marker)
    assert calls == [("combined", 10, None), ("marker", 10, "pcna")]


def test_native_correction_rows_accept_none_entries(tmp_path: Path):
    config = three_marker_config()
    path = tmp_path / "labels.csv"
    predictions = pd.DataFrame(
        {
            "cell": ["three|1", "three|2", "three|3"],
            "split": ["training", "training", "validation"],
            "tranche": ["cortex"] * 3,
        }
    )
    field_source = SimpleNamespace(
        annotation_paths=(path,), default_annotation_path=path, display_name="Three marker",
        config=config, gallery_predictions=predictions, model_version="m",
    )
    source = NativeSource(field_source, label_root=tmp_path)
    live = {"three|1": (1, 0, 1), "three|2": (1, 1, 1), "three|3": (1, 0, 0)}

    def cells(identity, subset="All labeled"):
        return source.correction_rows(
            live, identity=identity, subset=subset, dataset="All datasets"
        ).cell.tolist()

    assert cells((1, None, None)) == ["three|1", "three|2", "three|3"]
    assert cells((None, 0, None)) == ["three|1", "three|3"]
    assert cells((None, 0, None), "Training") == ["three|1"]
    assert cells((1, 0, 1)) == ["three|1"]
