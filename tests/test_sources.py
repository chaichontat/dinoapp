from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from dinoapp import app
from dinoapp.app import new_gallery_state, update_gallery_card
from dinoapp.label_store import LabelConflictError, load_labels, update_label
from dinoapp.sources import AnnotationFiles, BarrageSource, NativeSource


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
    annotations = AnnotationFiles(source.annotation_paths, source.default_annotation_path)

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


def test_annotation_undo_uses_compare_and_set(tmp_path: Path):
    path = tmp_path / "labels.csv"
    update_label(path, "cell|1", (0, 0))
    annotations = AnnotationFiles((path,), path)
    edit = annotations.assign("cell|1", (1, 0), expected=(0, 0))
    update_label(path, "cell|1", (0, 1), expected=(1, 0))

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

    field_source = SimpleNamespace(
        training_predictions=predictions,
        model_version="model-1",
        origin=[100.0],
        scale=[2.0],
        brdu_cell_crop=crop("brdu"),
        edu_cell_crop=crop("edu"),
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


def test_gallery_queue_is_stable_and_cards_change_only_one_marker(tmp_path: Path):
    write_barrage(tmp_path)
    source = BarrageSource(tmp_path)
    annotations = AnnotationFiles(source.annotation_paths, source.default_annotation_path)
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
    update_gallery_card(card, (1, 0), gallery["initial"]["one:1|10"])
    assert card["expected"] == [1, 0]
    assert card["value"] == [1, 1]
    assert card["changed"] is True


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
