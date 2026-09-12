from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pytest

from dinoapp.label_store import LabelConflictError, load_labels, update_label


def _write_cell(args):
    path, number = args
    update_label(path, f"cell-{number}", (number % 2, 1))


def test_edits_preserve_other_cells_and_extra_columns(tmp_path):
    path = tmp_path / "labels.csv"
    pd.DataFrame([{"cell": "a", "brdu": 0, "edu": 0, "note": "keep"}]).to_csv(path, index=False)
    update_label(path, "b", (1, 0))
    assert update_label(path, "a", (0, 1)) == (0, 0)
    table = load_labels(path).set_index("cell")
    assert table.loc["a", "note"] == "keep"
    assert table.loc["b", "brdu"] == 1
    assert len(table) == 2


def test_concurrent_cell_edits_survive(tmp_path):
    path = tmp_path / "labels.csv"
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(_write_cell, [(path, i) for i in range(20)]))
    assert len(load_labels(path)) == 20


def test_undo_restores_old_value_and_removes_new_annotation(tmp_path):
    path = tmp_path / "labels.csv"
    assert update_label(path, "new", (1, 1), expected=None) is None
    update_label(path, "new", None, expected=(1, 1))
    assert load_labels(path).empty
    update_label(path, "old", (0, 1))
    previous = update_label(path, "old", (1, 0), expected=(0, 1))
    update_label(path, "old", previous, expected=(1, 0))
    assert load_labels(path).iloc[0][["brdu", "edu"]].tolist() == [0, 1]


def test_stale_undo_does_not_overwrite_another_edit(tmp_path):
    path = tmp_path / "labels.csv"
    update_label(path, "cell", (1, 1))
    with pytest.raises(LabelConflictError):
        update_label(path, "cell", (0, 0), expected=(1, 0))
    assert load_labels(path).iloc[0][["brdu", "edu"]].tolist() == [1, 1]


def test_invalid_input_is_rejected_without_changing_file(tmp_path):
    path = tmp_path / "labels.csv"
    update_label(path, "cell", (1, 0))
    original = path.read_bytes()
    with pytest.raises(ValueError):
        update_label(path, "cell", (2, 0))
    assert path.read_bytes() == original
