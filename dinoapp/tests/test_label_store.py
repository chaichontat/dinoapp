from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pytest
from dino_markers import (
    DEFAULT_CONTRACT,
    ContractMismatchError,
    MarkerConfig,
    MarkerContract,
    parse_config,
    save_config,
)

from dinoapp.label_store import LabelConflictError, load_labels, update_label

REORDERED = MarkerContract("pi", ("edu", "brdu"))


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
                "crop_panels": ["dapi", "edu", "brdu", "pcna"],
                "negative_color": [0.25, 0.25, 0.25],
                "uncalled_color": "#999999",
            },
        },
        source="test",
    )


def _write_cell(args):
    path, number = args
    update_label(path, f"cell-{number}", (number % 2, 1), contract=DEFAULT_CONTRACT)


def test_edits_preserve_other_cells_and_extra_columns(tmp_path):
    path = tmp_path / "labels.csv"
    pd.DataFrame([{"cell": "a", "brdu": 0, "edu": 0, "note": "keep"}]).to_csv(path, index=False)
    update_label(path, "b", (1, 0), contract=DEFAULT_CONTRACT)
    assert update_label(path, "a", (0, 1), contract=DEFAULT_CONTRACT) == (0, 0)
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
    assert update_label(path, "new", (1, 1), contract=DEFAULT_CONTRACT, expected=None) is None
    update_label(path, "new", None, contract=DEFAULT_CONTRACT, expected=(1, 1))
    assert load_labels(path).empty
    update_label(path, "old", (0, 1), contract=DEFAULT_CONTRACT)
    previous = update_label(path, "old", (1, 0), contract=DEFAULT_CONTRACT, expected=(0, 1))
    update_label(path, "old", previous, contract=DEFAULT_CONTRACT, expected=(1, 0))
    assert load_labels(path).iloc[0][["brdu", "edu"]].tolist() == [0, 1]


def test_stale_undo_does_not_overwrite_another_edit(tmp_path):
    path = tmp_path / "labels.csv"
    update_label(path, "cell", (1, 1), contract=DEFAULT_CONTRACT)
    with pytest.raises(LabelConflictError):
        update_label(path, "cell", (0, 0), contract=DEFAULT_CONTRACT, expected=(1, 0))
    assert load_labels(path).iloc[0][["brdu", "edu"]].tolist() == [1, 1]


def test_three_marker_labels_are_created_and_edited_with_extra_columns_kept(tmp_path):
    config = three_marker_config()
    contract = config.contract
    save_config(config, tmp_path)
    path = tmp_path / "labels.csv"

    assert load_labels(path).columns.tolist() == ["cell", "edu", "brdu", "pcna"]
    assert update_label(path, "b|2", (1, 0, 1), contract=contract) is None
    assert path.read_text() == "cell,edu,brdu,pcna\nb|2,1,0,1\n"
    path.write_text("cell,edu,brdu,pcna,note\nb|2,1,0,1,keep\n")
    assert update_label(path, "a|1", (0, 1, 1), contract=contract, expected=None) is None
    assert update_label(path, "b|2", (1, 1, 0), contract=contract, expected=(1, 0, 1)) == (1, 0, 1)
    table = load_labels(path).set_index("cell")
    assert table.loc["b|2", ["edu", "brdu", "pcna"]].tolist() == [1, 1, 0]
    assert table.loc["a|1", ["edu", "brdu", "pcna"]].tolist() == [0, 1, 1]
    assert table.loc["b|2", "note"] == "keep"
    update_label(path, "a|1", None, contract=contract, expected=(0, 1, 1))
    assert load_labels(path).cell.tolist() == ["b|2"]


def test_three_marker_values_must_be_binary_triples(tmp_path):
    config = three_marker_config()
    save_config(config, tmp_path)
    path = tmp_path / "labels.csv"
    update_label(path, "x|1", (1, 0, 1), contract=config.contract)
    original = path.read_bytes()

    with pytest.raises(ValueError, match="3-tuple of binary labels"):
        update_label(path, "x|1", (1, 0), contract=config.contract)
    assert path.read_bytes() == original


def test_labels_file_must_hold_every_contract_marker_as_binary_values(tmp_path):
    config = three_marker_config()
    save_config(config, tmp_path)
    path = tmp_path / "labels.csv"
    path.write_text("cell,edu,brdu\nx|1,1,0\n")
    with pytest.raises(ValueError, match="missing"):
        load_labels(path)
    path.write_text("cell,edu,brdu,pcna\nx|1,1,0,2\n")
    with pytest.raises(ValueError, match="binary marker values"):
        load_labels(path)
    assert path.read_text() == "cell,edu,brdu,pcna\nx|1,1,0,2\n"


def test_stale_session_cannot_write_after_the_directory_contract_is_reordered(tmp_path):
    path = tmp_path / "labels.csv"
    update_label(path, "x|1", (1, 0), contract=DEFAULT_CONTRACT)
    (tmp_path / "markers.json").write_text(
        '{"version": 1, "reference": {"key": "pi", "label": "PI"},'
        ' "markers": [{"key": "edu", "label": "EdU", "color": "#00ff00"},'
        ' {"key": "brdu", "label": "BrdU", "color": "#ff00ff"}],'
        ' "presentation": {"crop_panels": ["pi", "edu", "brdu"],'
        ' "negative_color": [0.25, 0.25, 0.25], "uncalled_color": "#999999"}}'
    )
    original = path.read_bytes()

    with pytest.raises(ContractMismatchError, match="marker order is part of the contract"):
        update_label(path, "x|1", (0, 1), contract=DEFAULT_CONTRACT)
    with pytest.raises(ContractMismatchError):
        update_label(path, "x|2", (0, 1), contract=DEFAULT_CONTRACT, expected=None)
    assert path.read_bytes() == original

    assert update_label(path, "x|1", (1, 0), contract=REORDERED) == (0, 1)
    assert load_labels(path, contract=REORDERED)[["edu", "brdu"]].values.tolist() == [[1, 0]]


def test_reordered_default_directory_loads_columns_by_name(tmp_path):
    save_config(
        parse_config(
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
        ),
        tmp_path,
    )
    path = tmp_path / "labels.csv"
    assert update_label(path, "x|1", (1, 0), contract=REORDERED) is None
    assert path.read_text() == "cell,edu,brdu\nx|1,1,0\n"
    assert load_labels(path).iloc[0][["edu", "brdu"]].tolist() == [1, 0]
