from __future__ import annotations

import sys

import pandas as pd
import pytest

from dinoapp import legacy_labeler as labeler


def test_audit_top_n_slider_args_skips_slider_for_small_cell_counts() -> None:
    assert labeler._audit_top_n_slider_args(3) is None
    assert labeler._audit_top_n_slider_args(10) is None
    assert labeler._audit_top_n_slider_args(11) == (10, 11, 11, 10)
    assert labeler._audit_top_n_slider_args(57) == (10, 57, 50, 10)


def test_label_tab_cells_excludes_already_labeled_cells() -> None:
    meta = pd.DataFrame(
        {
            "cell": [
                "20251228_JaxA4_Sag4:1|1",
                "20251228_JaxA4_Sag4:1|2",
                "20251224_JaxA4_Sag1:6|3",
            ]
        }
    )
    labels = pd.DataFrame(
        {
            "cell": [
                "20251228_JaxA4_Sag4:1|2",
                "20251224_JaxA4_Sag1:6|3",
            ],
            "brdu": [1, 0],
            "edu": [0, 1],
        }
    )

    assert labeler._label_tab_cells(meta=meta, labels=labels) == ["20251228_JaxA4_Sag4:1|1"]


def test_label_picker_cells_keeps_history_selected_labeled_cell_visible() -> None:
    assert labeler._label_picker_cells(
        available_cells=["20251228_JaxA4_Sag4:1|2", "20251224_JaxA4_Sag1:6|3"],
        current_cell="20251228_JaxA4_Sag4:1|1",
    ) == [
        "20251228_JaxA4_Sag4:1|1",
        "20251228_JaxA4_Sag4:1|2",
        "20251224_JaxA4_Sag1:6|3",
    ]
    assert labeler._label_picker_cells(
        available_cells=["20251228_JaxA4_Sag4:1|1", "20251228_JaxA4_Sag4:1|2"],
        current_cell="20251228_JaxA4_Sag4:1|1",
    ) == [
        "20251228_JaxA4_Sag4:1|1",
        "20251228_JaxA4_Sag4:1|2",
    ]


def test_label_action_specs_exposes_four_one_click_choices() -> None:
    assert labeler._label_action_specs() == [
        ("BrdU only", 1, 0),
        ("EdU only", 0, 1),
        ("Dual pos", 1, 1),
        ("None", 0, 0),
    ]


def test_gallery_matches_filters_identity_and_preserves_barrage_rank() -> None:
    meta = pd.DataFrame(
        {
            "cell": ["c3", "c1", "c2"],
            "rank": [3, 1, 2],
            "image_path": ["3.png", "1.png", "2.png"],
        }
    )
    labels = pd.DataFrame(
        {
            "cell": ["c1", "c2", "c3"],
            "brdu": [1, 0, 1],
            "edu": [0, 1, 0],
        }
    )

    result = labeler._gallery_matches(meta=meta, labels=labels, identity=(1, 0))

    assert result["cell"].tolist() == ["c1", "c3"]
    assert result["rank"].tolist() == [1, 3]


def test_gallery_toggle_changes_only_selected_marker() -> None:
    assert labeler._toggle_marker((0, 1), "brdu") == (1, 1)
    assert labeler._toggle_marker((1, 0), "brdu") == (0, 0)
    assert labeler._toggle_marker((1, 0), "edu") == (1, 1)
    assert labeler._toggle_marker((0, 1), "edu") == (0, 0)
    with pytest.raises(ValueError, match="Unexpected marker"):
        labeler._toggle_marker((0, 0), "other")


def test_launch_starts_streamlit_for_packaged_app(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[tuple[list[str], bool]] = []
    monkeypatch.setattr(sys, "argv", ["dino label", "--server.port", "8502"])
    monkeypatch.setattr(
        labeler.subprocess,
        "run",
        lambda command, check: commands.append((command, check)),
    )

    labeler.launch()

    assert commands == [
        (
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(labeler.Path(labeler.__file__).resolve()),
                "--server.port",
                "8502",
            ],
            True,
        )
    ]
