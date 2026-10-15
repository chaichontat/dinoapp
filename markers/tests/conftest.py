import copy
import json
from pathlib import Path

import pytest

DEFAULT_TEXT = """
{
  "version": 1,
  "reference": {"key": "pi", "label": "PI"},
  "markers": [
    {"key": "brdu", "label": "BrdU", "color": "#ff00ff"},
    {"key": "edu", "label": "EdU", "color": "#00ff00"}
  ],
  "presentation": {
    "crop_panels": ["pi", "edu", "brdu"],
    "negative_color": [0.25, 0.25, 0.25],
    "uncalled_color": "#999999",
    "badge_colors": [
      {"state": {"brdu": 0, "edu": 1}, "background": "#008f3a", "foreground": "#ffffff"},
      {"state": null, "background": "#777777", "foreground": "#ffffff"}
    ]
  }
}
"""

THREE = {
    "version": 1,
    "reference": {"key": "dapi", "label": "DAPI"},
    "markers": [
        {"key": "edu", "label": "EdU", "color": "#00ff00"},
        {"key": "brdu", "label": "BrdU", "color": "#ff00ff"},
        {"key": "pcna", "label": "PCNA", "color": [0.0, 0.0, 1.0]},
    ],
    "presentation": {
        "crop_panels": ["dapi", "edu", "brdu", "pcna"],
        "negative_color": "#202020",
        "uncalled_color": "#999999",
    },
}


@pytest.fixture
def default_json():
    return json.loads(DEFAULT_TEXT)


@pytest.fixture
def three_json():
    return copy.deepcopy(THREE)


@pytest.fixture
def write():
    def write_json(path: Path, value) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    return write_json
