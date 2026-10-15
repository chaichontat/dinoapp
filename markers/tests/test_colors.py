import math
import re

import pytest

from dino_markers import (
    MarkerConfigError,
    badge_colors,
    decode_call,
    default_config,
    hex_color,
    identity_color,
    parse_color,
    parse_config,
)

PALETTE = [(0.25, 0.25, 0.25), (0, 1, 0), (1, 0, 1), (1, 1, 1), (0.6, 0.6, 0.6)]
FIELD_MARKS = ["#404040", "#00ff00", "#ff00ff", "#ffffff", "#999999"]
BADGES = [
    ("#404040", "#ffffff"),
    ("#008f3a", "#ffffff"),
    ("#ff00ff", "#ffffff"),
    ("#ffffff", "#202020"),
    ("#777777", "#ffffff"),
]


def default_states():
    contract = default_config().contract
    return [decode_call(contract, code) for code in range(5)]


def test_parse_color():
    assert parse_color("#ff00ff", where="x") == (1.0, 0.0, 1.0)
    assert parse_color("#FF00FF", where="x") == (1.0, 0.0, 1.0)
    assert parse_color("#999999", where="x") == (0.6, 0.6, 0.6)
    assert parse_color([0, 1, 0.5], where="x") == (0.0, 1.0, 0.5)
    assert parse_color((0.25, 0.25, 0.25), where="x") == (0.25, 0.25, 0.25)
    assert all(type(c) is float for c in parse_color([0, 1, 0], where="x"))


@pytest.mark.parametrize(
    "value",
    ["#fff", "ff00ff", "#gg0000", "#ff00ff0", [True, 0, 0], [0, 0], [0, 0, 1.01], [math.nan, 0, 0], [math.inf, 0, 0], 1, None, ["0", 0, 0]],
)
def test_parse_color_rejects(value):
    with pytest.raises(MarkerConfigError, match=re.escape(f"invalid color {value!r} for here")):
        parse_color(value, where="here")


def test_hex_color():
    assert hex_color((1.0, 0.0, 1.0)) == "#ff00ff"
    assert hex_color((0.25, 0.25, 0.25)) == "#404040"
    assert hex_color((0.6, 0.6, 0.6)) == "#999999"
    assert hex_color((0.0, 0.0, 0.0)) == "#000000"
    assert hex_color((1.0, 1.0, 1.0)) == "#ffffff"
    assert hex_color(parse_color("#008f3a", where="x")) == "#008f3a"


def test_identity_color_default_equals_the_dinoapp_palette():
    config = default_config()
    colors = [identity_color(config, v) for v in default_states()]
    assert colors == [tuple(float(c) for c in row) for row in PALETTE]
    assert [hex_color(c) for c in colors] == FIELD_MARKS


def test_identity_color_clips_and_sums_in_contract_order(three_json):
    three_json["markers"][0]["color"] = [0.75, 0.5, 0.0]
    three_json["markers"][1]["color"] = [0.5, 0.5, 0.25]
    three_json["markers"][2]["color"] = [0.0, 0.25, 1.0]
    config = parse_config(three_json, source="t")
    assert identity_color(config, (1, 0, 0)) == (0.75, 0.5, 0.0)
    assert identity_color(config, (1, 1, 0)) == (1.0, 1.0, 0.25)
    assert identity_color(config, (0, 1, 1)) == (0.5, 0.75, 1.0)
    assert identity_color(config, (1, 1, 1)) == (1.0, 1.0, 1.0)
    assert identity_color(config, (0, 0, 0)) == (32 / 255,) * 3
    assert identity_color(config, None) == (0.6, 0.6, 0.6)


def test_identity_color_validates(three_json):
    config = parse_config(three_json, source="t")
    with pytest.raises(ValueError):
        identity_color(config, (1, 0))


def test_default_badges_equal_dinoapp_native_app():
    config = default_config()
    assert [badge_colors(config, v) for v in default_states()] == BADGES


def test_badges_without_overrides_use_identity_color_and_luminance(three_json):
    config = parse_config(three_json, source="t")
    assert badge_colors(config, (0, 0, 0)) == ("#202020", "#ffffff")
    assert badge_colors(config, (1, 0, 0)) == ("#00ff00", "#202020")
    assert badge_colors(config, (0, 1, 0)) == ("#ff00ff", "#ffffff")
    assert badge_colors(config, (0, 0, 1)) == ("#0000ff", "#ffffff")
    assert badge_colors(config, (1, 1, 1)) == ("#ffffff", "#202020")
    assert badge_colors(config, (1, 1, 0)) == ("#ffffff", "#202020")
    assert badge_colors(config, None) == ("#999999", "#ffffff")


def test_badge_override_matches_state(three_json):
    three_json["presentation"]["badge_colors"] = [
        {"state": {"edu": 1, "brdu": 0, "pcna": 1}, "background": "#102030", "foreground": "#fefefe"},
        {"state": None, "background": [0, 0, 0], "foreground": [1, 1, 1]},
    ]
    config = parse_config(three_json, source="t")
    assert badge_colors(config, (1, 0, 1)) == ("#102030", "#fefefe")
    assert badge_colors(config, None) == ("#000000", "#ffffff")
    assert badge_colors(config, (1, 0, 0)) == ("#00ff00", "#202020")


def test_luminance_threshold_follows_the_linearized_formula(default_json):
    del default_json["presentation"]["badge_colors"]
    # a gray c has luminance ((c + 0.055) / 1.055) ** 2.4, which reaches 0.5 at byte 187.5
    default_json["presentation"]["negative_color"] = "#bbbbbb"
    assert badge_colors(parse_config(default_json, source="t"), (0, 0))[1] == "#ffffff"
    default_json["presentation"]["negative_color"] = "#bcbcbc"
    assert badge_colors(parse_config(default_json, source="t"), (0, 0))[1] == "#202020"
    default_json["presentation"]["negative_color"] = "#ff0000"
    assert badge_colors(parse_config(default_json, source="t"), (0, 0))[1] == "#ffffff"
    default_json["presentation"]["negative_color"] = "#ffff00"
    assert badge_colors(parse_config(default_json, source="t"), (0, 0)) == ("#ffff00", "#202020")
