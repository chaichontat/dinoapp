import copy
import importlib.resources
import json

import pytest

from dino_markers import (
    DEFAULT_CONTRACT,
    BadgeColor,
    Channel,
    ContractMismatchError,
    Marker,
    MarkerConfig,
    MarkerConfigError,
    MarkerContract,
    Presentation,
    config_to_json,
    default_config,
    load_config,
    parse_config,
    require_same_contract,
)


def parse(value):
    return parse_config(value, source="test.json")


def test_packaged_default_equals_the_spec(default_json):
    packaged = importlib.resources.files("dino_markers").joinpath("markers.json").read_text("utf-8")
    assert json.loads(packaged) == default_json


def test_default_config_values():
    config = default_config()
    assert config.reference == Channel("pi", "PI")
    assert config.markers == (
        Marker("brdu", "BrdU", (1.0, 0.0, 1.0)),
        Marker("edu", "EdU", (0.0, 1.0, 0.0)),
    )
    assert config.presentation == Presentation(
        crop_panels=("pi", "edu", "brdu"),
        negative_color=(0.25, 0.25, 0.25),
        uncalled_color=(0.6, 0.6, 0.6),
        badge_colors=(
            BadgeColor((0, 1), (0.0, 143 / 255, 58 / 255), (1.0, 1.0, 1.0)),
            BadgeColor(None, (119 / 255, 119 / 255, 119 / 255), (1.0, 1.0, 1.0)),
        ),
    )
    assert config.contract == DEFAULT_CONTRACT == MarkerContract("pi", ("brdu", "edu"))
    assert config.display_markers == ("edu", "brdu")
    assert config.panels == (("pi", "PI"), ("edu", "EdU"), ("brdu", "BrdU"))
    assert default_config() is default_config()


def test_label_and_marker_lookup():
    config = default_config()
    assert config.label("pi") == "PI"
    assert config.label("edu") == "EdU"
    assert config.marker("brdu").color == (1.0, 0.0, 1.0)
    with pytest.raises(KeyError):
        config.label("dapi")
    with pytest.raises(KeyError):
        config.marker("pi")


def set_markers(value, keys):
    value["markers"] = [
        {"key": key, "label": key.upper() + "!", "color": "#ffffff"} for key in keys
    ]
    value["presentation"]["crop_panels"] = [value["reference"]["key"], *keys]
    value["presentation"].pop("badge_colors", None)
    return value


def test_seven_markers_are_accepted_and_eight_rejected(default_json):
    seven = set_markers(default_json, [f"m{i}" for i in range(7)])
    assert len(parse(seven).markers) == 7
    eight = set_markers(copy.deepcopy(default_json), [f"m{i}" for i in range(8)])
    with pytest.raises(MarkerConfigError, match="between 1 and 7 markers are required, got 8"):
        parse(eight)


def test_zero_markers_rejected(default_json):
    value = set_markers(default_json, [])
    with pytest.raises(MarkerConfigError, match="between 1 and 7 markers are required, got 0"):
        parse(value)


@pytest.mark.parametrize("key", ["labels"])
def test_reserved_keys_rejected(default_json, key):
    value = set_markers(default_json, [key, "edu"])
    with pytest.raises(MarkerConfigError, match=f"key '{key}' is reserved"):
        parse(value)


def test_reserved_derived_name_rejected(default_json):
    value = set_markers(default_json, ["embedding", "edu"])
    with pytest.raises(
        MarkerConfigError, match="key 'embedding' collides with 'embedding_model' through 'embedding_model'"
    ):
        parse(value)


@pytest.mark.parametrize("key", ["BrdU", "_x", "a|b", 3])
def test_invalid_key_syntax_rejected(default_json, key):
    value = set_markers(default_json, ["edu"])
    value["markers"][0]["key"] = key
    value["presentation"]["crop_panels"] = ["pi", key]
    with pytest.raises(MarkerConfigError, match=r"key .* must match \[a-z\]\[a-z0-9_\]\*"):
        parse(value)


def test_reference_key_syntax_rejected(default_json):
    default_json["reference"]["key"] = "PI"
    default_json["presentation"]["crop_panels"] = ["PI", "edu", "brdu"]
    with pytest.raises(MarkerConfigError, match="key 'PI' must match"):
        parse(default_json)


def test_duplicate_key_and_reference_as_marker_rejected(default_json):
    value = set_markers(default_json, ["edu", "edu"])
    with pytest.raises(MarkerConfigError, match="duplicate key 'edu'"):
        parse(value)
    value = set_markers(copy.deepcopy(default_json), ["pi", "edu"])
    with pytest.raises(MarkerConfigError, match="duplicate key 'pi'"):
        parse(value)


def test_derived_name_collision_between_markers(default_json):
    value = set_markers(default_json, ["brdu", "p_brdu"])
    with pytest.raises(MarkerConfigError, match="key 'brdu' collides with 'p_brdu' through 'p_brdu'"):
        parse(value)


def test_derived_name_collision_with_reference(default_json):
    value = set_markers(default_json, ["pi_mean"])
    with pytest.raises(MarkerConfigError, match="collides with"):
        parse(value)


def test_duplicate_and_empty_labels(default_json):
    default_json["markers"][1]["label"] = "BrdU"
    with pytest.raises(MarkerConfigError, match="duplicate label 'BrdU'"):
        parse(default_json)
    default_json["markers"][1]["label"] = "PI"
    with pytest.raises(MarkerConfigError, match="duplicate label 'PI'"):
        parse(default_json)
    default_json["markers"][1]["label"] = ""
    with pytest.raises(MarkerConfigError, match="empty label for 'edu'"):
        parse(default_json)


@pytest.mark.parametrize(
    "panels",
    [["pi", "edu"], ["pi", "edu", "dapi"]],
)
def test_crop_panels_must_be_a_permutation(default_json, panels):
    default_json["presentation"]["crop_panels"] = panels
    with pytest.raises(
        MarkerConfigError, match=r"crop_panels must list each of \['pi', 'brdu', 'edu'\] exactly once"
    ):
        parse(default_json)


def test_crop_panels_order_is_free(default_json):
    default_json["presentation"]["crop_panels"] = ["brdu", "pi", "edu"]
    config = parse(default_json)
    assert config.display_markers == ("brdu", "edu")
    assert config.panels == (("brdu", "BrdU"), ("pi", "PI"), ("edu", "EdU"))
    assert config.contract == DEFAULT_CONTRACT


@pytest.mark.parametrize(
    "color",
    [True, [True, 0, 0], [1, 1, False], "#fff", "ff00ff", "#gg0000", "#ff00ff00", [0, 0], [0, 0, 0, 0],
     [0, 0, 2], [-0.1, 0, 0], [float("nan"), 0, 0], [float("inf"), 0, 0], ["0", 0, 0], 5, None, {}],
)
def test_invalid_colors_rejected(default_json, color):
    default_json["markers"][0]["color"] = color
    with pytest.raises(MarkerConfigError, match=r"^test\.json: invalid color .* for "):
        parse(default_json)


def test_colors_accept_hex_and_triples(default_json):
    default_json["markers"][0]["color"] = "#FF00ff"
    default_json["markers"][1]["color"] = [0, 1, 0.5]
    config = parse(default_json)
    assert config.markers[0].color == (1.0, 0.0, 1.0)
    assert config.markers[1].color == (0.0, 1.0, 0.5)
    assert all(isinstance(c, float) for marker in config.markers for c in marker.color)


@pytest.mark.parametrize("field", ["negative_color", "uncalled_color"])
def test_presentation_colors_validated(default_json, field):
    default_json["presentation"][field] = [True, 0, 0]
    with pytest.raises(MarkerConfigError, match=f"invalid color .* for .*{field}"):
        parse(default_json)


def test_badge_color_validated(default_json):
    default_json["presentation"]["badge_colors"][0]["background"] = "red"
    with pytest.raises(MarkerConfigError, match="invalid color 'red'"):
        parse(default_json)


@pytest.mark.parametrize(
    "state",
    [{"brdu": 1}, {"brdu": True, "edu": 0}, {"brdu": 2, "edu": 0}, [1, 0]],
)
def test_invalid_badge_states_rejected(default_json, state):
    default_json["presentation"]["badge_colors"][0]["state"] = state
    with pytest.raises(MarkerConfigError, match="invalid badge state"):
        parse(default_json)


def test_duplicate_badge_states_rejected(default_json):
    entry = copy.deepcopy(default_json["presentation"]["badge_colors"][0])
    default_json["presentation"]["badge_colors"].append(entry)
    with pytest.raises(MarkerConfigError, match=r"duplicate badge state \(0, 1\)"):
        parse(default_json)
    default_json["presentation"]["badge_colors"] = [
        default_json["presentation"]["badge_colors"][1],
        {"state": None, "background": "#000000", "foreground": "#ffffff"},
    ]
    with pytest.raises(MarkerConfigError, match="duplicate badge state None"):
        parse(default_json)


def test_badge_entry_keys_are_exact(default_json):
    del default_json["presentation"]["badge_colors"][0]["foreground"]
    with pytest.raises(MarkerConfigError, match="missing keys"):
        parse(default_json)
    default_json["presentation"]["badge_colors"][0].update(foreground="#ffffff", extra=1)
    with pytest.raises(MarkerConfigError, match="unexpected keys"):
        parse(default_json)


def test_badge_state_is_in_contract_order(default_json):
    default_json["presentation"]["badge_colors"][0]["state"] = {"edu": 1, "brdu": 0}
    assert parse(default_json).presentation.badge_colors[0].state == (0, 1)


def test_unexpected_and_missing_keys(default_json):
    default_json["extra"] = 1
    with pytest.raises(MarkerConfigError, match=r"^test\.json: unexpected keys \['extra'\]"):
        parse(default_json)
    del default_json["extra"]
    del default_json["presentation"]
    with pytest.raises(MarkerConfigError, match=r"^test\.json: missing keys \['presentation'\]"):
        parse(default_json)


@pytest.mark.parametrize(
    ("path", "key"),
    [(("reference",), "extra"), (("markers", 0), "extra"), (("presentation",), "extra")],
)
def test_nested_unexpected_keys(default_json, path, key):
    node = default_json
    for step in path:
        node = node[step]
    node[key] = 1
    with pytest.raises(MarkerConfigError, match=r"^test\.json: .*unexpected keys \['extra'\]"):
        parse(default_json)


@pytest.mark.parametrize(
    ("path", "key"),
    [(("reference",), "label"), (("markers", 0), "color"), (("presentation",), "crop_panels")],
)
def test_nested_missing_keys(default_json, path, key):
    node = default_json
    for step in path:
        node = node[step]
    del node[key]
    with pytest.raises(MarkerConfigError, match=rf"^test\.json: .*missing keys \['{key}'\]"):
        parse(default_json)


def test_badge_colors_are_optional(default_json):
    del default_json["presentation"]["badge_colors"]
    assert parse(default_json).presentation.badge_colors == ()


@pytest.mark.parametrize("version", [2, "1", None, True, 1.5])
def test_unsupported_version(default_json, version):
    default_json["version"] = version
    with pytest.raises(MarkerConfigError, match=rf"^test\.json: unsupported config version {version!r}; expected 1"):
        parse(default_json)


@pytest.mark.parametrize("value", [[]])
def test_top_level_must_be_an_object(value):
    with pytest.raises(MarkerConfigError, match=r"^test\.json: "):
        parse(value)


def test_errors_are_value_errors():
    for error in (MarkerConfigError, ContractMismatchError):
        assert issubclass(error, ValueError)


def test_direct_construction_cannot_skip_validation():
    presentation = default_config().presentation
    with pytest.raises(MarkerConfigError):
        MarkerConfig(Channel("PI", "PI"), (Marker("edu", "EdU", (0.0, 1.0, 0.0)),), presentation)
    with pytest.raises(MarkerConfigError):
        MarkerConfig(Channel("pi", "PI"), (), presentation)
    with pytest.raises(MarkerConfigError):
        MarkerConfig(
            Channel("pi", "PI"),
            (Marker("edu", "EdU", (0.0, 1.0, 0.0)),),
            presentation,
        )
    with pytest.raises(MarkerConfigError):
        MarkerConfig(
            Channel("pi", "PI"),
            (Marker("edu", "EdU", (True, 1.0, 0.0)),),
            Presentation(("pi", "edu"), (0.0, 0.0, 0.0), (0.5, 0.5, 0.5)),
        )
    with pytest.raises(MarkerConfigError):
        MarkerConfig(
            Channel("pi", "PI"),
            (Marker("edu", "EdU", (0.0, 1.0, 0.0)),),
            Presentation(("pi", "edu"), (0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (BadgeColor((0, 1), (0.0,) * 3, (1.0,) * 3),)),
        )


@pytest.mark.parametrize(
    ("reference", "markers"),
    [("PI", ("a",)), ("pi", ()), ("pi", ("a", "a")), ("pi", ("labels",)), ("pi", ("brdu", "p_brdu")),
     ("pi", ["a"])],
)
def test_contract_construction_applies_key_rules(reference, markers):
    with pytest.raises(MarkerConfigError):
        MarkerContract(reference, markers)


def test_contract_basics():
    contract = MarkerContract("dapi", ("edu", "brdu", "pcna"))
    assert contract.channel_keys == ("dapi", "edu", "brdu", "pcna")
    assert contract.to_json() == {"version": 1, "reference": "dapi", "markers": ["edu", "brdu", "pcna"]}
    assert MarkerContract.from_json(contract.to_json(), where="x") == contract
    assert str(DEFAULT_CONTRACT) == "reference 'pi', markers ['brdu', 'edu']"


@pytest.mark.parametrize(
    ("value", "problem"),
    [
        ({"version": 1, "reference": "pi"}, "missing keys"),
        ({"version": 1, "reference": "pi", "markers": ["a"], "extra": 1}, "unexpected keys"),
        ({"version": 2, "reference": "pi", "markers": ["a"]}, "unsupported config version 2; expected 1"),
        ({"version": 1, "reference": "Pi", "markers": ["a"]}, "key 'Pi' must match"),
        ({"version": 1, "reference": "pi", "markers": "ab"}, "markers"),
        ([], "object"),
        (None, "object"),
    ],
)
def test_contract_from_json_errors(value, problem):
    with pytest.raises(MarkerConfigError, match=rf"^rec\.json: invalid marker_contract: .*{problem}"):
        MarkerContract.from_json(value, where="rec.json")


def round_trips(config):
    value = config_to_json(config)
    assert parse(value) == config
    assert parse(json.loads(json.dumps(value))) == config
    return value


def test_default_config_round_trip_writes_triples():
    value = round_trips(default_config())
    assert value["markers"][0]["color"] == [1.0, 0.0, 1.0]
    assert value["presentation"]["uncalled_color"] == [0.6, 0.6, 0.6]
    assert value["presentation"]["negative_color"] == [0.25, 0.25, 0.25]
    assert value["presentation"]["badge_colors"][0]["state"] == {"brdu": 0, "edu": 1}
    assert value["presentation"]["badge_colors"][1]["state"] is None
    for badge in value["presentation"]["badge_colors"]:
        assert isinstance(badge["background"], list) and len(badge["background"]) == 3
        assert isinstance(badge["foreground"], list) and len(badge["foreground"]) == 3
    assert list(value) == ["version", "reference", "markers", "presentation"]


def test_round_trip_without_badge_colors(default_json):
    del default_json["presentation"]["badge_colors"]
    value = round_trips(parse(default_json))
    assert "badge_colors" not in value["presentation"]


def test_round_trip_three_markers_with_badges(three_json):
    three_json["presentation"]["badge_colors"] = [
        {"state": {"edu": 1, "brdu": 0, "pcna": 1}, "background": "#123456", "foreground": [1, 1, 1]},
        {"state": None, "background": "#000000", "foreground": "#ffffff"},
    ]
    config = parse(three_json)
    value = round_trips(config)
    assert list(value["presentation"]["badge_colors"][0]["state"]) == ["edu", "brdu", "pcna"]
    assert config.presentation.badge_colors[0].state == (1, 0, 1)
    assert config.contract == MarkerContract("dapi", ("edu", "brdu", "pcna"))


def test_load_config(tmp_path, default_json, write):
    path = write(tmp_path / "markers.json", default_json)
    assert load_config(path) == default_config()
    default_json["version"] = 3
    path = write(tmp_path / "bad.json", default_json)
    with pytest.raises(MarkerConfigError, match=rf"^{path}: unsupported config version 3"):
        load_config(path)


def test_load_config_json_errors(tmp_path):
    path = tmp_path / "markers.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(MarkerConfigError, match=rf"^{path}: "):
        load_config(path)
    path.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(MarkerConfigError, match=rf"^{path}: "):
        load_config(path)


def test_require_same_contract():
    require_same_contract(DEFAULT_CONTRACT, MarkerContract("pi", ("brdu", "edu")), expected_from="a", actual_from="b")
    reordered = MarkerContract("pi", ("edu", "brdu"))
    with pytest.raises(ContractMismatchError) as info:
        require_same_contract(DEFAULT_CONTRACT, reordered, expected_from="--markers", actual_from="complete.json")
    assert str(info.value) == (
        "Marker contract mismatch: complete.json records reference 'pi', markers ['edu', 'brdu'], "
        "but --markers requires reference 'pi', markers ['brdu', 'edu']; marker order is part of the contract"
    )
