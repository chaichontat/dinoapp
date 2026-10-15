import itertools

import pytest

from dino_markers import (
    DEFAULT_CONTRACT,
    MAX_MARKERS,
    MarkerContract,
    call_weights,
    decode_call,
    default_config,
    encode_call,
    identity_name,
    matches_saved,
    parse_config,
    state_label,
    toggle_marker,
    uncalled_code,
    validate_label,
)

MINUS = "−"


def contract_of_size(n):
    return MarkerContract("ref", tuple(f"m{i}" for i in range(n)))


def test_default_codes_are_edu_plus_twice_brdu():
    assert call_weights(DEFAULT_CONTRACT) == (2, 1)
    assert uncalled_code(DEFAULT_CONTRACT) == 4
    for brdu, edu in itertools.product((0, 1), repeat=2):
        assert encode_call(DEFAULT_CONTRACT, (brdu, edu)) == edu + 2 * brdu
    assert [encode_call(DEFAULT_CONTRACT, v) for v in ((0, 0), (0, 1), (1, 0), (1, 1))] == [0, 1, 2, 3]
    assert [decode_call(DEFAULT_CONTRACT, c) for c in range(4)] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert decode_call(DEFAULT_CONTRACT, 4) is None


def test_weights_for_three_markers():
    contract = contract_of_size(3)
    assert call_weights(contract) == (4, 2, 1)
    assert encode_call(contract, (1, 0, 1)) == 5
    assert uncalled_code(contract) == 8


@pytest.mark.parametrize("n", range(1, MAX_MARKERS + 1))
def test_call_codes_round_trip(n):
    contract = contract_of_size(n)
    assert call_weights(contract) == tuple(2 ** (n - 1 - i) for i in range(n))
    assert uncalled_code(contract) == 2**n <= 255
    codes = []
    for value in itertools.product((0, 1), repeat=n):
        code = encode_call(contract, value)
        assert decode_call(contract, code) == value
        codes.append(code)
    assert sorted(codes) == list(range(2**n))
    assert decode_call(contract, 2**n) is None


@pytest.mark.parametrize("code", [-1, 5, 100, 1.5, "1", None])
def test_decode_rejects_out_of_range_codes(code):
    with pytest.raises(ValueError):
        decode_call(DEFAULT_CONTRACT, code)


def test_validate_label_accepts_bools_and_returns_ints():
    value = validate_label(DEFAULT_CONTRACT, (True, False))
    assert value == (1, 0)
    assert all(type(x) is int for x in value)
    assert validate_label(DEFAULT_CONTRACT, [0, 1]) == (0, 1)
    assert isinstance(validate_label(DEFAULT_CONTRACT, [0, 1]), tuple)


@pytest.mark.parametrize("value", [(0,), (0, 1, 1), (2, 0), (-1, 0), (0.5, 1), ("1", 0), (None, 0), "01", None, 1, (), [[0], [1]]])
def test_validate_label_rejects(value):
    with pytest.raises(ValueError) as info:
        validate_label(DEFAULT_CONTRACT, value)
    assert str(info.value) == f"Marker values must be a 2-tuple of binary labels, got {value!r}"


def test_validate_label_size_in_message():
    with pytest.raises(ValueError, match="must be a 3-tuple of binary labels"):
        validate_label(contract_of_size(3), (0, 1))


def test_encode_validates():
    with pytest.raises(ValueError):
        encode_call(DEFAULT_CONTRACT, (0, 1, 1))


def test_toggle_marker():
    assert toggle_marker(DEFAULT_CONTRACT, (0, 1), "brdu") == (1, 1)
    assert toggle_marker(DEFAULT_CONTRACT, (0, 1), "edu") == (0, 0)
    assert toggle_marker(contract_of_size(3), (1, 0, 1), "m1") == (1, 1, 1)
    with pytest.raises(ValueError):
        toggle_marker(DEFAULT_CONTRACT, (0, 1), "pi")
    with pytest.raises(ValueError):
        toggle_marker(DEFAULT_CONTRACT, (0, 2), "edu")


def test_matches_saved():
    assert matches_saved((0, 1), (None, None))
    assert matches_saved((0, 1), (0, None))
    assert matches_saved((0, 1), (None, 1))
    assert matches_saved((0, 1), (0, 1))
    assert not matches_saved((0, 1), (1, None))
    assert not matches_saved((0, 1), (None, 0))
    assert not matches_saved((0, 1), (0, 0))
    assert matches_saved((1, 0, 1), (1, None, 1))
    assert not matches_saved((1, 0, 1), (1, 1, None))
    with pytest.raises(ValueError):
        matches_saved((0, 1), (None,))


def test_state_label_default():
    config = default_config()
    assert state_label(config, (0, 0)) == f"BrdU{MINUS} / EdU{MINUS}"
    assert state_label(config, (0, 1)) == f"BrdU{MINUS} / EdU+"
    assert state_label(config, (1, 0)) == f"BrdU+ / EdU{MINUS}"
    assert state_label(config, (1, 1)) == "BrdU+ / EdU+"
    assert "−" in state_label(config, (0, 1)) and "-" not in state_label(config, (0, 1))
    with pytest.raises(ValueError):
        state_label(config, (1,))


def test_state_label_three_markers(three_json):
    config = parse_config(three_json, source="t")
    assert state_label(config, (1, 0, 1)) == f"EdU+ / BrdU{MINUS} / PCNA+"


def test_identity_name_default():
    config = default_config()
    assert identity_name(config, None) == "No saved call"
    assert identity_name(config, (0, 0)) == "Neither"
    assert identity_name(config, (0, 1)) == "EdU only"
    assert identity_name(config, (1, 0)) == "BrdU only"
    assert identity_name(config, (1, 1)) == "Dual positive"
    codes = [decode_call(DEFAULT_CONTRACT, c) for c in range(5)]
    assert tuple(identity_name(config, v) for v in codes) == (
        "Neither", "EdU only", "BrdU only", "Dual positive", "No saved call",
    )


def test_identity_name_three_markers(three_json):
    config = parse_config(three_json, source="t")
    assert identity_name(config, (0, 0, 0)) == "All negative"
    assert identity_name(config, (1, 1, 1)) == "All positive"
    assert identity_name(config, (0, 1, 0)) == "BrdU only"
    assert identity_name(config, (1, 1, 0)) == "EdU + BrdU only"
    assert identity_name(config, (1, 0, 1)) == "EdU + PCNA only"
    assert identity_name(config, None) == "No saved call"


def test_identity_name_single_marker(default_json):
    default_json["markers"] = default_json["markers"][:1]
    default_json["presentation"]["crop_panels"] = ["pi", "brdu"]
    del default_json["presentation"]["badge_colors"]
    config = parse_config(default_json, source="t")
    assert identity_name(config, (0,)) == "All negative"
    assert identity_name(config, (1,)) == "BrdU only"
