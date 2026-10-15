import json
import os

import pytest

from dino_markers import (
    CONTRACT_FILES,
    DEFAULT_CONTRACT,
    BARRAGE,
    PREPARED_SOURCE,
    SPATIAL_REVIEW,
    ContractMismatchError,
    MarkerConfigError,
    MarkerContract,
    UnsupportedArtifactError,
    config_to_json,
    contract_of,
    declared_contract,
    default_config,
    load_config,
    marker_revision,
    parse_config,
    resolve_directory,
    save_config,
)

REORDERED = MarkerContract("pi", ("edu", "brdu"))
CUSTOM = MarkerContract("dapi", ("edu", "brdu", "pcna"))


def custom_config(three_json):
    return parse_config(three_json, source="three")


def reordered_json(default_json):
    default_json["markers"].reverse()
    default_json["presentation"].pop("badge_colors")
    return default_json


def test_contract_files():
    assert CONTRACT_FILES == ("complete.json", "run.json", "spatial.json", "markers.json")


def test_missing_and_empty_directories_declare_nothing(tmp_path):
    for directory in (tmp_path / "absent", tmp_path):
        assert declared_contract(directory) is None
        assert contract_of(directory) == DEFAULT_CONTRACT


def test_legacy_prepared_source(tmp_path, write):
    write(tmp_path / "complete.json", {"artifact": "dino-zarr-source-complete-v1", "n": 1})
    assert declared_contract(tmp_path) == DEFAULT_CONTRACT
    assert contract_of(tmp_path) == DEFAULT_CONTRACT


def test_v3_barrage_record(tmp_path, write):
    write(tmp_path / "run.json", {"artifact": "dino-zarr-barrage-v3"})
    write(tmp_path / "complete.json", {"artifact": "dino-zarr-barrage-v3"})
    assert declared_contract(tmp_path) == DEFAULT_CONTRACT


def test_current_records_declare_their_contract(tmp_path, write):
    write(tmp_path / "complete.json", PREPARED_SOURCE.stamp({}, CUSTOM))
    write(tmp_path / "run.json", BARRAGE.stamp({}, CUSTOM))
    assert declared_contract(tmp_path) == CUSTOM
    assert contract_of(tmp_path) == CUSTOM


def test_current_record_without_the_field_fails(tmp_path, write):
    write(tmp_path / "complete.json", {"artifact": "dino-zarr-barrage-v4"})
    with pytest.raises(ValueError, match=rf"^{tmp_path / 'complete.json'}: barrage record"):
        declared_contract(tmp_path)


@pytest.mark.parametrize("prefix", ["dino-zarr-barrage-v", "dino-zarr-source-complete-v"])
def test_unknown_version_in_a_known_family(tmp_path, write, prefix):
    write(tmp_path / "complete.json", {"artifact": f"{prefix}9"})
    with pytest.raises(UnsupportedArtifactError, match=rf"^{tmp_path / 'complete.json'}: unsupported "):
        declared_contract(tmp_path)


def test_handmade_directory_with_undecodable_complete_and_custom_markers(tmp_path, three_json, write):
    (tmp_path / "complete.json").write_bytes(b"\xff\xfe\x00 not json")
    write(tmp_path / "markers.json", three_json)
    assert declared_contract(tmp_path) == CUSTOM
    assert contract_of(tmp_path) == CUSTOM


@pytest.mark.parametrize("content", [b"{broken", b"[1, 2]"])
def test_opaque_complete_json_is_skipped(tmp_path, content):
    (tmp_path / "complete.json").write_bytes(content)
    assert declared_contract(tmp_path) is None


@pytest.mark.parametrize("content", [b"{broken", b"\xff\xfe\x00", b"[1, 2]"])
def test_run_json_must_be_an_object(tmp_path, content):
    (tmp_path / "run.json").write_bytes(content)
    with pytest.raises(MarkerConfigError, match=rf"^{tmp_path / 'run.json'}: run.json must be a JSON object$"):
        declared_contract(tmp_path)


def test_foreign_records_are_skipped(tmp_path, write):
    write(tmp_path / "run.json", {"artifact": "solarmap-native-barrage-v1", "rois": 3})
    write(tmp_path / "complete.json", {"artifact": "solarmap-native-barrage-v1"})
    assert declared_contract(tmp_path) is None
    assert contract_of(tmp_path) == DEFAULT_CONTRACT


@pytest.mark.parametrize("record", [{"artifact": 7}])
def test_records_without_a_string_artifact_are_skipped(tmp_path, write, record):
    write(tmp_path / "run.json", record)
    assert declared_contract(tmp_path) is None


def test_foreign_record_carrying_a_contract_fails(tmp_path, write):
    write(tmp_path / "run.json", {"artifact": "solarmap-native-barrage-v1", "marker_contract": CUSTOM.to_json()})
    with pytest.raises(
        UnsupportedArtifactError,
        match=rf"^{tmp_path / 'run.json'}: unrecognized record 'solarmap-native-barrage-v1' carries marker_contract$",
    ):
        declared_contract(tmp_path)


def test_record_without_artifact_carrying_a_contract_fails(tmp_path, write):
    write(tmp_path / "run.json", {"marker_contract": CUSTOM.to_json()})
    with pytest.raises(UnsupportedArtifactError, match="unrecognized record None carries marker_contract"):
        declared_contract(tmp_path)


def test_v3_record_with_reordered_markers_mismatches(tmp_path, default_json, write):
    write(tmp_path / "run.json", {"artifact": "dino-zarr-barrage-v3"})
    write(tmp_path / "markers.json", reordered_json(default_json))
    with pytest.raises(ContractMismatchError) as info:
        declared_contract(tmp_path)
    assert str(tmp_path / "run.json") in str(info.value)
    assert str(tmp_path / "markers.json") in str(info.value)


def test_v4_record_with_matching_markers(tmp_path, three_json, write):
    write(tmp_path / "run.json", BARRAGE.stamp({}, CUSTOM))
    write(tmp_path / "markers.json", three_json)
    assert declared_contract(tmp_path) == CUSTOM


def test_complete_and_run_disagreeing(tmp_path, write):
    write(tmp_path / "complete.json", BARRAGE.stamp({}, CUSTOM))
    write(tmp_path / "run.json", BARRAGE.stamp({}, REORDERED))
    with pytest.raises(ContractMismatchError) as info:
        declared_contract(tmp_path)
    assert str(tmp_path / "complete.json") in str(info.value)
    assert str(tmp_path / "run.json") in str(info.value)


def test_invalid_markers_json_raises(tmp_path, default_json, write):
    default_json["version"] = 9
    write(tmp_path / "markers.json", default_json)
    with pytest.raises(MarkerConfigError, match="unsupported config version 9"):
        declared_contract(tmp_path)


def test_spatial_json_versions(tmp_path, write):
    write(tmp_path / "spatial.json", {"artifact": "rhu2-spatial-review-v1", "n": 1})
    assert declared_contract(tmp_path) == DEFAULT_CONTRACT
    write(tmp_path / "spatial.json", SPATIAL_REVIEW.stamp({}, CUSTOM))
    assert declared_contract(tmp_path) == CUSTOM


def test_spatial_json_must_be_recognized(tmp_path, write):
    write(tmp_path / "spatial.json", {"artifact": "something-else"})
    with pytest.raises(UnsupportedArtifactError, match=rf"^{tmp_path / 'spatial.json'}: "):
        declared_contract(tmp_path)
    write(tmp_path / "spatial.json", [1])
    with pytest.raises(MarkerConfigError, match=rf"^{tmp_path / 'spatial.json'}: spatial.json must be a JSON object"):
        declared_contract(tmp_path)


def test_spatial_json_disagrees_with_markers(tmp_path, write, default_json):
    write(tmp_path / "spatial.json", {"artifact": "rhu2-spatial-review-v1"})
    write(tmp_path / "markers.json", reordered_json(default_json))
    with pytest.raises(ContractMismatchError):
        contract_of(tmp_path)


def test_resolve_directory_default_paths(tmp_path, write):
    assert resolve_directory(tmp_path) == default_config()
    write(tmp_path / "run.json", {"artifact": "dino-zarr-barrage-v3"})
    assert resolve_directory(tmp_path) == default_config()


def test_resolve_directory_uses_markers_json(tmp_path, three_json, write):
    write(tmp_path / "run.json", BARRAGE.stamp({}, CUSTOM))
    write(tmp_path / "markers.json", three_json)
    assert resolve_directory(tmp_path) == custom_config(three_json)


def test_resolve_directory_supplied_config_only_asserts(tmp_path, default_json, write):
    write(tmp_path / "run.json", {"artifact": "dino-zarr-barrage-v3"})
    default_json["markers"][0]["color"] = "#ff00fe"
    supplied = parse_config(default_json, source="s")
    assert resolve_directory(tmp_path, supplied=supplied) == supplied


def test_resolve_directory_supplied_reordered_config_raises(tmp_path, default_json, write):
    write(tmp_path / "run.json", {"artifact": "dino-zarr-barrage-v3"})
    supplied = parse_config(reordered_json(default_json), source="s")
    with pytest.raises(ContractMismatchError, match=r"^Marker contract mismatch: --markers records "):
        resolve_directory(tmp_path, supplied=supplied)
    with pytest.raises(ContractMismatchError, match=r"^Marker contract mismatch: given\.json records "):
        resolve_directory(tmp_path, supplied=supplied, supplied_from="given.json")


def test_resolve_directory_custom_record_without_markers_json_raises(tmp_path, write):
    write(tmp_path / "run.json", BARRAGE.stamp({}, CUSTOM))
    with pytest.raises(ContractMismatchError, match="packaged default markers.json records"):
        resolve_directory(tmp_path)


def test_save_config_writes_exact_text(tmp_path):
    config = default_config()
    path = save_config(config, tmp_path)
    assert path == tmp_path / "markers.json"
    assert path.read_text(encoding="utf-8") == json.dumps(config_to_json(config), indent=2) + "\n"
    assert load_config(path) == config
    assert sorted(item.name for item in tmp_path.iterdir()) == ["markers.json"]


def test_save_config_is_a_noop_for_an_equal_config(tmp_path):
    path = save_config(default_config(), tmp_path)
    before = path.stat()
    assert save_config(default_config(), tmp_path) == path
    after = path.stat()
    assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)
    assert sorted(item.name for item in tmp_path.iterdir()) == ["markers.json"]


def test_save_config_equal_config_ignores_file_formatting(tmp_path, default_json, write):
    write(tmp_path / "markers.json", default_json)
    before = (tmp_path / "markers.json").read_bytes()
    save_config(default_config(), tmp_path)
    assert (tmp_path / "markers.json").read_bytes() == before


def test_save_config_conflicts(tmp_path, three_json, default_json):
    path = save_config(default_config(), tmp_path)
    before = path.read_bytes()
    with pytest.raises(MarkerConfigError) as info:
        save_config(custom_config(three_json), tmp_path)
    assert str(info.value) == (
        f"{path} differs from this run's marker configuration; "
        "supply that file with --markers or use a new output directory"
    )
    default_json["markers"][0]["color"] = "#ff00fe"
    with pytest.raises(MarkerConfigError, match="differs from this run's marker configuration"):
        save_config(parse_config(default_json, source="s"), tmp_path)
    assert path.read_bytes() == before
    assert sorted(item.name for item in tmp_path.iterdir()) == ["markers.json"]


def test_marker_revision_format_and_changes(tmp_path):
    directory = tmp_path / "d"
    directory.mkdir()
    assert marker_revision([directory]) == json.dumps([[str(directory), [None, None, None, None]]])
    first = marker_revision([directory])
    path = save_config(default_config(), directory)
    second = marker_revision([directory])
    assert second != first
    assert second == json.dumps([[str(directory), [None, None, None, os.stat(path).st_mtime_ns]]])
    os.utime(path, ns=(1_000_000_000, 2_000_000_000))
    third = marker_revision([directory])
    assert third == json.dumps([[str(directory), [None, None, None, 2_000_000_000]]])
    assert marker_revision([directory]) == third


def test_marker_revision_order_and_multiple_directories(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    (a / "run.json").write_text("{}")
    os.utime(a / "run.json", ns=(1, 5))
    value = json.loads(marker_revision(iter([a, b])))
    assert value == [[str(a), [None, 5, None, None]], [str(b), [None, None, None, None]]]
    assert marker_revision([a, b]) != marker_revision([b, a])
    assert marker_revision([]) == "[]"


def test_save_config_file_mode_follows_the_umask(tmp_path):
    plain = tmp_path / "plain"
    plain.write_text("x")
    path = save_config(default_config(), tmp_path)
    assert path.stat().st_mode == plain.stat().st_mode


def test_save_config_reports_a_missing_directory_as_the_original_error(tmp_path):
    with pytest.raises(FileNotFoundError) as info:
        save_config(default_config(), tmp_path / "absent")
    assert info.value.__context__ is None
