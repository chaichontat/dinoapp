import json
import re
from pathlib import Path

import pytest

from dino_markers import (
    MarkerConfigError,
    ReviewTarget,
    config_to_json,
    load_config,
    load_review_targets,
    parse_config,
    save_config,
)

CHANNELS = {"561": "pi", "BrdU": "brdu", "EdU": "edu"}


def entry(**changes):
    return {
        "name": "Prod3", "adapter": "prod3", "root": "data", "review_root": "data/review",
        "cell_prefix": "prod3-v1", "raw_channels": dict(CHANNELS), **changes,
    }


@pytest.fixture
def config_file(tmp_path, default_json, write):
    def write_targets(targets):
        return write(tmp_path / "lab" / "markers.json", {**default_json, "review_targets": targets})

    return write_targets


def test_targets_resolve_relative_paths_against_the_file_directory(tmp_path, config_file):
    other = tmp_path / "elsewhere"
    path = config_file([
        entry(),
        entry(name="Prod3-2", adapter="prod3_2", root=str(other), review_root="../review",
              raw_channels={"DAPI": "pi", "BrdU": "brdu", "EdU": "edu"}, embedding_run=str(other / "run")),
    ])

    first, second = load_review_targets(path)

    lab = tmp_path / "lab"
    assert first == ReviewTarget("Prod3", "prod3", lab / "data", lab / "data/review", "prod3-v1", CHANNELS)
    assert first.embedding_run is None
    assert second.root == other
    assert second.review_root == lab / "../review"
    assert second.embedding_run == other / "run"
    assert second.raw_channels == {"DAPI": "pi", "BrdU": "brdu", "EdU": "edu"}
    assert hash(first) == hash(ReviewTarget("Prod3", "prod3", lab / "data", lab / "data/review", "prod3-v1", CHANNELS))


def test_relative_config_path_resolves_against_its_directory(tmp_path, config_file, monkeypatch):
    config_file([entry()])
    monkeypatch.chdir(tmp_path)

    (target,) = load_review_targets(Path("lab/markers.json"))

    assert target.root == tmp_path / "lab/data"


def test_a_file_without_review_targets_has_none(tmp_path, default_json, write):
    assert load_review_targets(write(tmp_path / "markers.json", default_json)) == ()


@pytest.mark.parametrize(
    ("targets", "problem"),
    [
        ({}, r"review_targets must be a list"),
        (["Prod3"], r"review_targets\[0\]: expected a JSON object"),
        ([entry(name="")], r"review_targets\[0\]\.name must be a non-empty string"),
        ([entry(adapter=3)], r"review_targets\[0\]\.adapter must be a non-empty string"),
        ([entry(root=None)], r"review_targets\[0\]\.root must be a non-empty string"),
        ([entry(cell_prefix=1)], r"review_targets\[0\]\.cell_prefix must be a string"),
        ([entry(), entry()], r"duplicate review target name 'Prod3'"),
        ([entry(dataset="Prod3")], r"review_targets\[0\]: unexpected keys \['dataset'\]"),
        ([{k: v for k, v in entry().items() if k != "name"}], r"review_targets\[0\]: missing keys \['name'\]"),
        ([entry(raw_channels={"561": "pi", "BrdU": "brdu"})], r"review_targets\[0\]\.raw_channels must map one raw array to each of"),
        ([entry(raw_channels={**CHANNELS, "DAPI": "pi"})], r"review_targets\[0\]\.raw_channels must map one raw array to each of"),
        ([entry(raw_channels={**CHANNELS, "PCNA": "pcna"})], r"review_targets\[0\]\.raw_channels must map one raw array to each of"),
        ([entry(raw_channels=["pi", "brdu", "edu"])], r"review_targets\[0\]\.raw_channels must map one raw array to each of"),
    ],
)
def test_invalid_targets_are_rejected_with_the_file_path(config_file, targets, problem):
    path = config_file(targets)

    with pytest.raises(MarkerConfigError, match=rf"^{re.escape(str(path))}: {problem}"):
        load_review_targets(path)


def test_raw_channels_follow_the_files_own_contract(tmp_path, three_json, write):
    channels = {"DAPI": "dapi", "EdU": "edu", "BrdU": "brdu", "PCNA": "pcna"}
    path = write(tmp_path / "markers.json", {**three_json, "review_targets": [entry(raw_channels=channels)]})

    (target,) = load_review_targets(path)

    assert target.raw_channels == channels


def test_review_targets_are_not_part_of_the_marker_configuration(tmp_path, config_file, default_json):
    path = config_file([entry()])
    config = load_config(path)

    assert config == parse_config(default_json, source="spec")
    assert "review_targets" not in config_to_json(config)
    saved = save_config(config, tmp_path / "lab")
    assert saved == path
    assert load_review_targets(path)[0].name == "Prod3"

    directory = tmp_path / "run"
    directory.mkdir()
    assert "review_targets" not in json.loads(save_config(config, directory).read_text())
