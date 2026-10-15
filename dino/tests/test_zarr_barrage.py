from __future__ import annotations

import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
import zarr
from dino_markers import (
    DEFAULT_CONTRACT,
    ContractMismatchError,
    MarkerConfigError,
    config_to_json,
    default_config,
    load_config,
    parse_config,
    save_config,
)
from sklearn.preprocessing import SplineTransformer

from dino import crop_rendering, zarr_barrage
from dino.artifacts import file_sha256
from dino.periodic_pixels import PeriodicPixels
from dino.zarr_barrage import (
    feature_columns,
    masked_channel_stats,
    parse_display_overrides,
    stratified_sample,
)
from dino.zarr_source import CellCrop, build_source_manifest, write_json_atomic


def _config(reference, markers, panels):
    labels = {"pi": "PI", "dapi": "DAPI", "edu": "EdU", "brdu": "BrdU", "pcna": "PCNA"}
    colors = {"edu": "#00ff00", "brdu": "#ff00ff", "pcna": "#00ffff"}
    return parse_config(
        {
            "version": 1,
            "reference": {"key": reference, "label": labels[reference]},
            "markers": [{"key": k, "label": labels[k], "color": colors[k]} for k in markers],
            "presentation": {
                "crop_panels": panels,
                "negative_color": [0.25, 0.25, 0.25],
                "uncalled_color": "#999999",
            },
        },
        source="test",
    )


THREE_MARKERS = _config("dapi", ["edu", "brdu", "pcna"], ["dapi", "pcna", "edu", "brdu"])
REORDERED = _config("pi", ["edu", "brdu"], ["pi", "edu", "brdu"])
REPANELED = _config("pi", ["brdu", "edu"], ["brdu", "pi", "edu"])


def test_masked_channel_stats_excludes_neighbors_without_thresholding() -> None:
    image = np.array([[0, 5, 1000], [7, 9, 2000]], dtype=np.uint16)
    mask = np.array([[True, True, False], [True, True, False]])

    stats = masked_channel_stats(image, mask)

    assert stats == {
        "mean": 5.25,
        "min": 0.0,
        "max": 9.0,
        "median": 6.0,
        "std": np.std([0, 5, 7, 9]),
    }


def test_stratified_sample_is_deterministic_and_bounded() -> None:
    frame = pd.DataFrame(
        {
            "cell": [f"c{i}" for i in range(100)],
            "a": np.arange(100),
            "b": np.arange(100)[::-1],
        }
    )

    first = stratified_sample(frame, columns=("a", "b"), n_rows=20, n_bins=4, seed=3)
    second = stratified_sample(frame, columns=("a", "b"), n_rows=20, n_bins=4, seed=3)

    assert first["cell"].tolist() == second["cell"].tolist()
    assert len(first) == 20
    assert first["cell"].is_unique


def test_render_crop_matches_scaled_legacy_presentation() -> None:
    size = 60
    labels = np.zeros((size, size), dtype=np.uint32)
    labels[20:40, 20:40] = 1
    labels[8:16, 8:16] = 2
    mask = labels == 1
    crop = CellCrop(
        labels=labels,
        mask=mask,
        neighbor_mask=labels == 2,
        channels={
            "pi": np.full((size, size), 4096, dtype=np.uint16),
            "edu": np.full((size, size), 4096, dtype=np.uint16),
            "brdu": np.full((size, size), 4096, dtype=np.uint16),
        },
    )

    image = zarr_barrage.render_crop(
        crop,
        cell="roi|1",
        display_max={"pi": 4096.0, "edu": 8192.0, "brdu": 16_384.0},
        panels=default_config().panels,
    )

    assert image.size == (1440, 480)
    pixels = np.asarray(image)
    panel_brightness = [int(pixels[250, x].sum()) for x in (240, 720, 1200)]
    assert panel_brightness[0] > panel_brightness[1] > panel_brightness[2]


def test_display_max_cache_reuses_matching_source_and_replaces_stale_value(tmp_path, monkeypatch) -> None:
    source = SimpleNamespace(
        manifest={
            "source_sha256": "source-a",
            "labels": {"shape": [4, 20, 20]},
            "offset_zyx": [0, 0, 0],
        },
        channels=dict.fromkeys(("pi", "edu", "brdu")),
    )
    measured = {"pi": 100.0, "edu": 200.0, "brdu": 300.0}
    calls = 0

    def measure(_source):
        nonlocal calls
        calls += 1
        return measured, 400

    monkeypatch.setattr(zarr_barrage, "_measure_display_max", measure)

    assert zarr_barrage._load_or_measure_display_max(tmp_path, source) == measured
    assert json.loads((tmp_path / "display.json").read_text()) == {
        "artifact": "dino-zarr-display-v1",
        "source_sha256": "source-a",
        "percentile": 99.9,
        "z_indices": [0, 1, 2, 3],
        "xy_stride": 8,
        "samples_per_channel": 400,
        "vmax": measured,
    }
    assert zarr_barrage._load_or_measure_display_max(tmp_path, source) == measured
    assert calls == 1

    cache_path = tmp_path / "display.json"
    stale = json.loads(cache_path.read_text())
    stale["source_sha256"] = "source-b"
    cache_path.write_text(json.dumps(stale))

    assert zarr_barrage._load_or_measure_display_max(tmp_path, source) == measured
    assert calls == 2


def test_barrage_writes_existing_labeler_schema(tmp_path, monkeypatch) -> None:
    masked = tmp_path / "dataset" / "roi" / "masked"
    masked.mkdir(parents=True)
    labels_path = masked / "labels.zarr"
    labels = zarr.open_array(labels_path, mode="w", shape=(2, 20, 20), chunks=(1, 10, 10), dtype="u4")
    centers = [(4, 4), (4, 14), (14, 4), (14, 14)]
    for label, (y, x) in enumerate(centers, start=1):
        labels[1, y - 1 : y + 2, x - 1 : x + 2] = label

    channel_paths = {}
    for name, value in (("pi", 100), ("edu", 200), ("brdu", 300)):
        path = tmp_path / f"{name}.zarr"
        array = zarr.open_array(path, mode="w", shape=(2, 30, 30), chunks=(1, 10, 10), dtype="u2")
        array[:] = value
        channel_paths[name] = path

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    manifest = build_source_manifest(
        labels_path=labels_path,
        channel_paths=channel_paths,
        offset_zyx=(0, 5, 5),
        required_channels=DEFAULT_CONTRACT.channel_keys,
    )
    write_json_atomic(manifest, source_dir / "source.json")
    pd.DataFrame(
        [
            {
                "cell": f"roi|{label}",
                "dataset": "dataset",
                "roi": "roi",
                "label": label,
                "area": 9,
                "centroid_z": 1.0,
                "centroid_y": float(y),
                "centroid_x": float(x),
                "plane_z": 1,
                "plane_area": 9,
            }
            for label, (y, x) in enumerate(centers, start=1)
        ]
    ).to_parquet(source_dir / "cells.parquet", index=False)
    eligibility_path = tmp_path / "eligibility.parquet"
    pd.DataFrame(
        {
            "cell": [f"roi|{label}" for label in range(1, 5)],
            "inside_cortex": [True, False, True, False],
        }
    ).to_parquet(eligibility_path, index=False)
    exclude_path = tmp_path / "exclude.csv"
    pd.DataFrame({"cell": ["roi|1"]}).to_csv(exclude_path, index=False)
    out_dir = tmp_path / "barrage"
    monkeypatch.setattr(
        "sys.argv",
        [
            "dino-zarr barrage",
            "--source-dir",
            str(source_dir),
            "--out-dir",
            str(out_dir),
            "--eligibility-table",
            str(eligibility_path),
            "--exclude-index",
            str(exclude_path),
            "--candidate-cells",
            "4",
            "--n-cells",
            "2",
            "--size",
            "60",
            "--cache-gb",
            "0.01",
        ],
    )

    zarr_barrage.main()

    index = pd.read_csv(out_dir / "index.csv")
    required = {
        "rank",
        "cell",
        "image_path",
        "plane_z",
        "brdu_mean",
        "brdu_min",
        "brdu_max",
        "brdu_median",
        "brdu_std",
        "edu_mean",
        "edu_min",
        "edu_max",
        "edu_median",
        "edu_std",
    }
    assert required <= set(index.columns)
    assert len(index) == 1
    assert set(index["cell"]) == {"roi|3"}
    assert set(index["brdu_mean"]) == {300.0}
    assert all((out_dir / path).is_file() for path in index["image_path"])


def test_parse_display_overrides() -> None:
    keys = DEFAULT_CONTRACT.channel_keys
    assert parse_display_overrides(["brdu=22199", "edu=20000.5"], keys=keys) == {
        "brdu": 22199.0,
        "edu": 20000.5,
    }

    with pytest.raises(ValueError, match="Duplicate"):
        parse_display_overrides(["brdu=1", "brdu=2"], keys=keys)
    with pytest.raises(ValueError, match="finite and positive"):
        parse_display_overrides(["brdu=0"], keys=keys)
    with pytest.raises(ValueError, match="Invalid display maximum"):
        parse_display_overrides(["brdu=high"], keys=keys)


def test_display_overrides_accept_exactly_the_contract_channel_keys() -> None:
    keys = THREE_MARKERS.contract.channel_keys

    assert parse_display_overrides(["dapi=1", "pcna=2.5", "edu=3", "brdu=4"], keys=keys) == {
        "dapi": 1.0,
        "pcna": 2.5,
        "edu": 3.0,
        "brdu": 4.0,
    }
    for rejected in ("pi=1", "PCNA=1", "pcna"):
        with pytest.raises(ValueError, match=r"Display maximum must be KEY=VALUE for one of"):
            parse_display_overrides([rejected], keys=keys)


def test_feature_columns_follow_marker_order() -> None:
    assert feature_columns(("edu", "brdu", "pcna")) == (
        "edu_mean",
        "edu_std",
        "brdu_mean",
        "brdu_std",
        "pcna_mean",
        "pcna_std",
    )


def _crop(size: int, channels: dict[str, int]) -> CellCrop:
    labels = np.zeros((size, size), dtype=np.uint32)
    labels[5:11, 5:11] = 1
    return CellCrop(
        labels=labels,
        mask=labels == 1,
        neighbor_mask=np.zeros((size, size), dtype=bool),
        channels={name: np.full((size, size), value, dtype=np.uint16) for name, value in channels.items()},
    )


def test_render_crop_draws_one_titled_panel_per_crop_panel(monkeypatch) -> None:
    import matplotlib.axes

    titles: list[str] = []
    vmax: list[float] = []
    original_title = matplotlib.axes.Axes.set_title
    original_imshow = matplotlib.axes.Axes.imshow

    def record_title(self, label, *args, **kwargs):
        titles.append(label)
        return original_title(self, label, *args, **kwargs)

    def record_imshow(self, data, *args, **kwargs):
        vmax.append(kwargs["vmax"])
        return original_imshow(self, data, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", record_title)
    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", record_imshow)
    crop = _crop(16, {"dapi": 10, "edu": 20, "brdu": 30, "pcna": 40})

    image = crop_rendering.render_crop(
        crop,
        cell="roi|1",
        display_max={"dapi": 100.0, "edu": 200.0, "brdu": 300.0, "pcna": 400.0},
        panels=THREE_MARKERS.panels,
    )

    assert image.size == (512, 128)
    assert titles == [
        "cell=roi|1\nch=DAPI mean=10.0",
        "cell=roi|1\nch=PCNA mean=40.0",
        "cell=roi|1\nch=EdU mean=20.0",
        "cell=roi|1\nch=BrdU mean=30.0",
    ]
    assert vmax == [100.0, 400.0, 200.0, 300.0]


def test_render_crop_rejects_empty_panels_and_absent_channels() -> None:
    crop = _crop(16, {"pi": 10, "edu": 20})
    display_max = {"pi": 1.0, "edu": 1.0, "brdu": 1.0}

    with pytest.raises(ValueError, match="panel"):
        crop_rendering.render_crop(crop, cell="roi|1", display_max=display_max, panels=())
    with pytest.raises(ValueError, match="brdu"):
        crop_rendering.render_crop(
            crop, cell="roi|1", display_max=display_max, panels=default_config().panels
        )


def test_display_cache_round_trips_the_source_channel_set(tmp_path, monkeypatch) -> None:
    channels = ("dapi", "edu", "brdu", "pcna")
    source = SimpleNamespace(
        manifest={"source_sha256": "source-a", "labels": {"shape": [4, 20, 20]}, "offset_zyx": [0, 0, 0]},
        channels=dict.fromkeys(channels),
    )
    measured = dict(zip(channels, (100.0, 200.0, 300.0, 400.0), strict=True))
    calls = 0

    def measure(_source):
        nonlocal calls
        calls += 1
        return measured, 400

    monkeypatch.setattr(zarr_barrage, "_measure_display_max", measure)

    assert zarr_barrage._load_or_measure_display_max(tmp_path, source) == measured
    assert zarr_barrage._load_or_measure_display_max(tmp_path, source) == measured
    assert calls == 1
    assert set(json.loads((tmp_path / "display.json").read_text())["vmax"]) == set(channels)


def test_display_cache_rejects_a_channel_set_other_than_the_source(tmp_path, monkeypatch) -> None:
    channels = ("dapi", "edu", "brdu", "pcna")
    source = SimpleNamespace(
        manifest={"source_sha256": "source-a", "labels": {"shape": [4, 20, 20]}, "offset_zyx": [0, 0, 0]},
        channels=dict.fromkeys(channels),
    )
    monkeypatch.setattr(
        zarr_barrage, "_measure_display_max", lambda _source: (dict.fromkeys(channels, 100.0), 400)
    )
    zarr_barrage._load_or_measure_display_max(tmp_path, source)
    cache_path = tmp_path / "display.json"
    cached = json.loads(cache_path.read_text())
    del cached["vmax"]["pcna"]
    cache_path.write_text(json.dumps(cached))

    with pytest.raises(ValueError, match="Invalid display maxima in cache"):
        zarr_barrage._load_or_measure_display_max(tmp_path, source)


THREE_MARKER_CHANNEL_VALUES = {"dapi": 100, "edu": 200, "brdu": 300, "pcna": 400}


def _write_source(tmp_path, channel_values, *, config=None):
    """A one-cell prepared source holding `channel_values`; `config` is stored as its markers.json."""
    labels_path = tmp_path / "dataset" / "roi" / "masked" / "labels.zarr"
    labels = zarr.open_array(labels_path, mode="w", shape=(1, 16, 16), chunks=(1, 8, 8), dtype="u4")
    labels[0, 6:10, 6:10] = 1
    channels = {}
    for name, value in channel_values.items():
        path = tmp_path / f"{name}.zarr"
        array = zarr.open_array(path, mode="w", shape=(1, 16, 16), chunks=(1, 8, 8), dtype="u2")
        array[:] = value
        channels[name] = path
    source_dir = tmp_path / "source"
    manifest = build_source_manifest(
        labels_path=labels_path,
        channel_paths=channels,
        offset_zyx=(0, 0, 0),
        required_channels=tuple(channel_values),
    )
    write_json_atomic(manifest, source_dir / "source.json")
    pd.DataFrame([{
        "cell": "roi|1", "dataset": "dataset", "roi": "roi", "label": 1,
        "area": 16, "centroid_z": 0.0, "centroid_y": 8.0, "centroid_x": 8.0,
        "plane_z": 0, "plane_area": 16,
    }]).to_parquet(source_dir / "cells.parquet", index=False)
    if config is not None:
        save_config(config, source_dir)
    return source_dir


@pytest.fixture
def run_barrage(monkeypatch):
    def run(source_dir, out_dir, *extra):
        monkeypatch.setattr("sys.argv", [
            "barrage", "--source-dir", str(source_dir), "--out-dir", str(out_dir),
            "--candidate-cells", "1", "--n-cells", "1", "--size", "8",
            "--cache-gb", "0.001", *extra,
        ])
        zarr_barrage.main()

    return run


@pytest.fixture
def barrage_run(tmp_path, run_barrage):
    source_dir = _write_source(tmp_path, {"pi": 100, "brdu": 300, "edu": 200})

    def run(out_dir, *extra):
        run_barrage(source_dir, out_dir, *extra)

    run.source_dir = source_dir
    return run


@pytest.fixture
def three_marker_source(tmp_path):
    return _write_source(tmp_path / "three", THREE_MARKER_CHANNEL_VALUES, config=THREE_MARKERS)


def _legacy_record(record):
    """The dino-zarr-barrage-v3 form of a fresh default barrage record."""
    legacy = {key: value for key, value in record.items() if key not in ("marker_contract", "crop_panels")}
    legacy["artifact"] = "dino-zarr-barrage-v3"
    return legacy


def _tree(root):
    return {path.relative_to(root): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def test_fresh_default_barrage_writes_v4_records_and_markers(barrage_run, tmp_path) -> None:
    out_dir = tmp_path / "barrage"

    barrage_run(out_dir)

    run = json.loads((out_dir / "run.json").read_text())
    assert run["artifact"] == "dino-zarr-barrage-v4"
    assert run["marker_contract"] == DEFAULT_CONTRACT.to_json()
    assert run["crop_panels"] == [["pi", "PI"], ["edu", "EdU"], ["brdu", "BrdU"]]
    assert json.loads((out_dir / "complete.json").read_text()) == run
    assert load_config(out_dir / "markers.json") == default_config()


def test_three_marker_barrage_measures_every_marker_and_renders_crop_panels(
    run_barrage, three_marker_source, tmp_path
) -> None:
    from PIL import Image

    out_dir = tmp_path / "barrage"

    run_barrage(three_marker_source, out_dir)

    run = json.loads((out_dir / "run.json").read_text())
    assert run["marker_contract"] == THREE_MARKERS.contract.to_json()
    assert run["crop_panels"] == [["dapi", "DAPI"], ["pcna", "PCNA"], ["edu", "EdU"], ["brdu", "BrdU"]]
    assert run["display_max"] == {name: float(value) for name, value in THREE_MARKER_CHANNEL_VALUES.items()}
    assert load_config(out_dir / "markers.json") == THREE_MARKERS
    statistics = [f"{marker}_{stat}" for marker in ("edu", "brdu", "pcna") for stat in ("mean", "min", "max", "median", "std")]
    index = pd.read_csv(out_dir / "index.csv")
    assert index.columns.tolist()[-len(statistics) :] == statistics
    assert not {"dapi_mean", "pi_mean"} & set(index.columns)
    assert index.loc[0, ["edu_mean", "brdu_mean", "pcna_mean"]].tolist() == [200.0, 300.0, 400.0]
    candidates = pd.read_parquet(out_dir / "candidates.parquet")
    assert candidates.columns.tolist()[-len(statistics) :] == statistics
    with Image.open(out_dir / index.loc[0, "image_path"]) as image:
        assert image.size == (256, 64)


def test_three_marker_destination_with_seeded_labels_inherits_the_source_contract(
    run_barrage, three_marker_source, tmp_path
) -> None:
    out_dir = tmp_path / "barrage"
    out_dir.mkdir()
    seeded = b"cell,edu,brdu,pcna\nroi|1,1,0,1\n"
    (out_dir / "labels.csv").write_bytes(seeded)

    run_barrage(three_marker_source, out_dir)

    assert json.loads((out_dir / "run.json").read_text())["marker_contract"] == (
        THREE_MARKERS.contract.to_json()
    )
    assert (out_dir / "labels.csv").read_bytes() == seeded
    assert pd.read_csv(out_dir / "index.csv").loc[0, "cell"] == "roi|1"


def test_existing_labels_must_hold_every_marker_column(run_barrage, three_marker_source, tmp_path) -> None:
    out_dir = tmp_path / "barrage"
    out_dir.mkdir()
    (out_dir / "labels.csv").write_text("cell,edu,brdu\nroi|1,1,0\n")

    with pytest.raises(ValueError, match="pcna"):
        run_barrage(three_marker_source, out_dir)

    assert not (out_dir / "complete.json").exists()


def test_display_max_follows_the_contract_channels(run_barrage, three_marker_source, tmp_path) -> None:
    out_dir = tmp_path / "barrage"

    run_barrage(three_marker_source, out_dir, "--display-max", "pcna=5000", "--display-max", "dapi=900.5")

    run = json.loads((out_dir / "run.json").read_text())
    assert run["display_overrides"] == {"pcna": 5000.0, "dapi": 900.5}
    assert run["display_max"] == {"dapi": 900.5, "edu": 200.0, "brdu": 300.0, "pcna": 5000.0}


def test_source_channels_must_cover_the_contract(barrage_run, tmp_path) -> None:
    save_config(THREE_MARKERS, barrage_run.source_dir)

    with pytest.raises(ValueError, match=r"dapi.*pcna"):
        barrage_run(tmp_path / "barrage")

    assert not (tmp_path / "barrage" / "run.json").exists()


def test_reordered_markers_file_is_rejected_for_a_default_source(barrage_run, tmp_path) -> None:
    markers = tmp_path / "markers.json"
    markers.write_text(json.dumps(config_to_json(REORDERED)))

    with pytest.raises(ContractMismatchError, match="marker order is part of the contract"):
        barrage_run(tmp_path / "barrage", "--markers", str(markers))

    assert not (tmp_path / "barrage" / "run.json").exists()


def test_reordered_source_contract_is_rejected_for_an_existing_default_barrage(barrage_run, tmp_path) -> None:
    out_dir = tmp_path / "barrage"
    barrage_run(out_dir)
    save_config(REORDERED, barrage_run.source_dir)
    before = _tree(out_dir)

    with pytest.raises(ContractMismatchError, match="marker order is part of the contract"):
        barrage_run(out_dir)

    assert _tree(out_dir) == before


def test_three_marker_source_is_rejected_for_an_existing_default_barrage(
    barrage_run, run_barrage, three_marker_source, tmp_path
) -> None:
    out_dir = tmp_path / "barrage"
    barrage_run(out_dir)
    before = _tree(out_dir)

    with pytest.raises(ContractMismatchError):
        run_barrage(three_marker_source, out_dir)

    assert _tree(out_dir) == before


def test_supplied_markers_must_match_the_source_contract(run_barrage, three_marker_source, tmp_path) -> None:
    markers = tmp_path / "markers.json"
    markers.write_text(json.dumps(config_to_json(default_config())))

    with pytest.raises(ContractMismatchError, match="--markers"):
        run_barrage(three_marker_source, tmp_path / "barrage", "--markers", str(markers))


def test_invalid_markers_file_is_rejected(barrage_run, tmp_path) -> None:
    markers = tmp_path / "markers.json"
    markers.write_text("{}")

    with pytest.raises(MarkerConfigError):
        barrage_run(tmp_path / "barrage", "--markers", str(markers))


def test_legacy_partial_run_resumes_as_v3_without_markers(barrage_run, tmp_path) -> None:
    baseline = tmp_path / "baseline"
    barrage_run(baseline)
    legacy = _legacy_record(json.loads((baseline / "run.json").read_text()))
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "run.json").write_text(json.dumps(legacy))

    barrage_run(partial)

    assert json.loads((partial / "run.json").read_text()) == legacy
    assert json.loads((partial / "complete.json").read_text()) == legacy
    assert not (partial / "markers.json").exists()
    assert (partial / "index.csv").read_bytes() == (baseline / "index.csv").read_bytes()


@pytest.mark.parametrize("as_record", [_legacy_record, dict], ids=["legacy_v3", "current"])
def test_partial_run_rejects_changed_crop_panels(barrage_run, tmp_path, as_record) -> None:
    baseline = tmp_path / "baseline"
    barrage_run(baseline)
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "run.json").write_text(json.dumps(as_record(json.loads((baseline / "run.json").read_text()))))
    save_config(REPANELED, barrage_run.source_dir)

    with pytest.raises(ValueError, match="different rendering contract"):
        barrage_run(partial)

    assert not (partial / "complete.json").exists()


def test_completed_legacy_barrage_without_run_record_is_skipped(barrage_run, tmp_path, capsys) -> None:
    baseline = tmp_path / "baseline"
    barrage_run(baseline)
    completed = tmp_path / "completed"
    completed.mkdir()
    legacy = _legacy_record(json.loads((baseline / "complete.json").read_text()))
    (completed / "complete.json").write_text(json.dumps(legacy))
    capsys.readouterr()

    barrage_run(completed)

    assert capsys.readouterr().out.splitlines()[-1] == "status=skipped_complete"
    assert [path.name for path in completed.iterdir()] == ["complete.json"]


@pytest.mark.parametrize("complete_version", ["v3", "v4"])
def test_completed_barrage_whose_run_record_has_another_version_is_not_skipped(
    barrage_run, tmp_path, complete_version
) -> None:
    baseline = tmp_path / "baseline"
    barrage_run(baseline)
    current = json.loads((baseline / "run.json").read_text())
    legacy = _legacy_record(current)
    completed = tmp_path / "completed"
    completed.mkdir()
    complete, run = (legacy, current) if complete_version == "v3" else (current, legacy)
    (completed / "complete.json").write_text(json.dumps(complete))
    (completed / "run.json").write_text(json.dumps(run))
    before = _tree(completed)

    with pytest.raises(ValueError, match="different rendering contract"):
        barrage_run(completed)

    assert _tree(completed) == before


def _write_field(directory, targets):
    spline = SplineTransformer(
        knots=np.tile(np.linspace(0, 1, 5)[:, None], (1, 2)), extrapolation="periodic"
    ).fit([[0.0, 0.0], [1.0, 1.0]])
    field = directory / "field.joblib"
    joblib.dump(
        dict(
            period_xy_um=np.array([11.0, 17.0]),
            periodic_spline=spline,
            periodic_center=np.full(16, 1 / 16),
            coefficient=np.random.default_rng(7).normal(0, 0.5, (16, len(targets))),
            targets=list(targets),
        ),
        field,
    )
    return PeriodicPixels(str(field), file_sha256(field), (1.0, 1.0, 1.0)).identity


def test_pixel_model_with_other_periodic_targets_is_rejected(barrage_run, tmp_path) -> None:
    manifest = json.loads((barrage_run.source_dir / "source.json").read_text())
    identity = _write_field(tmp_path, ("edu", "brdu"))
    model = tmp_path / "model.joblib"
    joblib.dump({"source_sha256": manifest["source_sha256"], "pixel_correction": identity}, model)

    with pytest.raises(ContractMismatchError, match="model.joblib"):
        barrage_run(tmp_path / "barrage", "--pixel-model", str(model))

    assert not (tmp_path / "barrage" / "run.json").exists()


def test_interrupted_render_does_not_publish_png(barrage_run, tmp_path, monkeypatch) -> None:
    from PIL import Image

    out_dir = tmp_path / "interrupted"

    def interrupted_save(self, path, *args, **kwargs):
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
        raise OSError("interrupted PNG save")

    monkeypatch.setattr(Image.Image, "save", interrupted_save)
    with pytest.raises(OSError, match="interrupted PNG save"):
        barrage_run(out_dir)
    assert not list(out_dir.glob("images-*/*.png"))
    assert not (out_dir / "complete.json").exists()


def test_partial_resume_rejects_corrupt_png(barrage_run, tmp_path) -> None:
    baseline = tmp_path / "baseline"
    barrage_run(baseline)
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "run.json").write_bytes((baseline / "run.json").read_bytes())
    relative = pd.read_csv(baseline / "index.csv").loc[0, "image_path"]
    image = partial / relative
    image.parent.mkdir()
    corrupt = b"\x89PNG\r\n\x1a\n"
    image.write_bytes(corrupt)

    with pytest.raises(ValueError, match="PNG"):
        barrage_run(partial)

    assert image.read_bytes() == corrupt
    assert not (partial / "complete.json").exists()


def test_reused_index_refreshes_reviewer_paths(barrage_run, tmp_path) -> None:
    from dinoapp.sources import BarrageSource

    baseline = tmp_path / "baseline"
    barrage_run(baseline)
    index = pd.read_csv(baseline / "index.csv")
    for marker in ("combined", "brdu", "edu"):
        index[f"{marker}_image_path"] = f"old_{marker}.png"
    index["review_round"] = 2
    reference = tmp_path / "reference.csv"
    index.to_csv(reference, index=False)
    refreshed = tmp_path / "refreshed"

    barrage_run(refreshed, "--reuse-index", str(reference))

    refreshed_index = pd.read_csv(refreshed / "index.csv")
    assert refreshed_index["review_round"].tolist() == [2]
    reviewer = BarrageSource(refreshed)
    image_bytes = (refreshed / refreshed_index.loc[0, "image_path"]).read_bytes()
    for marker in ("combined", "brdu", "edu"):
        assert reviewer.crop("roi|1", marker).image == image_bytes
    assert not {"combined_image_path", "brdu_image_path", "edu_image_path"} & set(refreshed_index)


@pytest.mark.parametrize("state", ["complete", "unrecognized"])
def test_no_resume_rejects_occupied_barrage_unchanged(barrage_run, tmp_path, state) -> None:
    occupied = tmp_path / "occupied"
    if state == "complete":
        barrage_run(occupied)
    else:
        occupied.mkdir()
        (occupied / "keep.txt").write_text("existing output")
    before = {p.relative_to(occupied): p.read_bytes() for p in occupied.rglob("*") if p.is_file()}

    with pytest.raises(ValueError, match="resume"):
        barrage_run(occupied, "--no-resume")

    assert {p.relative_to(occupied): p.read_bytes() for p in occupied.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("state", ["partial", "complete"])
def test_resume_reuses_valid_published_png(barrage_run, tmp_path, monkeypatch, state) -> None:
    baseline = tmp_path / "baseline"
    barrage_run(baseline)
    out_dir = tmp_path / "resumed"
    out_dir.mkdir()
    (out_dir / "run.json").write_bytes((baseline / "run.json").read_bytes())
    relative = pd.read_csv(baseline / "index.csv").loc[0, "image_path"]
    image = out_dir / relative
    image.parent.mkdir()
    original = (baseline / relative).read_bytes()
    image.write_bytes(original)
    if state == "complete":
        (out_dir / "complete.json").write_bytes((baseline / "complete.json").read_bytes())
        monkeypatch.setattr(
            zarr_barrage.Image, "open", lambda *args: pytest.fail("completed run must skip image audit")
        )
    monkeypatch.setattr(
        zarr_barrage, "render_crop", lambda *args, **kwargs: pytest.fail("published image must be reused")
    )

    barrage_run(out_dir)

    assert image.read_bytes() == original
    assert (out_dir / "complete.json").is_file()
