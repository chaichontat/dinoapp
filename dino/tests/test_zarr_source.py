from __future__ import annotations

import numpy as np
import pytest
import zarr

from dino import zarr_source
from dino.zarr_source import (
    CellCrop,
    CellCropReader,
    build_source_manifest,
    cell_dino_images,
    load_source,
    manifest_digest,
    open_review_source,
    neighbor_cell_intensity,
    write_json_atomic,
)
from dino_markers import DEFAULT_CONTRACT


def test_crop_reader_uses_target_mask_and_fused_offset(tmp_path) -> None:
    labels = zarr.open_array(
        tmp_path / "labels.zarr", mode="w", shape=(2, 6, 6), chunks=(1, 3, 3), dtype="u4"
    )
    labels[1, 2:5, 2:5] = 11
    labels[1, 0:2, 0:2] = 12

    channels = {}
    for name, value in (("pi", 100), ("edu", 200), ("brdu", 300)):
        array = zarr.open_array(
            tmp_path / f"{name}.zarr",
            mode="w",
            shape=(2, 10, 10),
            chunks=(1, 4, 4),
            dtype="u2",
        )
        array[:] = value
        channels[name] = array
    channels["edu"][1, 4:7, 5:8] = 900

    reader = CellCropReader(
        labels=labels,
        channels=channels,
        offset_zyx=(0, 2, 3),
        cache_bytes=2_000_000,
    )
    crop = reader.read_cell(label=11, plane_z=1, center_y=3, center_x=3, size=5)

    assert crop.mask.sum() == 9
    assert np.all(crop.channels["edu"][crop.mask] == 900)
    assert np.all(crop.channels["brdu"][crop.mask] == 300)
    assert np.any(crop.neighbor_mask)


def test_cell_dino_images_respects_explicit_channel_order() -> None:
    mask = np.array([[True, False], [False, True]])
    crop = CellCrop(
        labels=np.zeros((2, 2), dtype=np.uint32),
        mask=mask,
        neighbor_mask=np.zeros((2, 2), dtype=bool),
        channels={
            "brdu": np.full((2, 2), 256, dtype=np.uint16),
            "marker": np.full((2, 2), 512, dtype=np.uint16),
            "edu": np.full((2, 2), 768, dtype=np.uint16),
        },
    )

    image = cell_dino_images(
        crop,
        input_channels=4,
        channel_names_by_target={"brdu": ("brdu", "marker", "edu")},
    )["brdu"]

    assert np.array_equal(image[:, mask], [[1, 1], [2, 2], [3, 3], [0, 0]])
    assert not image[:, ~mask].any()


def test_cell_dino_highpass_removes_offset_without_rescaling_signal() -> None:
    mask = np.ones((31, 31), dtype=bool)
    signal = np.zeros((31, 31), dtype=np.uint16)
    signal[15, 15] = 8192

    def render(offset: int) -> np.ndarray:
        crop = CellCrop(
            labels=np.ones((31, 31), dtype=np.uint32),
            mask=mask,
            neighbor_mask=np.zeros((31, 31), dtype=bool),
            channels={"brdu": signal + offset},
        )
        return cell_dino_images(
            crop,
            input_channels=4,
            channel_names_by_target={"brdu": ("brdu",)},
            highpass_sigma_px=10.0,
        )["brdu"]

    zero_background = render(0)
    raised_background = render(4096)

    assert 0 < zero_background[0, 15, 15] < 32
    assert zero_background[0, 15, 15] == raised_background[0, 15, 15]
    assert not zero_background[1:].any()


def test_cell_dino_images_erodes_mask_by_disk_radius() -> None:
    crop = CellCrop(
        labels=np.ones((7, 7), dtype=np.uint32),
        mask=np.ones((7, 7), dtype=bool),
        neighbor_mask=np.zeros((7, 7), dtype=bool),
        channels={"brdu": np.full((7, 7), 256, dtype=np.uint16)},
    )

    image = cell_dino_images(
        crop,
        input_channels=4,
        channel_names_by_target={"brdu": ("brdu",)},
        mask_erosion_px=2,
    )["brdu"]

    assert np.count_nonzero(image[0]) == 9
    assert np.all(image[0, 2:5, 2:5] == 1)
    assert not image[1:].any()


def test_cell_dino_images_filters_each_unique_channel_once(monkeypatch) -> None:
    crop = CellCrop(
        labels=np.ones((5, 5), dtype=np.uint32),
        mask=np.ones((5, 5), dtype=bool),
        neighbor_mask=np.zeros((5, 5), dtype=bool),
        channels={
            "brdu": np.full((5, 5), 256, dtype=np.uint16),
            "edu": np.full((5, 5), 512, dtype=np.uint16),
            "561": np.full((5, 5), 768, dtype=np.uint16),
        },
    )
    calls: list[int] = []

    def filter_once(signal, **kwargs):
        calls.append(1)
        return np.zeros_like(signal)

    monkeypatch.setattr(zarr_source, "gaussian_filter", filter_once)
    images = cell_dino_images(
        crop,
        input_channels=4,
        channel_names_by_target={
            "brdu": ("brdu", "561"),
            "edu": ("edu", "561"),
        },
        highpass_sigma_px=20.0,
    )

    assert len(calls) == 3
    np.testing.assert_array_equal(images["brdu"][:, 0, 0], [1, 3, 0, 0])
    np.testing.assert_array_equal(images["edu"][:, 0, 0], [2, 3, 0, 0])


def test_neighbor_cell_intensity_uses_nearest_cell_means() -> None:
    labels = np.zeros((9, 9), dtype=np.uint32)
    labels[4, 4] = 1
    labels[4, 5:7] = 2
    labels[1:3, 4] = 3
    labels[7:9, 7:9] = 4
    channel = np.zeros((9, 9), dtype=np.uint16)
    channel[labels == 2] = 100
    channel[labels == 3] = 300
    channel[labels == 4] = 900
    crop = CellCrop(
        labels=labels,
        mask=labels == 1,
        neighbor_mask=(labels != 0) & (labels != 1),
        channels={"brdu": channel},
    )

    intensity, count = neighbor_cell_intensity(crop, channel_name="brdu", max_neighbors=2)

    assert intensity == 200
    assert count == 2


def test_neighbor_cell_intensity_reports_missing_neighborhood() -> None:
    labels = np.ones((3, 3), dtype=np.uint32)
    crop = CellCrop(
        labels=labels,
        mask=np.ones((3, 3), dtype=bool),
        neighbor_mask=np.zeros((3, 3), dtype=bool),
        channels={"brdu": np.full((3, 3), 100, dtype=np.uint16)},
    )

    intensity, count = neighbor_cell_intensity(crop, channel_name="brdu", max_neighbors=20)

    assert np.isnan(intensity)
    assert count == 0


def test_crop_reader_prefetches_each_required_chunk_once() -> None:
    class Array:
        shape = (1, 8, 8)
        chunks = (1, 4, 4)

        def __init__(self, dtype) -> None:
            self.dtype = np.dtype(dtype)
            self.calls = 0

        def __getitem__(self, selection):
            self.calls += 1
            return np.zeros(tuple(part.stop - part.start for part in selection), dtype=self.dtype)

    labels = Array(np.uint32)
    channel = Array(np.uint16)
    reader = CellCropReader(
        labels=labels,
        channels={"brdu": channel},
        offset_zyx=(0, 0, 0),
        cache_bytes=1_000_000,
    )
    cells = [(0, 3, 3)]

    reader.prefetch_cells(cells, size=4, workers=2)
    reader.read_cell(label=1, plane_z=0, center_y=3, center_x=3, size=4)

    assert labels.calls == 4
    assert channel.calls == 4
    reader.close()


def test_crop_reader_rejects_oversized_plan_before_loading() -> None:
    class Array:
        shape = (1, 8, 8)
        chunks = (1, 4, 4)
        dtype = np.dtype(np.uint32)

        def __init__(self) -> None:
            self.calls = 0

        def __getitem__(self, selection):
            self.calls += 1
            return np.zeros(tuple(part.stop - part.start for part in selection), dtype=self.dtype)

    labels = Array()
    reader = CellCropReader(
        labels=labels,
        channels={},
        offset_zyx=(0, 0, 0),
        cache_bytes=32,
    )
    plan = reader.plan_cells([(0, 3, 3)], size=4)

    with pytest.raises(RuntimeError, match="increase --cache-gb or reduce --batch-size"):
        reader.prefetch_plan(plan, workers=2)

    assert plan.decoded_bytes == 256
    assert labels.calls == 0


def test_load_source_rejects_recorded_chunk_mismatch(tmp_path) -> None:
    labels_path = tmp_path / "dataset" / "roi" / "masked" / "labels.zarr"
    labels_path.parent.mkdir(parents=True)
    zarr.open_array(labels_path, mode="w", shape=(1, 4, 4), chunks=(1, 2, 2), dtype="u4")
    channels = {}
    for name in ("pi", "edu", "brdu"):
        path = tmp_path / f"{name}.zarr"
        zarr.open_array(path, mode="w", shape=(1, 4, 4), chunks=(1, 2, 2), dtype="u2")
        channels[name] = path
    manifest = build_source_manifest(
        labels_path=labels_path,
        channel_paths=channels,
        offset_zyx=(0, 0, 0),
        required_channels=DEFAULT_CONTRACT.channel_keys,
    )
    manifest["labels"]["chunks"] = [1, 4, 4]
    manifest["source_sha256"] = manifest_digest(manifest)
    source_dir = tmp_path / "source"
    write_json_atomic(manifest, source_dir / "source.json")

    with pytest.raises(ValueError, match="masked labels no longer matches"):
        load_source(source_dir)
    with pytest.raises(ValueError, match="display source metadata differs for labels"):
        open_review_source(source_dir)


@pytest.mark.integration
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_source_fingerprints_stored_channel_and_label_content(tmp_path, zarr_format):
    labels_path = tmp_path / "labels.zarr"
    labels = zarr.open_array(labels_path, mode="w", shape=(1, 2, 2), chunks=(1, 2, 2), dtype="u4", zarr_format=zarr_format)
    labels[:] = 1
    channel_path = tmp_path / "image.zarr"
    group = zarr.open_group(channel_path, mode="w", zarr_format=zarr_format)
    channel = group.create_array("0", shape=(1, 2, 2), chunks=(1, 2, 2), dtype="u2")
    channel[:] = 256
    paths = dict.fromkeys(("brdu", "edu", "pi"), channel_path)
    original = build_source_manifest(
        labels_path=labels_path,
        channel_paths=paths,
        offset_zyx=(0, 0, 0),
        required_channels=DEFAULT_CONTRACT.channel_keys,
    )
    source_dir = tmp_path / "source"
    write_json_atomic(original, source_dir / "source.json")
    load_source(source_dir)
    open_review_source(source_dir)

    channel[0, 0, 0] = 512

    assert open_review_source(source_dir).channels["brdu"][0, 0, 0] == 512

    with pytest.raises(ValueError, match="channel .*no longer matches source.json"):
        load_source(source_dir)
    updated = build_source_manifest(
        labels_path=labels_path,
        channel_paths=paths,
        offset_zyx=(0, 0, 0),
        required_channels=DEFAULT_CONTRACT.channel_keys,
    )
    assert updated["source_sha256"] != original["source_sha256"]
    assert updated["channels"]["brdu"]["metadata_sha256"] == original["channels"]["brdu"]["metadata_sha256"]
    write_json_atomic(updated, source_dir / "source.json")
    labels[:] = 0  # Removing an encoded all-fill chunk must also change identity.
    with pytest.raises(ValueError, match="masked labels no longer matches source.json"):
        load_source(source_dir)
    relabeled = build_source_manifest(
        labels_path=labels_path,
        channel_paths=paths,
        offset_zyx=(0, 0, 0),
        required_channels=DEFAULT_CONTRACT.channel_keys,
    )
    assert relabeled["source_sha256"] != updated["source_sha256"]


def _source_inputs(tmp_path, names):
    labels_path = tmp_path / "dataset" / "roi" / "masked" / "labels.zarr"
    labels_path.parent.mkdir(parents=True)
    zarr.open_array(labels_path, mode="w", shape=(1, 4, 4), chunks=(1, 2, 2), dtype="u4")
    channels = {}
    for name in names:
        path = tmp_path / f"{name}.zarr"
        zarr.open_array(path, mode="w", shape=(1, 4, 4), chunks=(1, 2, 2), dtype="u2")
        channels[name] = path
    return labels_path, channels


def test_manifest_requires_the_channels_of_a_custom_contract(tmp_path) -> None:
    labels_path, channels = _source_inputs(tmp_path, ("dapi", "edu", "brdu", "pcna"))

    manifest = build_source_manifest(
        labels_path=labels_path,
        channel_paths=channels,
        offset_zyx=(0, 0, 0),
        required_channels=("dapi", "edu", "brdu", "pcna"),
    )

    assert set(manifest["channels"]) == {"dapi", "edu", "brdu", "pcna"}
    with pytest.raises(ValueError, match=r"Missing required channel sources: \['pcna'\]"):
        build_source_manifest(
            labels_path=labels_path,
            channel_paths={key: path for key, path in channels.items() if key != "pcna"},
            offset_zyx=(0, 0, 0),
            required_channels=("dapi", "edu", "brdu", "pcna"),
        )


def test_default_channels_are_not_required_by_a_custom_contract(tmp_path) -> None:
    labels_path, channels = _source_inputs(tmp_path, ("dapi", "edu"))

    manifest = build_source_manifest(
        labels_path=labels_path, channel_paths=channels, offset_zyx=(0, 0, 0), required_channels=("dapi", "edu")
    )

    assert set(manifest["channels"]) == {"dapi", "edu"}
    with pytest.raises(ValueError, match=r"Missing required channel sources: \['brdu', 'pi'\]"):
        build_source_manifest(
            labels_path=labels_path,
            channel_paths=channels,
            offset_zyx=(0, 0, 0),
            required_channels=DEFAULT_CONTRACT.channel_keys,
        )


def test_required_channel_order_does_not_change_the_manifest_or_digest(tmp_path) -> None:
    labels_path, channels = _source_inputs(tmp_path, ("pi", "edu", "brdu"))

    manifests = [
        build_source_manifest(
            labels_path=labels_path, channel_paths=channels, offset_zyx=(0, 0, 0), required_channels=required
        )
        for required in (("pi", "brdu", "edu"), ("pi", "edu", "brdu"), ("edu", "brdu", "pi"))
    ]

    assert manifests[0] == manifests[1] == manifests[2]
    assert manifests[0]["source_sha256"] == manifests[1]["source_sha256"] == manifests[2]["source_sha256"]
