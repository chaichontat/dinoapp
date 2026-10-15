import joblib
import numpy as np
import pandas as pd
import pytest
import zarr
from sklearn.preprocessing import SplineTransformer

from dino.artifacts import file_sha256
from dino.pixel_corrections import correct_channels
from dino.periodic import correction_gain
from dino.periodic_pixels import PeriodicPixels, require_periodic_targets
from dino.zarr_model import iter_model_image_batches, sort_cells_for_io
from dino.zarr_source import CellCropReader
from dino_markers import DEFAULT_CONTRACT, ContractMismatchError, MarkerContract
from squisher_segment.segment.regionprops import measure_zarr


def write_field(directory, targets, *, width=None, name="field.joblib"):
    spline = SplineTransformer(
        knots=np.tile(np.linspace(0, 1, 5)[:, None], (1, 2)), extrapolation="periodic"
    ).fit([[0.0, 0.0], [1.0, 1.0]])
    coefficient = np.random.default_rng(7).normal(0, 0.5, (16, len(targets) if width is None else width))
    bundle = dict(
        period_xy_um=np.array([11.0, 17.0]),
        periodic_spline=spline,
        periodic_center=np.full(16, 1 / 16),
        coefficient=coefficient,
        targets=targets,
    )
    path = directory / name
    joblib.dump(bundle, path)
    return PeriodicPixels(str(path), file_sha256(path), (1.0, 1.0, 1.0)), bundle


@pytest.fixture
def field(tmp_path):
    return write_field(tmp_path, ["brdu", "edu"])


def test_tiled_pixels_match_pointwise_field_across_xy_and_z(field):
    transform, bundle = field
    raw = np.arange(2 * 13 * 19, dtype=np.uint16).reshape(2, 13, 19)
    for y0, x0 in [(953, 957), (-4, -3)]:
        yy, xx = np.mgrid[y0 : y0 + 13, x0 : x0 + 19]
        xy = np.column_stack([xx.ravel(), yy.ravel()])
        gain = correction_gain(xy, bundle).reshape(13, 19, 2).astype(np.float32)
        for j, target in enumerate(["brdu", "edu"]):
            corrected = transform(target, raw, (4, y0, x0))
            np.testing.assert_allclose(corrected, raw.astype(np.float32) * gain[..., j], rtol=1e-6)
            np.testing.assert_array_equal(corrected, transform(target, raw, (800, y0, x0)))
    assert raw.dtype == np.uint16


def make_arrays(tmp_path):
    labels = zarr.open_array(
        tmp_path / "labels.zarr",
        mode="w",
        shape=(2, 8, 8),
        chunks=(1, 4, 4),
        dtype="u4",
    )
    labels[:] = 1
    channels = {}
    for target, offset in [("brdu", 1000), ("edu", 3000)]:
        array = zarr.open_array(
            tmp_path / f"{target}.zarr",
            mode="w",
            shape=labels.shape,
            chunks=labels.chunks,
            dtype="u2",
        )
        array[:] = np.arange(128).reshape(labels.shape) * 10 + offset
        channels[target] = array
    return labels, channels


@pytest.mark.parametrize("workers", [1, 2])
def test_regionprops_reduces_corrected_pixels_across_chunk_seams(tmp_path, field, workers):
    transform, _ = field
    labels, channels = make_arrays(tmp_path)
    output = tmp_path / "props.parquet"
    measure_zarr(
        tmp_path / "labels.zarr",
        output,
        intensity_paths={t: tmp_path / f"{t}.zarr" for t in channels},
        intensity_transform=transform,
        workers=workers,
    )
    row = pd.read_parquet(output).iloc[0]
    for target, array in channels.items():
        corrected = transform(target, array[:], (0, 0, 0))
        assert row[f"intensity_{target}_min"] == corrected.min()
        assert row[f"intensity_{target}_max"] == corrected.max()
        np.testing.assert_allclose(row[f"intensity_{target}_mean"], corrected.astype(float).mean(), rtol=1e-6)
    assert row.area == 128


def test_training_and_inference_batch_path_corrects_before_quantization(tmp_path, field):
    transform, _ = field
    labels, channels = make_arrays(tmp_path)
    reader = CellCropReader(labels=labels, channels=channels, offset_zyx=(0, 0, 0), cache_bytes=1000000)
    rows = pd.DataFrame(dict(cell=["cell-1"], label=[1], plane_z=[0], centroid_y=[4.0], centroid_x=[4.0]))
    bundle = dict(
        schema="dino-zarr-model-v6",
        crop_size=8,
        input_channels=4,
        highpass_sigma_px=0,
        mask_erosion_px=0,
        pixel_correction=transform.identity,
    )
    batches = list(
        iter_model_image_batches(
            sort_cells_for_io(rows, reader),
            bundle=bundle,
            channel_names_by_target={"brdu": ("brdu",), "edu": ("edu",)},
            reader=reader,
            batch_size=1,
            prefetch_workers=1,
            crop_workers=2,
            require_mask=True,
        )
    )
    images, _ = batches[0]
    for target in channels:
        corrected = transform(target, channels[target][0], (0, 0, 0))
        expected = np.clip(corrected / 256.0, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(images[target][0][0], expected)
    reader.close()


def test_rejects_changed_correction_model(tmp_path, field):
    transform, _ = field
    # A new identity cannot reuse a model loaded under another digest.
    invalid = PeriodicPixels(transform.path, "0" * 64, transform.scale_zyx_um)
    with pytest.raises(ValueError, match="hash mismatch"):
        invalid("brdu", np.ones((2, 2), dtype=np.uint16), (0, 0, 0))


def test_barrage_reuses_review_order_with_corrected_intensities(tmp_path, field, monkeypatch):
    from dino import zarr_barrage
    from dino.zarr_source import build_source_manifest, write_json_atomic

    transform, _ = field
    labels, channels = make_arrays(tmp_path)
    labels[:, :, 4:] = 2
    pi_path = tmp_path / "pi.zarr"
    pi = zarr.open_array(pi_path, mode="w", shape=labels.shape, chunks=labels.chunks, dtype="u2")
    pi[:] = 200
    source = tmp_path / "source"
    source.mkdir()
    manifest = build_source_manifest(
        labels_path=tmp_path / "labels.zarr",
        channel_paths={"pi": pi_path, **{t: tmp_path / f"{t}.zarr" for t in channels}},
        offset_zyx=(0, 0, 0),
        required_channels=DEFAULT_CONTRACT.channel_keys,
    )
    write_json_atomic(manifest, source / "source.json")
    pd.DataFrame(
        dict(
            cell=["roi|1", "roi|2"],
            dataset=["test"] * 2,
            roi=["roi"] * 2,
            label=[1, 2],
            area=[64, 64],
            centroid_z=[0.0, 0.0],
            centroid_y=[4.0, 4.0],
            centroid_x=[2.0, 6.0],
            plane_z=[0, 0],
            plane_area=[32, 32],
        )
    ).to_parquet(source / "cells.parquet")
    index = tmp_path / "index.csv"
    pd.DataFrame(dict(cell=["roi|2", "roi|1"], review_round=[0, 1], bin_x=[3, 4])).to_csv(index, index=False)
    model = tmp_path / "model.joblib"
    joblib.dump(dict(source_sha256=manifest["source_sha256"], pixel_correction=transform.identity), model)
    out = tmp_path / "barrage"
    monkeypatch.setattr(
        "sys.argv",
        [
            "dino-zarr barrage",
            "--source-dir",
            str(source),
            "--out-dir",
            str(out),
            "--pixel-model",
            str(model),
            "--reuse-index",
            str(index),
            "--candidate-cells",
            "2",
            "--n-cells",
            "2",
            "--size",
            "8",
        ],
    )
    zarr_barrage.main()
    result = pd.read_csv(out / "index.csv")
    assert result.cell.tolist() == ["roi|2", "roi|1"]
    assert result.review_round.tolist() == [0, 1]
    assert result.bin_x.tolist() == [3, 4]
    for target, array in channels.items():
        corrected = transform(target, array[0], (0, 0, 0))
        expected = [corrected[labels[0] == label].astype(float).mean() for label in [2, 1]]
        np.testing.assert_allclose(result[f"{target}_mean"], expected)
    assert all((out / p).is_file() for p in result.image_path)


@pytest.mark.parametrize("targets", [["edu", "brdu"], ["edu", "brdu", "pcna"], ["pcna"]])
def test_coefficient_columns_follow_the_bundle_targets(tmp_path, targets):
    transform, bundle = write_field(tmp_path, targets)
    assert transform.targets == tuple(targets)
    raw = np.arange(2 * 13 * 19, dtype=np.uint16).reshape(2, 13, 19)
    yy, xx = np.mgrid[0:13, 0:19]
    gain = correction_gain(np.column_stack([xx.ravel(), yy.ravel()]), bundle)
    assert gain.shape == (13 * 19, len(targets))
    gain = gain.reshape(13, 19, len(targets)).astype(np.float32)
    for column, target in enumerate(targets):
        np.testing.assert_allclose(transform(target, raw, (0, 0, 0)), raw.astype(np.float32) * gain[..., column], rtol=1e-6)


def test_name_outside_the_targets_is_rejected(tmp_path):
    transform, _ = write_field(tmp_path, ["edu", "brdu", "pcna"])
    with pytest.raises(ValueError, match="no target 'dapi'"):
        transform("dapi", np.ones((2, 2), dtype=np.uint16), (0, 0, 0))
    default, _ = write_field(tmp_path, ["brdu", "edu"], name="default.joblib")
    with pytest.raises(ValueError, match="no target 'pcna'"):
        default("pcna", np.ones((2, 2), dtype=np.uint16), (0, 0, 0))


@pytest.mark.parametrize(
    ("targets", "width"),
    [(["brdu", "edu"], 3), (["brdu", "edu", "pcna"], 2)],
)
def test_coefficient_width_must_match_the_targets(tmp_path, targets, width):
    transform, _ = write_field(tmp_path, targets, width=width)
    with pytest.raises(ValueError, match="coefficient"):
        transform.targets


@pytest.mark.parametrize("targets", [[], ["brdu", "brdu"], ("brdu", "edu"), "brdu", [1, 2]])
def test_field_targets_must_be_a_nonempty_list_of_unique_keys(tmp_path, targets):
    transform, _ = write_field(tmp_path, targets, width=2)
    with pytest.raises(ValueError, match="targets"):
        transform.targets


def test_require_periodic_targets_binds_the_field_to_the_contract(tmp_path):
    default, _ = write_field(tmp_path, ["brdu", "edu"], name="default.joblib")
    require_periodic_targets(default, DEFAULT_CONTRACT, artifact="barrage")
    custom = MarkerContract("dapi", ("edu", "brdu", "pcna"))
    three, _ = write_field(tmp_path, ["edu", "brdu", "pcna"], name="three.joblib")
    require_periodic_targets(three, custom, artifact="classifier")
    reordered, _ = write_field(tmp_path, ["edu", "brdu"], name="reordered.joblib")
    require_periodic_targets(reordered, MarkerContract("pi", ("edu", "brdu")), artifact="classifier")


@pytest.mark.parametrize(
    ("targets", "contract"),
    [
        (["edu", "brdu"], DEFAULT_CONTRACT),
        (["brdu", "edu"], MarkerContract("pi", ("edu", "brdu"))),
        (["brdu", "edu"], MarkerContract("dapi", ("edu", "brdu", "pcna"))),
        (["brdu", "edu", "pcna"], DEFAULT_CONTRACT),
    ],
)
def test_require_periodic_targets_rejects_reordered_or_different_targets(tmp_path, targets, contract):
    transform, _ = write_field(tmp_path, targets)
    with pytest.raises(ContractMismatchError, match="classifier bundle.*marker order is part of the contract"):
        require_periodic_targets(transform, contract, artifact="classifier bundle")


def test_periodic_correction_leaves_channels_outside_the_targets_untouched(tmp_path):
    transform, _ = write_field(tmp_path, ["edu", "brdu", "pcna"])
    channels = {name: np.full((2, 6), value, dtype=np.uint16) for name, value in [("dapi", 9), ("pcna", 50), ("brdu", 200), ("edu", 100), ("other", 7)]}
    corrected = correct_channels(channels, (0, 3, 4), pixel_correction=transform)
    for name in ("pcna", "brdu", "edu"):
        np.testing.assert_array_equal(corrected[name], transform(name, channels[name], (0, 3, 4)))
    for name in ("dapi", "other"):
        np.testing.assert_array_equal(corrected[name], channels[name])
    assert set(corrected) == set(channels)
