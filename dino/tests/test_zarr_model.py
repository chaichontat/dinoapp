from __future__ import annotations

import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
import zarr
from sklearn.preprocessing import SplineTransformer

from dino import zarr_model
from dino.artifacts import file_sha256
from dino.periodic_pixels import PeriodicPixels
from dino.zarr_model import (
    balanced_accuracy_cutoff,
    extract_normalized_features,
    extract_normalized_feature_tensor,
    iter_model_image_batches,
    load_cell_dino_model,
    model_crop_reader,
    prepare_model_batch,
    read_model_images,
    sort_cells_for_io,
)
from dino.spatial_pixels import SpatialPixels
from dino.zarr_source import CellCrop, CellCropReader, OpenSource
from dino_markers import ContractMismatchError, MarkerContract, UnsupportedArtifactError


def test_model_image_batches_require_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        list(
            iter_model_image_batches(
                pd.DataFrame(),
                bundle={},
                channel_names_by_target={"brdu": ("brdu",)},
                reader=object(),
                batch_size=0,
                prefetch_workers=1,
                crop_workers=1,
                require_mask=False,
            )
        )


def test_balanced_accuracy_cutoff_is_target_specific() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    cutoff, score = balanced_accuracy_cutoff(labels, probabilities)

    assert cutoff == 0.6
    assert score == 1.0


def test_balanced_accuracy_cutoff_tie_prefers_nearest_half_then_higher() -> None:
    labels = np.array([0, 1, 0, 1, 0, 1])
    probabilities = np.array([0, 0.125, 0.25, 0.375, 0.5, 0.625])

    cutoff, score = balanced_accuracy_cutoff(labels, probabilities)

    assert cutoff == 0.625
    assert score == pytest.approx(2 / 3)


class _LinearFeatureModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2, bias=False)
        self.output_dtype: torch.dtype | None = None

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        output = self.linear(batch)
        self.output_dtype = output.dtype
        return output


def test_load_cell_dino_model_casts_weights_to_native_bfloat16(monkeypatch, tmp_path) -> None:
    source_model = _LinearFeatureModel()
    monkeypatch.setattr(zarr_model, "cell_dino_backbone", lambda hub_model: lambda **kwargs: source_model)
    monkeypatch.setitem(
        sys.modules,
        "dinov2.layers.attention",
        SimpleNamespace(XFORMERS_AVAILABLE=False),
    )

    model = load_cell_dino_model(
        hub_model="cell_dino_hpa_vitl16",
        weights=tmp_path / "weights.pth",
        device=torch.device("cpu"),
        precision="bfloat16",
    )

    assert next(model.parameters()).dtype == torch.bfloat16
    assert not model.training
    assert __import__("os").environ["XFORMERS_DISABLED"] == "1"


def test_load_cell_dino_model_rejects_preimported_xformers(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(
        sys.modules,
        "dinov2.layers.attention",
        SimpleNamespace(XFORMERS_AVAILABLE=True),
    )

    with pytest.raises(RuntimeError, match="imported with xFormers enabled"):
        load_cell_dino_model(
            hub_model="cell_dino_hpa_vitl16",
            weights=tmp_path / "weights.pth",
            device=torch.device("cpu"),
            precision="bfloat16",
        )


def test_extract_normalized_features_runs_native_bfloat16_and_returns_float32(
    monkeypatch,
) -> None:
    model = _LinearFeatureModel().to(dtype=torch.bfloat16).eval()
    monkeypatch.setattr(
        torch,
        "autocast",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("autocast used")),
    )

    features = extract_normalized_features(
        model,
        torch.ones((3, 4), dtype=torch.bfloat16),
        precision="bfloat16",
    )

    assert model.output_dtype == torch.bfloat16
    assert features.dtype == np.float32


def test_extract_normalized_feature_tensor_preserves_native_bfloat16() -> None:
    model = _LinearFeatureModel().to(dtype=torch.bfloat16).eval()

    features = extract_normalized_feature_tensor(
        model,
        torch.ones((3, 4), dtype=torch.bfloat16),
        precision="bfloat16",
    )

    assert features.device.type == "cpu"
    assert features.dtype == torch.bfloat16
    torch.testing.assert_close(
        torch.linalg.vector_norm(features.float(), dim=1),
        torch.ones(3),
        atol=0.01,
        rtol=0.01,
    )


def test_prepare_model_batch_transfers_before_transform() -> None:
    transform_inputs: list[tuple[torch.device, torch.dtype]] = []

    def transform(batch: torch.Tensor) -> torch.Tensor:
        transform_inputs.append((batch.device, batch.dtype))
        return batch

    batch = prepare_model_batch(
        [np.zeros((4, 2, 2), dtype=np.uint8)],
        transform=transform,
        device=torch.device("meta"),
        dtype=torch.bfloat16,
    )

    assert transform_inputs == [(torch.device("meta"), torch.float32)]
    assert batch.device == torch.device("meta")
    assert batch.dtype == torch.bfloat16


def test_read_model_images_builds_target_specific_pairs_from_one_crop() -> None:
    crop = CellCrop(
        labels=np.array([[7, 0], [0, 7]], dtype=np.uint32),
        mask=np.array([[True, False], [False, True]]),
        neighbor_mask=np.zeros((2, 2), dtype=bool),
        channels={
            "brdu": np.full((2, 2), 256, dtype=np.uint16),
            "edu": np.full((2, 2), 512, dtype=np.uint16),
            "marker": np.full((2, 2), 768, dtype=np.uint16),
        },
    )

    class Reader:
        calls = 0

        def read_cell(self, **kwargs):
            self.calls += 1
            return crop

    reader = Reader()
    images, has_mask = read_model_images(
        reader,
        pd.Series({"label": 7, "plane_z": 3, "centroid_y": 10, "centroid_x": 11}),
        crop_size=2,
        input_channels=4,
        channel_names_by_target={
            "brdu": ("brdu", "marker"),
            "edu": ("edu", "marker"),
        },
    )

    assert has_mask
    assert reader.calls == 1
    np.testing.assert_array_equal(images["brdu"][:, 0, 0], [1, 3, 0, 0])
    np.testing.assert_array_equal(images["edu"][:, 0, 0], [2, 3, 0, 0])
    np.testing.assert_array_equal(images["brdu"][:, 0, 1], [0, 0, 0, 0])


def test_read_model_images_marks_fully_eroded_cell_invalid() -> None:
    crop = CellCrop(
        labels=np.ones((5, 5), dtype=np.uint32),
        mask=np.pad(np.ones((1, 1), dtype=bool), 2),
        neighbor_mask=np.zeros((5, 5), dtype=bool),
        channels={"brdu": np.full((5, 5), 256, dtype=np.uint16)},
    )

    class Reader:
        def read_cell(self, **kwargs):
            return crop

    images, has_mask = read_model_images(
        Reader(),
        pd.Series({"label": 1, "plane_z": 0, "centroid_y": 2, "centroid_x": 2}),
        crop_size=5,
        input_channels=4,
        channel_names_by_target={"brdu": ("brdu",)},
        mask_erosion_px=2,
    )

    assert not has_mask
    assert not images["brdu"].any()


class _Subtraction:
    target = "brdu"
    reference = "dapi"

    def __call__(self, target, reference, origin_zyx):
        return target - reference


class _Periodic:
    targets = ("brdu", "edu")

    def __call__(self, name, pixels, origin_zyx):
        return pixels + 1


def _read_images(monkeypatch, **corrections):
    references = []
    correct = zarr_model.correct_channels

    def spy(channels, origin, *, reference=None, **kwargs):
        references.append(reference)
        return correct(channels, origin, reference=reference, **kwargs)

    monkeypatch.setattr(zarr_model, "correct_channels", spy)

    class Reader:
        offset_zyx = (0, 0, 0)

        def read_cell(self, **kwargs):
            return CellCrop(
                labels=np.ones((2, 2), dtype=np.uint32),
                mask=np.ones((2, 2), dtype=bool),
                neighbor_mask=np.zeros((2, 2), dtype=bool),
                channels={
                    "brdu": np.full((2, 2), 512, dtype=np.uint16),
                    "edu": np.full((2, 2), 256, dtype=np.uint16),
                    "dapi": np.full((2, 2), 100, dtype=np.uint16),
                },
            )

    read_model_images(
        Reader(),
        pd.Series({"label": 1, "plane_z": 0, "centroid_y": 1, "centroid_x": 1}),
        crop_size=2,
        input_channels=4,
        channel_names_by_target={"brdu": ("brdu",), "edu": ("edu",)},
        **corrections,
    )
    return references


def test_read_model_images_passes_the_subtraction_reference_only_when_one_exists(monkeypatch) -> None:
    assert _read_images(monkeypatch, pixel_correction=_Periodic()) == [None]
    assert _read_images(monkeypatch, spatial_subtraction=_Subtraction()) == ["dapi"]
    assert _read_images(monkeypatch, spatial_subtraction=_Subtraction(), pixel_correction=_Periodic()) == ["dapi"]


def test_model_crop_reader_opens_the_subtraction_reference_channel(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(SpatialPixels, "from_spec", classmethod(lambda cls, spec: _Subtraction()))
    labels = zarr.open_array(tmp_path / "labels", mode="w", shape=(1, 4, 4), chunks=(1, 4, 4), dtype="u4")
    channels = {
        name: zarr.open_array(tmp_path / name, mode="w", shape=(1, 4, 4), chunks=(1, 4, 4), dtype="u2")
        for name in ("brdu", "edu", "dapi", "pi")
    }
    source = OpenSource({"offset_zyx": [0, 0, 0]}, labels, channels)

    reader = model_crop_reader(
        source, cache_gb=0.001, channel_names=("brdu", "edu"), spatial_subtraction={"model": "unused"}
    )

    assert set(reader.channels) == {"brdu", "edu", "dapi"}
    reader.close()


def _write_field(directory, targets):
    spline = SplineTransformer(
        knots=np.tile(np.linspace(0, 1, 5)[:, None], (1, 2)), extrapolation="periodic"
    ).fit([[0.0, 0.0], [1.0, 1.0]])
    path = directory / "field.joblib"
    joblib.dump(
        dict(
            period_xy_um=np.array([11.0, 17.0]),
            periodic_spline=spline,
            periodic_center=np.full(16, 1 / 16),
            coefficient=np.random.default_rng(7).normal(0, 0.5, (16, len(targets))),
            targets=targets,
        ),
        path,
    )
    return PeriodicPixels(str(path), file_sha256(path), (1.0, 1.0, 1.0))


def _batch_inputs(tmp_path, markers):
    labels = zarr.open_array(tmp_path / "labels.zarr", mode="w", shape=(2, 8, 8), chunks=(1, 4, 4), dtype="u4")
    labels[:] = 1
    channels = {}
    for index, marker in enumerate(markers):
        array = zarr.open_array(tmp_path / f"{marker}.zarr", mode="w", shape=labels.shape, chunks=labels.chunks, dtype="u2")
        array[:] = np.arange(128).reshape(labels.shape) * 10 + 1000 * (index + 1)
        channels[marker] = array
    reader = CellCropReader(labels=labels, channels=channels, offset_zyx=(0, 0, 0), cache_bytes=1_000_000)
    rows = pd.DataFrame(dict(cell=["cell-1"], label=[1], plane_z=[0], centroid_y=[4.0], centroid_x=[4.0]))
    return reader, rows, channels


def _bundle(**fields) -> dict:
    return dict(crop_size=8, input_channels=4, highpass_sigma_px=0, mask_erosion_px=0, **fields)


def _batches(reader, rows, bundle, markers):
    return list(
        iter_model_image_batches(
            sort_cells_for_io(rows, reader),
            bundle=bundle,
            channel_names_by_target={marker: (marker,) for marker in markers},
            reader=reader,
            batch_size=1,
            prefetch_workers=1,
            crop_workers=2,
            require_mask=True,
        )
    )


@pytest.mark.parametrize("markers", [("brdu", "edu"), ("edu", "brdu")])
def test_periodic_correction_binds_to_the_bundle_contract_order(tmp_path, markers) -> None:
    transform = _write_field(tmp_path, list(markers))
    reader, rows, channels = _batch_inputs(tmp_path, markers)
    contract = MarkerContract("pi", markers)
    bundle = _bundle(
        schema="dino-zarr-model-v7", marker_contract=contract.to_json(), pixel_correction=transform.identity
    )

    images, _ = _batches(reader, rows, bundle, markers)[0]

    for marker in markers:
        corrected = transform(marker, channels[marker][0], (0, 0, 0))
        np.testing.assert_array_equal(images[marker][0][0], np.clip(corrected / 256.0, 0, 255).astype(np.uint8))
    reader.close()


def test_periodic_correction_rejects_a_same_length_reorder_of_the_bundle_contract(tmp_path) -> None:
    transform = _write_field(tmp_path, ["edu", "brdu"])
    reader, rows, _ = _batch_inputs(tmp_path, ("brdu", "edu"))
    bundle = _bundle(schema="dino-zarr-model-v6", pixel_correction=transform.identity)

    with pytest.raises(ContractMismatchError, match="marker order is part of the contract"):
        _batches(reader, rows, bundle, ("brdu", "edu"))
    reader.close()


def test_periodic_correction_requires_a_versioned_bundle(tmp_path) -> None:
    transform = _write_field(tmp_path, ["brdu", "edu"])
    reader, rows, _ = _batch_inputs(tmp_path, ("brdu", "edu"))

    with pytest.raises(UnsupportedArtifactError, match="model bundle"):
        _batches(reader, rows, _bundle(pixel_correction=transform.identity), ("brdu", "edu"))
    reader.close()


def test_spatial_subtraction_rejects_contracts_it_cannot_serve(tmp_path) -> None:
    contract = MarkerContract("dapi", ("brdu", "edu"))
    reader, rows, _ = _batch_inputs(tmp_path, contract.markers)
    bundle = _bundle(
        schema="dino-zarr-model-v7", marker_contract=contract.to_json(), spatial_subtraction={"model": "unused"}
    )

    with pytest.raises(ValueError, match="spatial subtraction supports only BrdU against the pi channel"):
        _batches(reader, rows, bundle, contract.markers)
    reader.close()
