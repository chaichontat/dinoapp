"""Display crops obey the same native-pixel contract as corrected inference."""
import json
from threading import Lock
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
import zarr
from dino_markers import (
    DEFAULT_CONTRACT,
    ZARR_MODEL,
    ContractMismatchError,
    MarkerContract,
    config_to_json,
    parse_config,
)
from sklearn.preprocessing import SplineTransformer

from dino.artifacts import file_sha256
from dino.periodic_pixels import PeriodicPixels
from dino.spatial_pixels import SpatialPixels
from dino.spatial_subtraction import SpatialSubtraction
from dino.zarr_source import CellCropReader
from dinoapp import display_pixels
from dinoapp.display_pixels import DisplayPixels
from dinoapp.prod3 import FieldSource as Prod3

SCALE, ORIGIN = (.6, 2., 3.), (-13., 9., -5.)


def build_corrections(tmp_path, targets):
    splines = tuple(SplineTransformer(n_knots=k, degree=3, include_bias=True,
                                     extrapolation='constant').fit([[0], [100]]) for k in (3, 4))
    smooth = SpatialSubtraction(splines, np.array([[0, 100], [0, 100]]),
                                np.full(30, .25), np.full(30, .001), .001, .05, {})
    smooth_path = tmp_path / 'smooth.joblib'
    smooth.save(smooth_path)
    spline = SplineTransformer(knots=np.tile(np.linspace(0, 1, 5)[:, None], (1, 2)),
                               extrapolation='periodic').fit([[0., 0.], [1., 1.]])
    periodic_path = tmp_path / f'periodic-{len(targets)}.joblib'
    joblib.dump(dict(period_xy_um=np.array([11., 17.]), periodic_spline=spline,
                     periodic_center=np.full(16, 1 / 16),
                     coefficient=np.random.default_rng(3).normal(0, .5, (16, len(targets))),
                     targets=list(targets)), periodic_path)
    spatial = SpatialPixels(str(smooth_path), file_sha256(smooth_path), SCALE, ORIGIN)
    periodic = PeriodicPixels(str(periodic_path), file_sha256(periodic_path), SCALE)
    return spatial, periodic


@pytest.fixture
def display(tmp_path):
    pixels = DisplayPixels.__new__(DisplayPixels)
    pixels.spatial, pixels.periodic = build_corrections(tmp_path, ('brdu', 'edu'))
    pixels.contract = DEFAULT_CONTRACT
    return pixels


def test_native_crop_uses_true_dapi_and_subtracts_before_periodic(display, tmp_path):
    labels = zarr.open_array(tmp_path / 'labels', mode='w', shape=(1, 16, 16), chunks=(1, 4, 4), dtype='u2')
    labels[:] = 1
    channels = {}
    for marker, value in [('brdu', 3000), ('edu', 5000), ('pi', 50000), ('dapi', 8000)]:
        array = zarr.open_array(tmp_path / marker, mode='w', shape=(1, 16, 16), chunks=(1, 4, 4), dtype='u2')
        array[:] = np.broadcast_to(value + np.arange(16)[None, :, None] * 100, array.shape)
        channels[marker] = array
    source = Prod3.__new__(Prod3)
    source.root = tmp_path
    source.props_path = tmp_path / 'props-cortex.parquet'
    pd.DataFrame(dict(label=[1], plane_z=[0], centroid_y=[8.4], centroid_x=[9.4])).to_parquet(source.props_path)
    source.crop_contract = {'size': 8}
    (tmp_path / 'barrage-pixel').mkdir()
    (tmp_path / 'barrage-pixel/run.json').write_text(json.dumps(source.crop_contract))
    source.crop_lock = Lock()
    source.display_pixels = display
    source.shape = labels.shape
    source.crop_reader = CellCropReader(labels=labels, channels=channels, offset_zyx=(0, 0, 0), cache_bytes=1024**2)
    raw = source.crop_reader.read_cell(label=1, plane_z=0, center_y=8, center_x=9, size=8)
    actual = source._read_cell_crop(1)[0]
    origin = (0, 4, 5)
    expected_brdu = display.periodic('brdu', display.spatial(raw.channels['brdu'], raw.channels['dapi'], origin), origin)
    np.testing.assert_array_equal(actual.channels['brdu'], expected_brdu)
    np.testing.assert_array_equal(actual.channels['edu'], display.periodic('edu', raw.channels['edu'], origin))
    np.testing.assert_array_equal(actual.channels['pi'], raw.channels['pi'])
    wrong_reference = display.periodic('brdu', display.spatial(raw.channels['brdu'], raw.channels['pi'], origin), origin)
    assert not np.array_equal(actual.channels['brdu'], wrong_reference)
    assert actual.channels['brdu'].dtype == np.float32
    source.crop_reader.close()


def test_field_and_cell_crop_are_equal_at_same_native_coordinates(display):
    rng = np.random.default_rng(7)
    raw = {marker: rng.integers(low, low + 1000, (1, 16, 16), dtype=np.uint16)
           for marker, low in [('brdu', 3000), ('edu', 4000), ('pi', 8000)]}
    display.source = SimpleNamespace(channels=raw)
    selection = (0, slice(4, 12), slice(5, 13))
    anatomical = np.full((8, 8), 50000, dtype=np.uint16)
    field = display.field_images({'pi': anatomical}, selection)
    crop_channels = {marker: pixels[selection] for marker, pixels in raw.items()}
    corrected = display.channels(crop_channels, (0, 4, 5), reference='pi')
    np.testing.assert_array_equal(field['brdu'], corrected['brdu'])
    np.testing.assert_array_equal(field['edu'], corrected['edu'])
    np.testing.assert_array_equal(field['pi'], anatomical)


def test_native_grid_rejects_shifted_physical_origin(display, tmp_path):
    display.source = SimpleNamespace(manifest={'labels': {'root': str(tmp_path / 'labels')}},
                                     labels=np.ones((1, 16, 16)), offset_zyx=(0, 0, 0))
    display.validate_native_grid(labels=tmp_path / 'labels', shape=(1, 16, 16),
                                 scale=(.6, 2., 3.), origin=(-13., 9., -5.))
    with pytest.raises(ValueError, match='physical grid'):
        display.validate_native_grid(labels=tmp_path / 'labels', shape=(1, 16, 16),
                                     scale=(.6, 2., 3.), origin=(-13., 0., 0.))


def test_three_marker_field_images_correct_every_marker_in_contract_order(tmp_path):
    pixels = DisplayPixels.__new__(DisplayPixels)
    pixels.spatial, pixels.periodic = build_corrections(tmp_path, ('edu', 'brdu', 'pcna'))
    pixels.contract = MarkerContract('pi', ('edu', 'brdu', 'pcna'))
    rng = np.random.default_rng(11)
    raw = {key: rng.integers(low, low + 1000, (1, 16, 16), dtype=np.uint16)
           for key, low in [('pi', 8000), ('edu', 4000), ('brdu', 3000), ('pcna', 5000)]}
    pixels.source = SimpleNamespace(channels=raw)
    selection = (0, slice(4, 12), slice(5, 13))
    anatomical = np.full((8, 8), 50000, dtype=np.uint16)

    field = pixels.field_images({'pi': anatomical}, selection)

    corrected = pixels.channels({key: array[selection] for key, array in raw.items()}, (0, 4, 5), reference='pi')
    assert list(field) == ['pi', 'edu', 'brdu', 'pcna']
    np.testing.assert_array_equal(field['pi'], anatomical)
    for marker in ('edu', 'brdu', 'pcna'):
        np.testing.assert_array_equal(field[marker], corrected[marker])
    assert not np.array_equal(field['pcna'], raw['pcna'][selection])


@pytest.fixture
def display_inputs(tmp_path, monkeypatch):
    spatial, periodic = build_corrections(tmp_path, ('brdu', 'edu'))
    geometry = {'coordinate_transformations': [{'type': 'scale', 'scale': list(SCALE)},
                                               {'type': 'translation', 'translation': list(ORIGIN)}]}
    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    manifest = {'source_sha256': 'source-hash',
                'channels': {'brdu': {'ome': geometry}, 'pi': {'ome': geometry}}}
    monkeypatch.setattr(display_pixels, 'open_review_source', lambda directory: SimpleNamespace(manifest=manifest))

    def write_bundle(contract: MarkerContract = DEFAULT_CONTRACT, periodic_identity=periodic.identity):
        bundle = ZARR_MODEL.stamp({'source_sha256': 'source-hash', 'spatial_subtraction': spatial.identity,
                                   'pixel_correction': periodic_identity}, contract)
        path = tmp_path / 'model.joblib'
        joblib.dump(bundle, path)
        return path

    return source_dir, write_bundle


def test_display_pixels_bind_a_bundle_of_the_source_contract(display_inputs):
    source_dir, write_bundle = display_inputs

    pixels = DisplayPixels(write_bundle(), source_dir)

    assert pixels.contract == DEFAULT_CONTRACT
    assert pixels.periodic.targets == ('brdu', 'edu')


def test_display_pixels_accept_a_legacy_bundle_of_a_legacy_source(display_inputs):
    source_dir, write_bundle = display_inputs
    path = write_bundle()
    bundle = joblib.load(path)
    legacy = {key: value for key, value in bundle.items() if key != 'marker_contract'}
    legacy['schema'] = 'dino-zarr-model-v6'
    joblib.dump(legacy, path)

    assert DisplayPixels(path, source_dir).contract == DEFAULT_CONTRACT


def test_display_pixels_reject_a_bundle_whose_reference_is_not_pi(display_inputs):
    source_dir, write_bundle = display_inputs
    contract = MarkerContract('dapi', ('brdu', 'edu'))
    config = parse_config({
        'version': 1, 'reference': {'key': 'dapi', 'label': 'DAPI'},
        'markers': [{'key': 'brdu', 'label': 'BrdU', 'color': '#ff00ff'},
                    {'key': 'edu', 'label': 'EdU', 'color': '#00ff00'}],
        'presentation': {'crop_panels': ['dapi', 'edu', 'brdu'], 'negative_color': [.25, .25, .25],
                         'uncalled_color': '#999999'},
    }, source='test')
    (source_dir / 'markers.json').write_text(json.dumps(config_to_json(config)))

    with pytest.raises(ValueError, match='supports only BrdU against the pi channel'):
        DisplayPixels(write_bundle(contract), source_dir)


def test_display_pixels_reject_a_source_with_reordered_markers(display_inputs):
    source_dir, write_bundle = display_inputs
    config = parse_config({
        'version': 1, 'reference': {'key': 'pi', 'label': 'PI'},
        'markers': [{'key': 'edu', 'label': 'EdU', 'color': '#00ff00'},
                    {'key': 'brdu', 'label': 'BrdU', 'color': '#ff00ff'}],
        'presentation': {'crop_panels': ['pi', 'edu', 'brdu'], 'negative_color': [.25, .25, .25],
                         'uncalled_color': '#999999'},
    }, source='test')
    (source_dir / 'markers.json').write_text(json.dumps(config_to_json(config)))

    with pytest.raises(ContractMismatchError, match='marker order is part of the contract'):
        DisplayPixels(write_bundle(), source_dir)


def test_display_pixels_reject_periodic_targets_that_reorder_the_markers(tmp_path, display_inputs):
    source_dir, write_bundle = display_inputs
    (tmp_path / 'reordered').mkdir()
    _, reordered = build_corrections(tmp_path / 'reordered', ('edu', 'brdu'))

    with pytest.raises(ContractMismatchError, match='marker order is part of the contract'):
        DisplayPixels(write_bundle(periodic_identity=reordered.identity), source_dir)
