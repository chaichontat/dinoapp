import re

import joblib
import numpy as np
import pandas as pd
import pytest
import zarr
from sklearn.preprocessing import SplineTransformer

from dino.artifacts import file_sha256
from dino.periodic_pixels import PeriodicPixels
from dino.pixel_corrections import correct_channels
from dino.spatial_pixels import SpatialPixels, bind_spatial_subtraction, require_spatial_subtraction_contract
from dino.spatial_subtraction import SpatialSubtraction
from dino.zarr_model import iter_model_image_batches, model_crop_reader, sort_cells_for_io
from dino.zarr_source import CellCrop, OpenSource, cell_dino_images
from dino_markers import DEFAULT_CONTRACT, MarkerContract, MissingContractError, UnsupportedArtifactError


@pytest.fixture
def model(tmp_path):
    splines = tuple(SplineTransformer(n_knots=k, degree=3, include_bias=True, extrapolation="constant").fit([[0], [100]]) for k in (3, 4))
    fitted = SpatialSubtraction(splines, np.array([[0, 100], [0, 100]]), np.linspace(.1, .6, 30), np.linspace(-.1, .8, 30), .001, .05, {})
    path = tmp_path / 'smooth.joblib'
    fitted.save(path)
    return fitted, path


def manifest():
    geometry = {'axes': [dict(name=n, type='space', unit='micrometer') for n in 'zyx'], 'coordinate_transformations': [dict(type='scale', scale=[.6, 2., 3.]), dict(type='translation', translation=[-13., 9., -5.])]}
    return {'source_sha256': 'source', 'channels': {'brdu': {'ome': geometry}, 'pi': {'ome': geometry}}}


def test_raw_geometry_support_rounding_and_clipping(model):
    fitted, path = model
    transform = SpatialPixels(str(path), file_sha256(path), (.6, 2., 3.), (-13., 9., -5.))
    origin = (44, 7, 9)
    brdu = np.array([[0, 10, 400], [650, 70000, 900.]])
    dapi = np.array([[1000, 8000, 0], [1500, 2000, 1200.]])
    xy = np.array([[22, 23], [25, 23], [28, 23], [22, 25], [25, 25], [28, 25]])
    expected = np.clip(np.rint(fitted.signed_residual(xy, dapi.ravel(), brdu.ravel())), 0, 65535).astype(np.uint16).reshape(2, 3)
    np.testing.assert_array_equal(transform(brdu, dapi, origin), expected)
    np.testing.assert_array_equal(transform(brdu, dapi, (900, 7, 9)), expected)
    assert expected[0, 0] == expected[0, 1] == expected[0, 2] == 0
    assert expected[1, 1] == 65535


def test_batch_uses_raw_reference_before_periodic_and_preprocessing(model, tmp_path):
    _fitted, path = model
    source_manifest = manifest()
    source_manifest['offset_zyx'] = [0, 3, 4]
    labels = zarr.open_array(tmp_path / 'labels', mode='w', shape=(1, 8, 8), chunks=(1, 4, 4), dtype='u2')
    labels[:] = 1
    arrays = {}
    for name, value in [('brdu', 3000), ('edu', 5000), ('pi', 8000)]:
        array = zarr.open_array(tmp_path / name, mode='w', shape=(1, 16, 16), chunks=(1, 4, 4), dtype='u2')
        array[:] = np.broadcast_to(value + np.arange(16)[None, :, None] * 400, array.shape)
        arrays[name] = array
    spline = SplineTransformer(knots=np.tile(np.linspace(0, 1, 5)[:, None], (1, 2)), extrapolation='periodic').fit([[0., 0.], [1., 1.]])
    periodic_path = tmp_path / 'periodic.joblib'
    joblib.dump(dict(period_xy_um=np.array([11., 17.]), periodic_spline=spline, periodic_center=np.full(16, 1/16), coefficient=np.random.default_rng(3).normal(0, .5, (16, 2)), targets=['brdu', 'edu']), periodic_path)
    periodic = PeriodicPixels(str(periodic_path), file_sha256(periodic_path), (.6, 2., 3.))
    original = dict(schema='dino-zarr-model-v6', source_sha256='source', pixel_correction=periodic.identity, crop_size=8, input_channels=4, highpass_sigma_px=1., mask_erosion_px=0)
    bundle = bind_spatial_subtraction(original, model_path=path, source_manifest=source_manifest)
    assert 'spatial_subtraction' not in original
    source = OpenSource(source_manifest, labels, arrays)
    reader = model_crop_reader(source, cache_gb=.001, channel_names=('brdu', 'edu'), spatial_subtraction=bundle['spatial_subtraction'])
    assert set(reader.channels) == {'brdu', 'edu', 'pi'}
    row = pd.DataFrame(dict(cell=['one'], label=[1], plane_z=[0], centroid_y=[4.], centroid_x=[4.]))
    batches = list(iter_model_image_batches(sort_cells_for_io(row, reader), bundle=bundle, channel_names_by_target={'brdu': ('brdu',), 'edu': ('edu',)}, reader=reader, batch_size=1, prefetch_workers=1, crop_workers=2, require_mask=True))
    raw = reader.read_cell(label=1, plane_z=0, center_y=4, center_x=4, size=8)
    origin = (0, 3, 4)
    smooth = SpatialPixels.from_spec(bundle['spatial_subtraction'])
    corrected = {**raw.channels, 'brdu': periodic('brdu', smooth(raw.channels['brdu'], raw.channels['pi'], origin), origin), 'edu': periodic('edu', raw.channels['edu'], origin)}
    expected = cell_dino_images(CellCrop(raw.labels, raw.mask, raw.neighbor_mask, corrected), input_channels=4, channel_names_by_target={'brdu': ('brdu',), 'edu': ('edu',)}, highpass_sigma_px=1.)
    for target in ('brdu', 'edu'):
        np.testing.assert_array_equal(batches[0][0][target][0], expected[target])
    assert (batches[0][0]['edu'][0][1:] == 0).all()
    reader.close()


def test_binding_rejects_wrong_grid_and_mutated_artifact(model):
    _fitted, path = model
    bundle = bind_spatial_subtraction({'schema': 'dino-zarr-model-v6', 'source_sha256': 'source'}, model_path=path, source_manifest=manifest())
    with pytest.raises(ValueError, match='already has'):
        bind_spatial_subtraction(bundle, model_path=path, source_manifest=manifest())
    bad = manifest()
    bad['channels']['pi']['ome'] = {'axes': []}
    with pytest.raises(ValueError, match='grids must match'):
        bind_spatial_subtraction({'schema': 'dino-zarr-model-v6', 'source_sha256': 'source'}, model_path=path, source_manifest=bad)
    SpatialPixels.from_spec(bundle['spatial_subtraction'])
    path.write_bytes(b'mutated model')
    with pytest.raises(ValueError, match='hash mismatch'):
        SpatialPixels.from_spec(bundle['spatial_subtraction'])


def v7(contract, **fields):
    return dict(schema='dino-zarr-model-v7', marker_contract=contract.to_json(), source_sha256='source', **fields)


def test_spatial_pixels_expose_the_fixed_target_and_reference(model):
    _fitted, path = model
    transform = SpatialPixels(str(path), file_sha256(path), (.6, 2., 3.), (-13., 9., -5.))
    assert (transform.target, transform.reference) == ('brdu', 'pi')
    assert (transform.identity['target'], transform.identity['reference']) == ('brdu', 'pi')
    with pytest.raises(AttributeError):
        transform.target = 'edu'


@pytest.mark.parametrize('field, value', [('target', 'edu'), ('reference', 'dapi')])
def test_from_spec_rejects_another_target_or_reference(model, field, value):
    _fitted, path = model
    spec = SpatialPixels(str(path), file_sha256(path), (.6, 2., 3.), (-13., 9., -5.)).identity
    with pytest.raises(ValueError, match='Unsupported spatial pixel subtraction contract'):
        SpatialPixels.from_spec({**spec, field: value})


def test_contract_check_accepts_only_brdu_against_pi():
    for contract in (DEFAULT_CONTRACT, MarkerContract('pi', ('edu', 'brdu')), MarkerContract('pi', ('pcna', 'edu', 'brdu'))):
        require_spatial_subtraction_contract(contract, artifact='bundle')
    for contract in (MarkerContract('pi', ('edu', 'pcna')), MarkerContract('dapi', ('edu', 'brdu', 'pcna'))):
        with pytest.raises(ValueError, match='^' + re.escape(f'bundle: spatial subtraction supports only BrdU against the pi channel verified as 405 nm DAPI; contract is {contract}')):
            require_spatial_subtraction_contract(contract, artifact='bundle')


@pytest.mark.parametrize('contract', [MarkerContract('pi', ('edu', 'pcna')), MarkerContract('dapi', ('edu', 'brdu', 'pcna'))])
def test_binding_rejects_a_bundle_whose_contract_cannot_be_subtracted(model, contract):
    _fitted, path = model
    bundle = v7(contract)
    before = dict(bundle)
    with pytest.raises(ValueError, match='spatial subtraction supports only BrdU against the pi channel'):
        bind_spatial_subtraction(bundle, model_path=path, source_manifest=manifest())
    assert bundle == before


def test_binding_requires_a_recognized_bundle_with_its_contract(model):
    _fitted, path = model
    with pytest.raises(UnsupportedArtifactError):
        bind_spatial_subtraction({'source_sha256': 'source'}, model_path=path, source_manifest=manifest())
    with pytest.raises(MissingContractError, match='model bundle'):
        bind_spatial_subtraction({'schema': 'dino-zarr-model-v7', 'source_sha256': 'source'}, model_path=path, source_manifest=manifest())


@pytest.mark.parametrize('schema, current', [('dino-zarr-model-v6', 'dino-zarr-model-v7'), ('dino-knn-intensity-v1', 'dino-knn-intensity-v2')])
def test_binding_a_legacy_bundle_returns_a_current_default_bundle(model, schema, current):
    _fitted, path = model
    original = {'schema': schema, 'source_sha256': 'source', 'crop_size': 8}
    before = dict(original)
    bound = bind_spatial_subtraction(original, model_path=path, source_manifest=manifest())
    assert original == before
    assert bound['schema'] == current
    assert bound['marker_contract'] == DEFAULT_CONTRACT.to_json()
    assert {k: v for k, v in bound.items() if k not in ('schema', 'marker_contract', 'spatial_subtraction')} == {'source_sha256': 'source', 'crop_size': 8}
    assert bound['spatial_subtraction'] == SpatialPixels.from_spec(bound['spatial_subtraction']).identity


def test_binding_keeps_the_inputs_contract_including_marker_order(model):
    _fitted, path = model
    contract = MarkerContract('pi', ('pcna', 'edu', 'brdu'))
    original = v7(contract, crop_size=8)
    before = dict(original)
    bound = bind_spatial_subtraction(original, model_path=path, source_manifest=manifest())
    assert original == before
    assert bound['schema'] == 'dino-zarr-model-v7'
    assert bound['marker_contract'] == contract.to_json()
    assert bound['marker_contract']['markers'] == ['pcna', 'edu', 'brdu']


class Subtraction:
    target = 'brdu'
    reference = 'pi'

    def __init__(self, calls):
        self.calls = calls

    def __call__(self, target, reference, origin):
        self.calls.append(('subtract', float(target[0, 0]), float(reference[0, 0]), origin))
        return target - reference


class Correction:
    targets = ('brdu', 'edu')

    def __init__(self, calls, targets):
        self.targets = targets
        self.calls = calls

    def __call__(self, name, pixels, origin):
        self.calls.append(('correct', name, float(pixels[0, 0]), origin))
        return pixels * 10


def crops(**values):
    return {name: np.full((2, 2), value, dtype=np.float32) for name, value in values.items()}


def test_correct_channels_with_three_markers_subtracts_then_corrects_every_target():
    calls = []
    channels = crops(pi=5, pcna=3, edu=7, brdu=11)
    corrected = correct_channels(
        channels, (1, 2, 3), reference='pi', spatial_subtraction=Subtraction(calls), pixel_correction=Correction(calls, ('edu', 'brdu', 'pcna'))
    )
    assert calls == [('subtract', 11.0, 5.0, (1, 2, 3)), ('correct', 'pcna', 3.0, (1, 2, 3)), ('correct', 'edu', 7.0, (1, 2, 3)), ('correct', 'brdu', 6.0, (1, 2, 3))]
    assert {name: float(pixels[0, 0]) for name, pixels in corrected.items()} == {'pi': 5.0, 'pcna': 30.0, 'edu': 70.0, 'brdu': 60.0}
    assert float(channels['brdu'][0, 0]) == 11.0


def test_correct_channels_corrects_only_the_targets_of_the_correction():
    calls = []
    corrected = correct_channels(crops(pi=5, pcna=3, edu=7, brdu=11), (0, 0, 0), pixel_correction=Correction(calls, ('edu',)))
    assert calls == [('correct', 'edu', 7.0, (0, 0, 0))]
    assert {name: float(pixels[0, 0]) for name, pixels in corrected.items()} == {'pi': 5.0, 'pcna': 3.0, 'edu': 70.0, 'brdu': 11.0}


def test_subtraction_without_a_reference_channel_name_is_rejected():
    with pytest.raises(ValueError, match='Spatial subtraction requires the crop channel holding its reference pixels'):
        correct_channels(crops(pi=5, brdu=11), (0, 0, 0), spatial_subtraction=Subtraction([]))


def test_prod3_style_crops_subtract_against_dapi_not_pi():
    calls = []
    channels = crops(pi=500, dapi=5, edu=7, brdu=11)
    corrected = correct_channels(channels, (1, 2, 3), reference='dapi', spatial_subtraction=Subtraction(calls), pixel_correction=Correction(calls, ('brdu', 'edu')))
    assert calls == [('subtract', 11.0, 5.0, (1, 2, 3)), ('correct', 'edu', 7.0, (1, 2, 3)), ('correct', 'brdu', 6.0, (1, 2, 3))]
    assert {name: float(pixels[0, 0]) for name, pixels in corrected.items()} == {'pi': 500.0, 'dapi': 5.0, 'edu': 70.0, 'brdu': 60.0}


def test_subtraction_uses_the_uncorrected_reference_with_the_real_transform(model):
    fitted, path = model
    transform = SpatialPixels(str(path), file_sha256(path), (.6, 2., 3.), (-13., 9., -5.))
    channels = {'dapi': np.array([[1000, 8000, 0], [1500, 2000, 1200.]]), 'brdu': np.array([[0, 10, 400], [650, 70000, 900.]]), 'edu': np.ones((2, 3))}
    corrected = correct_channels(channels, (44, 7, 9), reference='dapi', spatial_subtraction=transform)
    np.testing.assert_array_equal(corrected['brdu'], transform(channels['brdu'], channels['dapi'], (44, 7, 9)))
    np.testing.assert_array_equal(corrected['dapi'], channels['dapi'])
    np.testing.assert_array_equal(corrected['edu'], channels['edu'])
