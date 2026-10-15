import joblib
import numpy as np
import pandas as pd
import pytest
import zarr
from sklearn.preprocessing import SplineTransformer

from dino.artifacts import file_sha256
from dino.periodic_pixels import PeriodicPixels
from dino.spatial_pixels import SpatialPixels, bind_spatial_subtraction
from dino.spatial_subtraction import SpatialSubtraction
from dino.zarr_model import iter_model_image_batches, model_crop_reader, sort_cells_for_io
from dino.zarr_source import CellCrop, OpenSource, cell_dino_images


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
    original = dict(source_sha256='source', pixel_correction=periodic.identity, crop_size=8, input_channels=4, highpass_sigma_px=1., mask_erosion_px=0)
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
    bundle = bind_spatial_subtraction({'source_sha256': 'source'}, model_path=path, source_manifest=manifest())
    with pytest.raises(ValueError, match='already has'):
        bind_spatial_subtraction(bundle, model_path=path, source_manifest=manifest())
    bad = manifest()
    bad['channels']['pi']['ome'] = {'axes': []}
    with pytest.raises(ValueError, match='grids must match'):
        bind_spatial_subtraction({'source_sha256': 'source'}, model_path=path, source_manifest=bad)
    SpatialPixels.from_spec(bundle['spatial_subtraction'])
    path.write_bytes(b'mutated model')
    with pytest.raises(ValueError, match='hash mismatch'):
        SpatialPixels.from_spec(bundle['spatial_subtraction'])
