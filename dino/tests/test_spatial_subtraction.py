import numpy as np
import pandas as pd
import pytest

from dino.spatial_subtraction import SpatialSubtraction, fit_spatial_subtraction
from dino.spatial_subtraction_validation import validate_fit


def test_recovers_smooth_fields_and_roundtrips(tmp_path):
    rng = np.random.default_rng(812)
    xy = rng.uniform([0, 0], [1200, 2200], size=(1000, 2))
    reference = rng.uniform(1000, 18000, len(xy))
    alpha = 0.12 + 0.15 * xy[:, 0] / 1200
    beta = 200 + 700 * xy[:, 1] / 2200
    target = alpha * reference + beta
    model = fit_spatial_subtraction(xy, reference, target, penalty=0.001, tolerance=1e-5)
    predicted_alpha, predicted_beta = model.fields(xy)
    assert np.mean(np.abs(predicted_alpha-alpha)) < 0.01
    assert np.mean(np.abs(predicted_beta-beta)) < 60
    assert np.mean(np.abs(model.predict(xy, reference)-target)) < 35
    probe = np.column_stack([np.linspace(500, 501, 100), np.full(100, 1200)])
    a, b = model.fields(probe)
    assert np.abs(np.diff(a)).max() < 0.0001
    assert np.abs(np.diff(b)).max() < 1
    assert a.min() >= 0
    edge=np.array([[model.domain_xy_um[0,0],1200]])
    beyond=edge-np.array([[100,0]])
    np.testing.assert_array_equal(model.fields(edge)[0],model.fields(beyond)[0])
    np.testing.assert_array_equal(model.fields(edge)[1],model.fields(beyond)[1])
    path = tmp_path/'model.joblib'
    model.save(path)
    restored = SpatialSubtraction.load(path)
    np.testing.assert_array_equal(model.predict(xy,reference),restored.predict(xy,reference))


def test_constant_relation_is_recovered_with_strong_smoothing():
    rng = np.random.default_rng(14)
    xy = rng.uniform(size=(700, 2))
    reference = rng.uniform(1000, 15000, len(xy))
    target = .2 * reference + 400
    model = fit_spatial_subtraction(xy,reference,target,penalty=.1)
    np.testing.assert_allclose(model.predict(xy,reference),target,atol=2)
    train = pd.DataFrame(dict(x_um=xy[:, 0], y_um=xy[:, 1], dapi=reference, brdu=target))
    validate_fit(model, train, label="constant relation", penalty=.1)
    assert model.signed_residual(xy[:1],np.array([0]),np.array([300]))[0] == 0


def test_noisy_fit_targets_lower_quantile_not_the_median():
    rng = np.random.default_rng(419)
    xy = rng.uniform(size=(2000,2))
    reference = rng.uniform(1000,18000,2000)
    target = .18*reference+350+rng.exponential(500,2000)
    model = fit_spatial_subtraction(xy,reference,target,penalty=.01,tolerance=1e-6,relative_tolerance=1e-7)
    residual = target-model.predict(xy,reference)
    assert abs(np.mean(residual<0)-.05) < .015
    assert np.median(residual)>200


def test_rejects_invalid_inputs_and_nonconvergence():
    rng = np.random.default_rng(32)
    xy = rng.uniform(size=(300,2))
    reference = rng.uniform(1000,15000,300)
    target = .2*reference+rng.exponential(500,300)
    with pytest.raises(ValueError,match='positive'):
        fit_spatial_subtraction(xy,reference,-target,penalty=.01)
    with pytest.raises(ValueError,match='variation'):
        fit_spatial_subtraction(xy,np.ones(300),target,penalty=.01)
    with pytest.raises(RuntimeError,match='did not converge'):
        fit_spatial_subtraction(xy,reference,target,penalty=.01,max_iter=1)
