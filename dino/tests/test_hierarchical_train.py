import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from dino.hierarchical_train import (
    dataset_cutoffs,
    fit_final,
    fit_model,
    grouped_folds,
    nested_evaluate,
)


def data(rows=2400, seed=8):
    rng = np.random.default_rng(seed)
    d = np.arange(rows) % 2
    x = rng.normal(size=(rows, 5))
    x[:, 2:] += d[:, None] * 0.7
    logit = 0.9 * x[:, 0] + np.where(d == 0, 1.5, -1.5) * x[:, 2] + d * 0.6
    y = (rng.random(rows) < 1 / (1 + np.exp(-logit))).astype(int)
    return x, y, d


def grouped_data():
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(36), 8)
    d = np.tile([0, 0, 0, 0, 1, 1, 1, 1], 36)
    y = np.tile([0, 0, 1, 1, 0, 0, 1, 1], 36)
    x = rng.normal(size=(len(y), 5))
    x[:, 0] += y * 1.5
    return x, y, d, groups


@pytest.mark.parametrize('family', ['hierarchical', 'joint'])
def test_collapse_preserves_probabilities_and_float32_calls(family):
    x, y, d = data(rows=500)
    fitted = fit_model(x, y, d, datasets=2, embedding_width=2, family=family, lambda0=10, lambda_delta=4)
    p = fitted.predict_proba(x, d)
    cutoffs = dataset_cutoffs(y, p, d, datasets=2)
    for index in range(2):
        scaler, model = fitted.collapse(index)
        reduced = model.predict_proba(scaler.transform(x[d == index]))[:, 1]
        np.testing.assert_allclose(reduced, p[d == index], atol=1e-10, rtol=0)
        np.testing.assert_array_equal(reduced.astype(np.float32) >= cutoffs[index],
                                      p[d == index].astype(np.float32) >= cutoffs[index])
        assert model.n_features_in_ == 5


def test_infinite_local_penalty_equals_pooled_classifier():
    x, y, d = data(rows=500)
    fitted = fit_model(x, y, d, datasets=2, embedding_width=2, lambda0=10, lambda_delta=np.inf)
    scaler = StandardScaler().fit(x)
    pooled = LogisticRegression(C=0.1, max_iter=5000, tol=1e-9).fit(scaler.transform(x), y)
    np.testing.assert_allclose(fitted.predict_proba(x, d), pooled.predict_proba(scaler.transform(x))[:, 1], atol=1e-10)
    for index in range(2):
        _, collapsed = fitted.collapse(index)
        np.testing.assert_array_equal(collapsed.coef_, fitted.model.coef_)
    assert fitted.model.coef_.shape == (1, 5)


def test_local_slopes_recover_opposite_effects_and_shrink_with_pooling():
    x, y, d = data()
    weak = fit_model(x, y, d, datasets=2, embedding_width=2, lambda0=1, lambda_delta=1)
    strong = fit_model(x, y, d, datasets=2, embedding_width=2, lambda0=1, lambda_delta=1000)
    slopes = []
    for fitted in (weak, strong):
        collapsed = [fitted.collapse(index) for index in range(2)]
        np.testing.assert_array_equal(collapsed[0][1].coef_[0, :2], collapsed[1][1].coef_[0, :2])
        np.testing.assert_array_equal(collapsed[0][0].scale_, collapsed[1][0].scale_)
        slopes.append([model.coef_[0, 2] / scaler.scale_[2] for scaler, model in collapsed])
    assert slopes[0][0] > 1 and slopes[0][1] < -1
    assert abs(slopes[1][0] - slopes[1][1]) < abs(slopes[0][0] - slopes[0][1]) / 2


def test_fit_satisfies_requested_two_penalty_objective():
    x, y, d = data(rows=600)
    fitted = fit_model(x, y, d, datasets=2, embedding_width=2, lambda0=3, lambda_delta=19)
    shared = fitted.model.coef_[0, :5]
    delta = fitted.local_scale * fitted.model.coef_[0, 5:].reshape(2, 4)
    scaled = fitted.scaler.transform(x)
    residual = fitted.predict_proba(x, d) - y
    shared_gradient = scaled.T @ residual + 3 * shared
    local_gradient = np.array([np.r_[scaled[d == index, 2:].T @ residual[d == index], residual[d == index].sum()]
                               for index in range(2)]) + 19 * delta
    assert np.max(np.abs(shared_gradient)) < 1e-4
    assert np.max(np.abs(local_gradient)) < 1e-4
    assert abs(residual.sum()) < 1e-4


def test_nested_predictions_cutoffs_and_scaling_ignore_outer_test_labels_and_features():
    x, y, d, groups = grouped_data()
    folds = grouped_folds(y, d, groups, datasets=2, n_splits=4, seed=3)
    kwargs = dict(datasets=2, embedding_width=2, outer_folds=folds, inner_splits=3, seed=3,
                  lambda0_grid=(1., 10.), lambda_delta_grid=(10., np.inf))
    first = nested_evaluate(x, y, d, groups, **kwargs)
    changed_y = y.copy()
    changed_y[folds[0][1]] = 1 - changed_y[folds[0][1]]
    second = nested_evaluate(x, changed_y, d, groups, **kwargs)
    test = folds[0][1]
    np.testing.assert_array_equal(first.probabilities[test], second.probabilities[test])
    np.testing.assert_array_equal(first.cutoffs[test], second.cutoffs[test])
    assert first.selections[0].lambda0 == second.selections[0].lambda0
    assert first.selections[0].lambda_delta == second.selections[0].lambda_delta
    # Extreme held-out features cannot alter inner OOF probabilities or training scaler.
    changed_x = x.copy()
    changed_x[test] += 100
    third = nested_evaluate(changed_x, y, d, groups, **kwargs)
    np.testing.assert_array_equal(first.selections[0].probabilities, third.selections[0].probabilities)
    np.testing.assert_array_equal(first.cutoffs[test], third.cutoffs[test])
    train = folds[0][0]
    selected = first.selections[0]
    model = fit_model(x[train], y[train], d[train], datasets=2, embedding_width=2,
                      lambda0=selected.lambda0, lambda_delta=selected.lambda_delta)
    np.testing.assert_allclose(model.scaler.mean_, x[train].mean(axis=0))
    np.testing.assert_allclose(first.probabilities[test], model.predict_proba(x[test], d[test]))
    for train, test in folds:
        assert not np.intersect1d(groups[train], groups[test]).size
    assert sum(item['cells'] for item in first.metrics.values()) == len(y)


def test_final_cutoffs_are_dataset_specific_and_chosen_after_float32_conversion():
    x, y, d, groups = grouped_data()
    final = fit_final(x, y, d, groups, datasets=2, embedding_width=2,
                      lambda0_grid=(10.,), lambda_delta_grid=(np.inf,))
    from dino.zarr_model import balanced_accuracy_cutoff
    for index in range(2):
        expected = balanced_accuracy_cutoff(y[d == index], final.selection.probabilities[d == index].astype(np.float32))[0]
        assert final.cutoffs[index] == expected
    assert final.fitted.scaler.n_samples_seen_ == len(y)


@pytest.mark.parametrize('change', ['label', 'dataset', 'missing_dataset', 'features', 'context_width'])
def test_invalid_training_input_is_rejected(change):
    x, y, d = data(rows=40)
    if change == 'label':
        y[0] = 2
    elif change == 'dataset':
        d = d.astype(float)
    elif change == 'missing_dataset':
        d[:] = 0
    elif change == 'features':
        x[0, 0] = np.nan
    else:
        x = x[:, :-1]
    with pytest.raises(ValueError):
        fit_model(x, y, d, datasets=2, embedding_width=2)


def test_fold_coverage_and_missing_groups_are_rejected():
    x, y, d, groups = grouped_data()
    missing = groups.astype(float)
    missing[0] = np.nan
    with pytest.raises(ValueError, match='nonmissing'):
        grouped_folds(y, d, missing, datasets=2, n_splits=3)
    with pytest.raises(ValueError, match='both classes'):
        grouped_folds(y, d, y, datasets=2, n_splits=2)
    with pytest.raises(ValueError, match='overlap'):
        nested_evaluate(x, y, d, groups, datasets=2, embedding_width=2,
                        outer_folds=[(np.arange(1, len(y)), np.array([0]))])


def test_nonconvergence_is_rejected():
    x, y, d = data(rows=300)
    with pytest.raises(RuntimeError, match='did not converge'):
        fit_model(x, y, d, datasets=2, embedding_width=2, max_iter=1)


@pytest.mark.parametrize('family', ['hierarchical', 'joint'])
def test_equal_inner_losses_choose_more_pooling_then_stronger_shared_penalty(family):
    from dino.hierarchical_train import select_parameters
    x, y, d, groups = grouped_data()
    x[:] = 0
    selected = select_parameters(x, y, d, groups, datasets=2, embedding_width=2, family=family,
                                 lambda0_grid=(1., 100., 10.), lambda_delta_grid=(1., np.inf, 100.))
    assert selected.lambda0 == 100
    assert selected.lambda_delta == np.inf
    np.testing.assert_array_equal(selected.probabilities, np.full(len(y), 0.5))
