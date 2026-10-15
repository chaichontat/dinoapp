import numpy as np
import pandas as pd
import pytest

from dino.spatial_subtraction import ACCURACY_CONTRACT, SpatialSubtraction
from sklearn.preprocessing import SplineTransformer
from dino.spatial_subtraction_validation import validate, validate_fit


def accepted_fit():
    train = pd.DataFrame(dict(x_um=[1.0, 7.0, 11.0], y_um=[12.0, 8.0, 3.0]))
    train["dapi"] = [1000., 2000., 3000.]
    train["brdu"] = train.dapi * 0.2 + 2000000.
    splines = tuple(SplineTransformer(n_knots=k, extrapolation="constant").fit(train.iloc[:, i:i+1].to_numpy()) for i, k in enumerate((3, 4)))
    model = SpatialSubtraction(
        splines=splines,
        domain_xy_um=np.array([[1.0, 11.0], [3.0, 12.0]]),
        penalty=0.01,
        quantile=0.05,
        alpha_coefficients=np.full(30, 0.2),
        beta_scaled_coefficients=np.zeros(30),
        diagnostics=dict(
            samples=3,
            status="solved",
            accuracy_contract=ACCURACY_CONTRACT,
            raw_primal_feasibility=0.3,
            relative_projected_objective_gap=1e-6,
            projected_objective_sum_scaled=300.0,
            dual_objective_sum_scaled=299.9997,
        ),
    )
    return model, train


def test_fit_acceptance_checks_exact_training_domain_and_raw_gap_units():
    model, train = accepted_fit()
    report = validate_fit(model, train, label="candidate", penalty=0.01)
    assert report["label"] == "candidate"
    assert report["raw_primal"] == 0.3
    assert report["raw_mean_gap"] == pytest.approx(0.1)
    altered = train.copy()
    altered.loc[0, "x_um"] = -5.0
    with pytest.raises(AssertionError):
        validate_fit(model, altered, label="wrong training domain", penalty=0.01)


@pytest.mark.parametrize(
    "change", ["inaccurate", "nan_offset", "negative_alpha", "nan_gap", "primal", "sample_count"]
)
def test_fit_rejects_invalid_artifact_or_numerical_acceptance(change):
    model, train = accepted_fit()
    if change == "inaccurate":
        model.diagnostics["status"] = "solved inaccurate"
    elif change == "nan_offset":
        model.beta_scaled_coefficients[0] = np.nan
    elif change == "negative_alpha":
        model.alpha_coefficients[0] = -0.001
    elif change == "nan_gap":
        model.diagnostics["relative_projected_objective_gap"] = np.nan
    elif change == "primal":
        model.diagnostics["raw_primal_feasibility"] = 1.01
    else:
        model.diagnostics["samples"] = 4
    with pytest.raises(AssertionError):
        validate_fit(model, train, label="invalid", penalty=0.01)


def test_validation_requires_explicit_existing_destination(tmp_path):
    with pytest.raises(ValueError, match="destination"):
        validate(tmp_path, "prod2", destination_dir=tmp_path / "missing")
    assert not list(tmp_path.iterdir())


def test_current_offset_coefficients_must_match_saved_objective():
    from dino.spatial_subtraction import SpatialSubtraction
    from sklearn.preprocessing import SplineTransformer

    rng = np.random.default_rng(1)
    xy = rng.uniform(size=(100, 2))
    reference = rng.uniform(1000, 2000, len(xy))
    train = pd.DataFrame(dict(x_um=xy[:, 0], y_um=xy[:, 1], dapi=reference, brdu=reference * 0.2))
    splines = tuple(SplineTransformer(n_knots=k, extrapolation='constant').fit(xy[:, i:i+1]) for i, k in enumerate((3, 4)))
    width = splines[0].n_features_out_ * splines[1].n_features_out_
    model = SpatialSubtraction(splines, np.column_stack([xy.min(axis=0), xy.max(axis=0)]), np.full(width, 0.2), np.zeros(width), 0.01, 0.05, dict(samples=len(train), status='solved', accuracy_contract=ACCURACY_CONTRACT, raw_primal_feasibility=0., relative_projected_objective_gap=0., projected_objective_sum_scaled=0., dual_objective_sum_scaled=0.))
    validate_fit(model, train, label='optimum', penalty=0.01)
    model.beta_scaled_coefficients[:] += 1
    with pytest.raises(AssertionError):
        validate_fit(model, train, label='altered', penalty=0.01)
