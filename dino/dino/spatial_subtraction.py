"""Cubic XY varying-coefficient quantile subtraction with quadratic roughness."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import joblib
import numpy as np
from scipy import sparse
from sklearn.preprocessing import SplineTransformer

INTENSITY_SCALE = 1000.0
ACCURACY_CONTRACT = "raw_primal_le1_relative_objective_gap_le1e-5_v1"


def _xy(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("XY must be a finite (N, 2) array in physical micrometers")
    return values


def _paired(reference: np.ndarray, target: np.ndarray, count: int, *, positive: bool) -> tuple[np.ndarray, np.ndarray]:
    reference, target = np.asarray(reference, dtype=float), np.asarray(target, dtype=float)
    if reference.shape != (count,) or target.shape != (count,) or not np.isfinite(reference).all() or not np.isfinite(target).all():
        raise ValueError("Require finite reference and target vectors matching XY")
    if np.any(reference < 0) or np.any(target < 0) or (positive and (np.any(reference == 0) or np.any(target == 0))):
        raise ValueError("Fit intensities must be positive; application intensities may be zero")
    return reference, target


def _basis(xy: np.ndarray, splines: tuple[SplineTransformer, SplineTransformer]) -> sparse.csc_matrix:
    bx, by = splines[0].transform(xy[:, :1]), splines[1].transform(xy[:, 1:])
    return sparse.csc_matrix(np.einsum("ij,ik->ijk", bx, by).reshape(len(xy), -1))


def _roughness(splines: tuple[SplineTransformer, SplineTransformer]) -> sparse.csc_matrix:
    nx, ny = splines[0].n_features_out_, splines[1].n_features_out_
    dx = sparse.csc_matrix(np.diff(np.eye(nx), n=2, axis=0))
    dy = sparse.csc_matrix(np.diff(np.eye(ny), n=2, axis=0))
    return sparse.vstack([sparse.kron(dx, sparse.eye(ny)), sparse.kron(sparse.eye(nx), dy)], format="csc")


def _objective(residual, alpha, beta, roughness, *, penalty, quantile) -> float:
    return float(np.maximum(quantile * residual, (quantile - 1) * residual).sum()
                 + len(residual) * penalty / 2 * (np.square(roughness @ alpha).sum()
                                                + np.square(roughness @ beta).sum()))


@dataclass(frozen=True)
class SpatialSubtraction:
    """Reloadable physical XY fields; basis extrapolation clamps to training edges."""

    splines: tuple[SplineTransformer, SplineTransformer]
    domain_xy_um: np.ndarray
    alpha_coefficients: np.ndarray
    beta_scaled_coefficients: np.ndarray
    penalty: float
    quantile: float
    diagnostics: dict[str, float | int | str]

    def fields(self, xy_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        basis = _basis(_xy(xy_um), self.splines)
        return np.asarray(basis @ self.alpha_coefficients), np.asarray(basis @ self.beta_scaled_coefficients) * INTENSITY_SCALE

    def predict(self, xy_um: np.ndarray, reference: np.ndarray) -> np.ndarray:
        xy = _xy(xy_um)
        reference = np.asarray(reference, dtype=float)
        if reference.shape != (len(xy),) or not np.isfinite(reference).all() or np.any(reference < 0):
            raise ValueError("Reference must be finite, nonnegative and match XY")
        alpha, beta = self.fields(xy)
        return alpha * reference + beta

    def signed_residual(self, xy_um: np.ndarray, reference: np.ndarray, target: np.ndarray) -> np.ndarray:
        xy = _xy(xy_um)
        reference, target = _paired(reference, target, len(xy), positive=False)
        residual = target - self.predict(xy, reference)
        residual[(reference == 0) | (target == 0)] = 0
        return residual

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> SpatialSubtraction:
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise ValueError("Artifact is not a spatial subtraction model")
        return model


def fit_spatial_subtraction(
    xy_um: np.ndarray, reference: np.ndarray, target: np.ndarray, *,
    penalty: float, knots_xy: tuple[int, int] = (3, 4), quantile: float = 0.05,
    tolerance: float = 1e-5, max_iter: int = 30_000, time_limit: float = 120.0, verbose: bool = False,
    objective_scale: float = 1.0,
    relative_tolerance: float = 1e-6,
) -> SpatialSubtraction:
    """Minimize mean pinball loss + penalty/2 times coefficient second differences.

    Target and reference are scaled by 1000. The multiplier coefficients are
    dimensionless and constrained nonnegative; offset coefficients are in
    target units / 1000. Both tensor axes have uniform cubic knots spanning
    the training coordinates. Roughness is an unweighted sum on their fixed
    coefficient lattice, with no physical derivative or ridge penalty.
    Accept only OSQP ``solved``, at most one raw target unit of primal
    infeasibility after enforcing multiplier bounds, and a relative primal /
    dual objective gap at most 1e-5. Objective normalization changes neither
    the optimizer nor the ratio of pinball loss to roughness.
    """
    import osqp

    started = time.perf_counter()
    xy = _xy(xy_um)
    reference, target = _paired(reference, target, len(xy), positive=True)
    if not np.isfinite(penalty) or penalty <= 0 or not 0 < quantile < 1:
        raise ValueError("Require positive finite penalty and quantile in (0, 1)")
    if len(knots_xy) != 2 or any(k < 2 or not isinstance(k, int) for k in knots_xy):
        raise ValueError("Require two integer knot counts >= 2")
    if tolerance <= 0 or not np.isfinite(tolerance) or max_iter < 1 or time_limit <= 0 or not np.isfinite(time_limit):
        raise ValueError("Require valid positive solver tolerance, iteration and time limits")
    if not np.isfinite(objective_scale) or objective_scale <= 0:
        raise ValueError("Objective scale must be finite and positive")
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0:
        raise ValueError("Relative tolerance must be finite and positive")
    if len(xy) < 2 * (knots_xy[0] + 2) * (knots_xy[1] + 2) or np.any(np.ptp(xy, axis=0) <= 0) or np.ptp(reference) == 0:
        raise ValueError("Insufficient samples or coordinate/reference variation for varying coefficients")
    domain = np.column_stack([xy.min(axis=0), xy.max(axis=0)])
    splines = tuple(SplineTransformer(n_knots=k, degree=3, knots="uniform", include_bias=True, extrapolation="constant").fit(xy[:, i:i+1]) for i, k in enumerate(knots_xy))
    basis = _basis(xy, splines)
    count, width = basis.shape
    roughness = _roughness(splines)
    gram = roughness.T @ roughness
    # Multiply the complete objective by N to avoid tiny slack costs. This is
    # equivalent to mean pinball + lambda/2 * roughness in the scaled units.
    penalty_matrix = sparse.block_diag([count * penalty * gram, count * penalty * gram, sparse.csc_matrix((2 * count, 2 * count))], format="csc")
    design = sparse.hstack([basis.multiply(reference[:, None] / INTENSITY_SCALE), basis], format="csc")
    equality = sparse.hstack([design, sparse.eye(count), -sparse.eye(count)], format="csc")
    bounds = np.full(2 * width + 2 * count, -np.inf)
    bounds[:width] = 0
    bounds[2 * width:] = 0
    constraint_scales = np.ones(len(bounds))
    constraint_scales[:width] = reference.max() / INTENSITY_SCALE
    # The equivalent alpha >= 0 rows express coefficient-bound errors at the
    # largest training reference intensity, rather than as raw ratios.
    constraints = sparse.vstack([equality, sparse.diags(constraint_scales)], format="csc")
    response = target / INTENSITY_SCALE
    solver = osqp.OSQP()
    solver.setup(P=sparse.triu(penalty_matrix * objective_scale, format="csc"), q=objective_scale * np.concatenate([np.zeros(2 * width), np.full(count, quantile), np.full(count, 1-quantile)]),
                 A=constraints, l=np.concatenate([response, bounds]), u=np.concatenate([response, np.full(len(bounds), np.inf)]),
                 eps_abs=tolerance * objective_scale, eps_rel=relative_tolerance, max_iter=max_iter, time_limit=time_limit, polishing=True, verbose=verbose)
    result = solver.solve(raise_error=False)
    if result.info.status != "solved":
        raise RuntimeError(f"Spatial quantile solver did not converge: {result.info.status}; iterations={result.info.iter}, primal={result.info.prim_res}, dual={result.info.dual_res}")
    coefficients = result.x[:2*width]
    if not np.isfinite(coefficients).all() or coefficients[:width].min() < -10*tolerance:
        raise RuntimeError("Spatial quantile solver returned invalid or infeasible coefficients")
    # Enforce the bound exactly for solver errors already within its tolerance.
    alpha = np.maximum(coefficients[:width], 0)
    projected = np.concatenate([alpha, coefficients[width:]])
    slacks = result.x[2*width:]
    equality_error = design @ projected + slacks[:count] - slacks[count:] - response
    raw_primal = INTENSITY_SCALE * max(float(np.abs(equality_error).max()), float(max(0, -slacks.min())))
    residual = response - design @ projected
    projected_objective = _objective(residual, alpha, coefficients[width:], roughness, penalty=penalty, quantile=quantile)
    dual_objective = float(result.info.dual_obj_val / objective_scale)
    relative_gap = abs(projected_objective-dual_objective) / max(1, abs(projected_objective),abs(dual_objective))
    if raw_primal > 1 or relative_gap > 1e-5:
        raise RuntimeError(f"Spatial quantile solution failed accuracy contract: raw primal={raw_primal}, relative objective gap={relative_gap}")
    diagnostics = {"status": result.info.status, "iterations": result.info.iter, "primal_residual": result.info.prim_res,
                   "dual_residual": result.info.dual_res / objective_scale, "solver_seconds": result.info.run_time, "wall_seconds": time.perf_counter()-started,
                   "samples": count, "basis_per_field": width, "tolerance": tolerance, "max_iter": max_iter,
                   "time_limit": time_limit, "objective_sum_scaled": result.info.obj_val / objective_scale,
                   "objective_scale": objective_scale, "effective_absolute_tolerance": tolerance * objective_scale,
                   "relative_tolerance": relative_tolerance, "alpha_bound_row_scale": float(constraint_scales[0]),
                   "accuracy_contract": ACCURACY_CONTRACT, "raw_primal_feasibility": raw_primal,
                   "relative_projected_objective_gap": relative_gap, "projected_objective_sum_scaled": projected_objective,
                   "dual_objective_sum_scaled": dual_objective, "polish_status": result.info.status_polish,
                   "raw_mean_objective": projected_objective * INTENSITY_SCALE / count,
                   "raw_mean_primal_dual_gap": (projected_objective-dual_objective) * INTENSITY_SCALE / count,
                   "rho_updates": result.info.rho_updates, "rho_estimate": result.info.rho_estimate,
                   "minimum_alpha_coefficient_before_projection": float(coefficients[:width].min())}
    return SpatialSubtraction(splines, domain, alpha, coefficients[width:], float(penalty), float(quantile), diagnostics)
