"""Grouped, nested evaluation of shared and dataset-adjusted intensity classifiers.

The hierarchical design standardizes embedding and three intensity terms first.
Only intensity slopes and intercepts receive dataset deviations. Scaling the
four local columns by sqrt(lambda0 / lambda_delta) gives their requested ridge
penalty under sklearn's summed-loss logistic regression objective.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from dino.joint_train import collapse as collapse_joint
from dino.joint_train import design_matrix
from dino.zarr_model import balanced_accuracy_cutoff

LAMBDA0_GRID = (1., 10., 100.)
LAMBDA_DELTA_GRID = (1., 10., 100., np.inf)


def _validate(x, y, dataset_index, datasets, embedding_width):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    d = np.asarray(dataset_index)
    if not isinstance(embedding_width, (int, np.integer)) or embedding_width < 1:
        raise ValueError('embedding_width must be a positive integer')
    if not isinstance(datasets, (int, np.integer)) or datasets < 2:
        raise ValueError('datasets must be an integer of at least two')
    if x.ndim != 2 or x.shape[1] != embedding_width + 3 or not np.isfinite(x).all():
        raise ValueError('Features must be finite embedding plus exactly three context columns')
    if y.shape != (len(x),) or not np.isin(y, [0, 1]).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError('Labels must contain both binary classes')
    if d.shape != y.shape or not np.issubdtype(d.dtype, np.integer) or not np.isin(d, np.arange(datasets)).all():
        raise ValueError('Dataset indices must be integers in the configured range')
    if set(np.unique(d)) != set(range(datasets)):
        raise ValueError('Every configured dataset must be present')
    return x, y.astype(np.int64), d.astype(np.int64)


def _local_design(scaled, d, datasets, embedding_width, local_scale):
    if local_scale == 0:
        return scaled
    local = np.zeros((len(scaled), datasets, 4))
    local[np.arange(len(scaled)), d, :3] = scaled[:, embedding_width:]
    local[np.arange(len(scaled)), d, 3] = 1.
    return np.column_stack([scaled, local_scale * local.reshape(len(scaled), -1)])


@dataclass
class FittedModel:
    family: str
    scaler: StandardScaler
    model: LogisticRegression
    datasets: int
    embedding_width: int
    lambda0: float
    lambda_delta: float
    local_scale: float

    def predict_proba(self, x: np.ndarray, dataset_index: np.ndarray) -> np.ndarray:
        """Return positive-class probabilities in float64; scoring calls cast to float32."""
        x = np.asarray(x, dtype=np.float64)
        d = np.asarray(dataset_index)
        if x.ndim != 2 or x.shape[1] != self.embedding_width + 3 or not np.isfinite(x).all():
            raise ValueError('Prediction features must be finite embedding plus three context columns')
        if d.shape != (len(x),) or not np.issubdtype(d.dtype, np.integer) or not np.isin(d, np.arange(self.datasets)).all():
            raise ValueError('Prediction dataset indices must be integers in the configured range')
        if self.family == 'joint':
            design = self.scaler.transform(design_matrix(x[:, :self.embedding_width], x[:, self.embedding_width:], d, self.datasets))
        else:
            design = _local_design(self.scaler.transform(x), d, self.datasets, self.embedding_width, self.local_scale)
        return self.model.predict_proba(design)[:, 1]

    def collapse(self, index: int) -> tuple[StandardScaler, LogisticRegression]:
        """Produce the standard embedding+context scorer objects for one dataset."""
        if not isinstance(index, (int, np.integer)) or not 0 <= index < self.datasets:
            raise ValueError('Collapse dataset index is outside the configured range')
        if self.family == 'joint':
            return collapse_joint(self.scaler, self.model, index=index, datasets=self.datasets,
                                  embedding_width=self.embedding_width, context_width=3)
        width = self.embedding_width + 3
        model = copy.deepcopy(self.model)
        model.coef_ = self.model.coef_[:, :width].copy()
        model.intercept_ = self.model.intercept_.copy()
        if self.local_scale:
            local = self.local_scale * self.model.coef_[0, width:].reshape(self.datasets, 4)[index]
            model.coef_[0, self.embedding_width:] += local[:3]
            model.intercept_[0] += local[3]
        model.n_features_in_ = width
        return copy.deepcopy(self.scaler), model


def fit_model(x, y, dataset_index, *, datasets: int, embedding_width: int = 1024,
              family: str = 'hierarchical', lambda0: float = 10., lambda_delta: float = 10.,
              max_iter: int = 5000, tol: float = 1e-9) -> FittedModel:
    x, y, d = _validate(x, y, dataset_index, datasets, embedding_width)
    if family not in ('hierarchical', 'joint'):
        raise ValueError('family must be hierarchical or joint')
    if not np.isfinite(lambda0) or lambda0 <= 0 or np.isnan(lambda_delta) or lambda_delta <= 0:
        raise ValueError('Penalties must be positive; only lambda_delta may be infinite')
    local_scale = 0. if np.isinf(lambda_delta) else np.sqrt(lambda0 / lambda_delta)
    if family == 'joint':
        x = design_matrix(x[:, :embedding_width], x[:, embedding_width:], d, datasets)
    scaler = StandardScaler().fit(x)
    design = scaler.transform(x)
    if family == 'hierarchical':
        design = _local_design(design, d, datasets, embedding_width, local_scale)
    model = LogisticRegression(solver='lbfgs', C=1. / lambda0, max_iter=max_iter, tol=tol)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        try:
            model.fit(design, y)
        except ConvergenceWarning as exc:
            raise RuntimeError('Logistic classifier did not converge') from exc
    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        raise RuntimeError('Logistic classifier has nonfinite fitted coefficients')
    return FittedModel(family, scaler, model, datasets, embedding_width, lambda0, lambda_delta, local_scale)


def _groups(groups, rows):
    groups = np.asarray(groups)
    if groups.shape != (rows,) or any(value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)) for value in groups):
        raise ValueError('Groups must contain one nonmissing identifier per row')
    return groups


def _check_folds(folds, y, d, groups, datasets):
    tested = np.zeros(len(y), dtype=int)
    checked = []
    for train, test in folds:
        train, test = np.asarray(train), np.asarray(test)
        if (train.ndim != 1 or test.ndim != 1 or not len(train) or not len(test)
                or not np.issubdtype(train.dtype, np.integer) or not np.issubdtype(test.dtype, np.integer)
                or not np.isin(np.r_[train, test], np.arange(len(y))).all()
                or len(np.unique(train)) != len(train) or len(np.unique(test)) != len(test)
                or np.intersect1d(train, test).size or len(train) + len(test) != len(y)):
            raise ValueError('Each fold must partition all rows into distinct train and test indices')
        if np.intersect1d(groups[train], groups[test]).size:
            raise ValueError('Spatial groups overlap training and test folds')
        for part in (train, test):
            for index in range(datasets):
                if set(np.unique(y[part[d[part] == index]])) != {0, 1}:
                    raise ValueError('Each fold requires both classes in every dataset, in training and test')
        tested[test] += 1
        checked.append((train, test))
    if not checked or not np.all(tested == 1):
        raise ValueError('Fold tests must cover every row exactly once')
    return checked


def grouped_folds(y, dataset_index, groups, *, datasets: int, n_splits: int, seed: int = 0):
    """Partition whole spatial blocks, stratifying by dataset and binary class."""
    y, d = np.asarray(y), np.asarray(dataset_index)
    if y.ndim != 1 or not np.isin(y, [0, 1]).all() or d.shape != y.shape or not np.issubdtype(d.dtype, np.integer) or not np.isin(d, np.arange(datasets)).all():
        raise ValueError('Fold labels and dataset indices are invalid')
    groups = _groups(groups, len(y))
    if n_splits < 2 or len(np.unique(groups)) < n_splits:
        raise ValueError('Not enough spatial groups for the requested folds')
    return _check_folds(
        list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
             .split(np.zeros((len(y), 1)), d * 2 + y, groups)),
        y, d, groups, datasets)


@dataclass
class Selection:
    lambda0: float
    lambda_delta: float
    loss: float
    probabilities: np.ndarray
    candidates: list[dict[str, float]]


def select_parameters(x, y, dataset_index, groups, *, datasets: int, embedding_width: int = 1024,
                      family: str = 'hierarchical', n_splits: int = 3, seed: int = 0,
                      lambda0_grid=LAMBDA0_GRID, lambda_delta_grid=LAMBDA_DELTA_GRID) -> Selection:
    """Choose penalties by macro-dataset grouped OOF log loss, on supplied rows only."""
    x, y, d = _validate(x, y, dataset_index, datasets, embedding_width)
    folds = grouped_folds(y, d, groups, datasets=datasets, n_splits=n_splits, seed=seed)
    if family not in ('hierarchical', 'joint') or not len(lambda0_grid) or not len(lambda_delta_grid):
        raise ValueError('A valid family and nonempty penalty grids are required')
    candidates, best = [], None
    for lambda0 in lambda0_grid:
        for delta in (lambda_delta_grid if family == 'hierarchical' else (np.inf,)):
            probability = np.empty(len(y))
            for train, test in folds:
                probability[test] = fit_model(
                    x[train], y[train], d[train], datasets=datasets, embedding_width=embedding_width,
                    family=family, lambda0=lambda0, lambda_delta=delta).predict_proba(x[test], d[test])
            loss = float(np.mean([log_loss(y[d == index], probability[d == index], labels=[0, 1]) for index in range(datasets)]))
            candidates.append({'lambda0': float(lambda0), 'lambda_delta': float(delta), 'loss': loss})
            tied = best is not None and np.isclose(loss, best.loss, rtol=1e-10, atol=1e-12)
            if best is None or (loss < best.loss and not tied) or (tied and (delta, lambda0) > (best.lambda_delta, best.lambda0)):
                best = Selection(float(lambda0), float(delta), loss, probability, [])
    best.candidates = candidates
    return best


def dataset_cutoffs(y, probability, dataset_index, *, datasets: int) -> np.ndarray:
    """Match deployed float32 comparisons before selecting each dataset cutoff."""
    return np.array([balanced_accuracy_cutoff(y[dataset_index == index], probability[dataset_index == index].astype(np.float32))[0]
                     for index in range(datasets)])


def dataset_metrics(y, probability, cutoffs, dataset_index, *, datasets: int) -> dict[int, dict[str, float | int]]:
    """Summarize predictions using the supplied cutoff for each row."""
    probability = np.asarray(probability, dtype=np.float32)
    metrics = {}
    for index in range(datasets):
        mask = dataset_index == index
        labels, p = y[mask], probability[mask]
        call = p >= cutoffs[mask]
        tp, fn = int(np.sum(call & (labels == 1))), int(np.sum(~call & (labels == 1)))
        fp, tn = int(np.sum(call & (labels == 0))), int(np.sum(~call & (labels == 0)))
        sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
        metrics[index] = {'cells': int(mask.sum()), 'positive': tp + fn, 'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
                          'sensitivity': sensitivity, 'specificity': specificity,
                          'balanced_accuracy': (sensitivity + specificity) / 2, 'accuracy': (tp + tn) / mask.sum(),
                          'auc': float(roc_auc_score(labels, p)), 'log_loss': float(log_loss(labels, p, labels=[0, 1]))}
    return metrics


@dataclass
class Evaluation:
    probabilities: np.ndarray
    cutoffs: np.ndarray
    fold: np.ndarray
    metrics: dict[int, dict[str, float | int]]
    selections: list[Selection]


def nested_evaluate(x, y, dataset_index, groups, *, datasets: int, embedding_width: int = 1024,
                    family: str = 'hierarchical', outer_splits: int = 4, inner_splits: int = 3,
                    seed: int = 0, outer_folds=None, lambda0_grid=LAMBDA0_GRID,
                    lambda_delta_grid=LAMBDA_DELTA_GRID) -> Evaluation:
    """Select penalties and local cutoffs strictly within each outer training fold."""
    x, y, d = _validate(x, y, dataset_index, datasets, embedding_width)
    groups = _groups(groups, len(y))
    folds = (grouped_folds(y, d, groups, datasets=datasets, n_splits=outer_splits, seed=seed)
             if outer_folds is None else _check_folds(outer_folds, y, d, groups, datasets))
    p, cutoffs, fold_id, selections = np.empty(len(y)), np.empty(len(y)), np.empty(len(y), dtype=int), []
    for fold, (train, test) in enumerate(folds):
        selected = select_parameters(x[train], y[train], d[train], groups[train], datasets=datasets,
                                     embedding_width=embedding_width, family=family, n_splits=inner_splits,
                                     seed=seed + fold + 1, lambda0_grid=lambda0_grid, lambda_delta_grid=lambda_delta_grid)
        local = dataset_cutoffs(y[train], selected.probabilities, d[train], datasets=datasets)
        p[test] = fit_model(
            x[train], y[train], d[train], datasets=datasets, embedding_width=embedding_width,
            family=family, lambda0=selected.lambda0, lambda_delta=selected.lambda_delta).predict_proba(x[test], d[test])
        cutoffs[test], fold_id[test] = local[d[test]], fold
        selections.append(selected)
    return Evaluation(p, cutoffs, fold_id, dataset_metrics(y, p, cutoffs, d, datasets=datasets), selections)


@dataclass
class FinalFit:
    fitted: FittedModel
    cutoffs: np.ndarray
    selection: Selection


def fit_final(x, y, dataset_index, groups, *, datasets: int, embedding_width: int = 1024,
              family: str = 'hierarchical', n_splits: int = 3, seed: int = 0,
              lambda0_grid=LAMBDA0_GRID, lambda_delta_grid=LAMBDA_DELTA_GRID) -> FinalFit:
    x, y, d = _validate(x, y, dataset_index, datasets, embedding_width)
    selected = select_parameters(x, y, d, groups, datasets=datasets, embedding_width=embedding_width,
                                 family=family, n_splits=n_splits, seed=seed,
                                 lambda0_grid=lambda0_grid, lambda_delta_grid=lambda_delta_grid)
    return FinalFit(
        fit_model(x, y, d, datasets=datasets, embedding_width=embedding_width, family=family,
                  lambda0=selected.lambda0, lambda_delta=selected.lambda_delta),
        dataset_cutoffs(y, selected.probabilities, d, datasets=datasets), selected)
