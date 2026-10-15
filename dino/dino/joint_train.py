"""Fit one Cell-DINO logistic classifier per marker across light-sheet datasets.

Embedding weights are shared. As in the RHU2 refit (`rhu2_train --section-offsets`),
each dataset gets its own regularized intercept, encoded as an indicator column;
each dataset also gets its own coefficients on the intensity terms, encoded as
indicator-masked copies of the context features. Within one dataset those extra
columns are constant, so the joint fit collapses exactly into one standard
`dino-knn-intensity` bundle per dataset, which the existing scorers apply.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dino_markers import (
    CONTRACT_FIELD,
    INFERENCE,
    INTENSITY_MODEL,
    JOINT_MODEL,
    MarkerContract,
    bundle_contract,
    contract_of,
    cutoff_key,
    metric_key,
    model_key,
    oof_probability_column,
    prediction_column,
    probability_column,
    require_same_contract,
    scaler_key,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

from dino.artifacts import (
    _dump_joblib_atomic,
    file_sha256,
    ordered_cell_sha256,
    read_embedding_artifact,
    require_embedding_markers,
)
from dino.classifier_features import load_context
from dino.training_split import split_dataset_rows
from dino.zarr_model import balanced_accuracy_cutoff
from dino.zarr_source import write_json_atomic
from dino.zarr_train import _best_logreg_c

CV_FOLDS = 5
INTENSITY_NEIGHBORS = 16
EMBEDDING_WIDTH = 1024
ENCODER_KEYS = (
    'feature_extractor', 'weights_sha256', 'model_source_sha256', 'crop_size', 'resize_size', 'crop_eval_size',
    'highpass_sigma_px', 'mask_erosion_px', 'input_channels', 'input_channel_names_by_target',
    'feature_precision', 'feature_execution',
)
# Classifier state inherited from the embedding bundle that a deployed bundle must not carry.
STALE_KEYS = (
    'embedding_pca', 'deployment', 'feature_cache', 'training_barrage_dirs', 'classifier_feature_schema',
    'feature_names', 'position_artifact', 'position_features', 'position_spline', 'position_spline_config',
    'test_fraction', 'split_source_model', 'split_source_model_sha256', 'label_snapshot',
    'label_snapshot_sha256', 'training_labels_snapshot', 'training_labels_snapshot_sha256',
)


@dataclass(frozen=True)
class Dataset:
    name: str
    tranches: tuple[Path, ...]
    embedding_run: Path
    embedding_model: Path
    intensity_props: Path
    intensity_provenance: Path
    dapi_props: Path
    scale_zyx_um: tuple[float, float, float]


def load_config(path: Path) -> tuple[list[Dataset], int, int, tuple[str, ...]]:
    config = json.loads(path.read_text())
    datasets = [
        Dataset(name=item['name'], tranches=tuple(Path(p) for p in item['tranches']),
                embedding_run=Path(item['embedding_run']), embedding_model=Path(item['embedding_model']),
                intensity_props=Path(item['intensity_props']), intensity_provenance=Path(item['intensity_provenance']),
                dapi_props=Path(item['dapi_props']), scale_zyx_um=tuple(item['scale_zyx_um']))
        for item in config['datasets']
    ]
    if len({d.name for d in datasets}) != len(datasets) or len(datasets) < 2:
        raise ValueError('A joint fit needs at least two uniquely named datasets')
    return datasets, int(config['seed']), int(config['validation_cells']), tuple(config['context_features'])


def read_labels(dataset: Dataset, *, contract: MarkerContract) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Freeze each live tranche once, hashing exactly the bytes that were parsed."""
    markers = list(contract.markers)
    frames, records = [], []
    for tranche in dataset.tranches:
        require_same_contract(contract, contract_of(tranche), expected_from=str(dataset.embedding_model),
                              actual_from=str(tranche))
        raw = (tranche / 'labels.csv').read_bytes()
        frame = pd.read_csv(io.BytesIO(raw), usecols=['cell', *markers])
        frame['tranche'] = tranche.name
        frames.append(frame)
        records.append({'path': str(tranche.resolve()), 'labels': len(frame),
                        'labels_sha256': hashlib.sha256(raw).hexdigest()})
    labels = pd.concat(frames, ignore_index=True)
    labels['dataset'] = dataset.name
    if labels.cell.duplicated().any():
        raise ValueError(f'{dataset.name} tranches contain duplicate cell {labels.cell[labels.cell.duplicated()].iloc[0]}')
    if not labels[markers].isin([0, 1]).all().all():
        raise ValueError(f'{dataset.name} labels must be binary')
    labels[markers] = labels[markers].astype(np.int64)
    return labels, records


def partition_embeddings(path: Path, embedding_path: Path, wanted: set[str],
                         markers: tuple[str, ...]) -> tuple[list[str], list[int], dict[str, np.ndarray]]:
    """Return the requested cells of one persisted partition, in partition order."""
    rows = pq.read_table(path, columns=['cell', 'label']).to_pandas()
    selected = np.flatnonzero(rows.cell.isin(wanted).to_numpy())
    if not len(selected):
        return [], [], {}
    artifact = read_embedding_artifact(embedding_path)
    require_embedding_markers(artifact, markers, path=embedding_path)
    if not torch.equal(artifact['cell_sha256'], ordered_cell_sha256(rows.cell.tolist())):
        raise ValueError(f'Prediction and embedding identities differ: {path}')
    index = torch.from_numpy(selected)
    return (rows.cell.iloc[selected].tolist(), rows.label.iloc[selected].astype(int).tolist(),
            {t: artifact[t][index].float().numpy() for t in markers})


def labeled_embeddings(cells: list[str], run: Path, *, model_sha256: str, source_sha256: str,
                       workers: int, contract: MarkerContract) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Collect persisted marker embeddings and source labels for the requested identities."""
    wanted = set(cells)
    partitions = []
    for shard in sorted(run.glob('shard-*')):
        complete = json.loads((shard / 'complete.json').read_text())
        require_same_contract(contract, INFERENCE.contract(complete, where=str(shard / 'complete.json')),
                              expected_from='the embedding bundle', actual_from=str(shard / 'complete.json'))
        if not complete.get('stores_embeddings'):
            raise ValueError(f'Embeddings were not persisted for {shard}')
        if complete.get('model_sha256') != model_sha256 or complete.get('source_sha256') != source_sha256:
            raise ValueError(f'Embedding model or source identity differs for {shard}')
        partitions.extend((path, shard / 'embeddings' / path.with_suffix('.pt').name)
                          for path in sorted((shard / 'predictions').glob('part-*.parquet')))
    found: dict[str, tuple[int, dict[str, np.ndarray]]] = {}
    for names, labels, values in joblib.Parallel(n_jobs=workers)(
            joblib.delayed(partition_embeddings)(path, embedding_path, wanted, contract.markers)
            for path, embedding_path in partitions):
        for row, (cell, label) in enumerate(zip(names, labels)):
            if cell in found:
                raise ValueError(f'Embedding run contains duplicate cell {cell}')
            found[cell] = (label, {t: values[t][row] for t in contract.markers})
    missing = sorted(wanted - set(found))
    if missing:
        raise ValueError(f'{len(missing)} labeled cells are absent from {run}, e.g. {missing[0]}')
    source_labels = np.array([found[cell][0] for cell in cells], dtype=np.int64)
    if not np.array_equal(source_labels, [int(cell.rsplit('|', 1)[1]) for cell in cells]):
        raise ValueError('Cell identities do not match source labels')
    return source_labels, {t: np.stack([found[cell][1][t] for cell in cells]) for t in contract.markers}


def design_matrix(embedding: np.ndarray, context: np.ndarray, dataset_index: np.ndarray,
                  datasets: int) -> np.ndarray:
    """Columns: shared embedding, then each dataset's masked context block, then dataset indicators."""
    indicators = np.zeros((len(embedding), datasets))
    indicators[np.arange(len(embedding)), dataset_index] = 1.
    return np.column_stack([embedding, *[context * indicators[:, [d]] for d in range(datasets)], indicators])


def collapse(scaler: StandardScaler, model: LogisticRegression, *, index: int, datasets: int,
             embedding_width: int, context_width: int) -> tuple[StandardScaler, LogisticRegression]:
    """Fold one dataset's constant columns into the intercept of an embedding+context classifier."""
    total = embedding_width + datasets * (context_width + 1)
    if scaler.n_features_in_ != total or model.coef_.shape != (1, total):
        raise ValueError(f'Joint classifier must have {total} features')
    own = np.r_[np.arange(embedding_width), embedding_width + index * context_width + np.arange(context_width)]
    fixed = np.zeros(total)
    fixed[embedding_width + datasets * context_width + index] = 1.
    dropped = np.setdiff1d(np.arange(total), own)
    offset = np.sum(model.coef_[0, dropped] * (fixed[dropped] - scaler.mean_[dropped]) / scaler.scale_[dropped])
    reduced_scaler = copy.deepcopy(scaler)
    reduced_scaler.mean_, reduced_scaler.var_, reduced_scaler.scale_ = (
        scaler.mean_[own], scaler.var_[own], scaler.scale_[own])
    reduced_scaler.n_features_in_ = len(own)
    reduced_model = copy.deepcopy(model)
    reduced_model.coef_ = model.coef_[:, own].copy()
    reduced_model.intercept_ = model.intercept_ + offset
    reduced_model.n_features_in_ = len(own)
    return reduced_scaler, reduced_model


def fit_and_score(train_x: np.ndarray, train_y: np.ndarray, c_value: float,
                  scored_x: np.ndarray) -> tuple[StandardScaler, LogisticRegression, np.ndarray]:
    scaler = StandardScaler().fit(train_x)
    with threadpool_limits(limits=1):
        model = LogisticRegression(solver='lbfgs', C=c_value, max_iter=1_000, tol=1e-12)
        model.fit(scaler.transform(train_x), train_y)
    return scaler, model, model.predict_proba(scaler.transform(scored_x))[:, 1]


def summarize(labels: np.ndarray, probability: np.ndarray, cutoff: float) -> dict[str, float | int | None]:
    both = len(np.unique(labels)) == 2
    return {'cells': len(labels), 'positive': int(labels.sum()),
            'auc': float(roc_auc_score(labels, probability)) if both else None,
            'balanced_accuracy': float(balanced_accuracy_score(labels, probability >= cutoff)) if both else None,
            'accuracy': float(np.mean(labels == (probability >= cutoff))),
            'brier': float(brier_score_loss(labels, probability, labels=[0, 1]))}


def fit(config_path: Path, output: Path, *, workers: int) -> None:
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing joint fit: {output}')
    datasets, seed, validation_cells, context_features = load_config(config_path)
    names = [d.name for d in datasets]
    frames, tranches, embeddings, contexts, bundles, provenance = [], {}, [], [], {}, {}
    contract = None
    for dataset in datasets:
        bundle = joblib.load(dataset.embedding_model)
        dataset_contract = bundle_contract(bundle, where=str(dataset.embedding_model))
        if contract is None:
            contract = dataset_contract
        else:
            require_same_contract(contract, dataset_contract, expected_from=str(datasets[0].embedding_model),
                                  actual_from=str(dataset.embedding_model))
        labels, tranches[dataset.name] = read_labels(dataset, contract=contract)
        model_sha256 = file_sha256(dataset.embedding_model)
        source_labels, values = labeled_embeddings(
            labels.cell.astype(str).tolist(), dataset.embedding_run,
            model_sha256=model_sha256, source_sha256=bundle['source_sha256'], workers=workers, contract=contract)
        record = json.loads(dataset.intensity_provenance.read_text())
        if record['props_sha256'] != file_sha256(dataset.intensity_props):
            raise ValueError(f'{dataset.name} intensity provenance does not match its props')
        context = load_context(context_features, dataset.intensity_props, dataset.dapi_props,
                               scale_zyx_um=np.asarray(dataset.scale_zyx_um), markers=contract.markers)
        labels['label'] = source_labels
        frames.append(labels)
        embeddings.append(values)
        contexts.append(context.features_for_labels(source_labels, neighbors=INTENSITY_NEIGHBORS))
        bundles[dataset.name] = bundle
        provenance[dataset.name] = {
            'embedding_model': str(dataset.embedding_model.resolve()), 'embedding_model_sha256': model_sha256,
            'embedding_run': str(dataset.embedding_run.resolve()), 'source_sha256': bundle['source_sha256'],
            'intensity_props': str(dataset.intensity_props.resolve()),
            'intensity_props_sha256': record['props_sha256'], 'intensity_measurement': record,
            'dapi_props': str(dataset.dapi_props.resolve()), 'dapi_props_sha256': file_sha256(dataset.dapi_props),
            'intensity_scale_zyx_um': list(dataset.scale_zyx_um), 'tranches': tranches[dataset.name]}
    reference = {key: bundles[names[0]][key] for key in ENCODER_KEYS}
    for name in names[1:]:
        if {key: bundles[name][key] for key in ENCODER_KEYS} != reference:
            raise ValueError(f'{name} embeddings use a different encoder than {names[0]}')

    markers = contract.markers
    rows = pd.concat(frames, ignore_index=True)
    dataset_index = pd.Index(names).get_indexer(rows.dataset)
    emb = {t: np.concatenate([e[t] for e in embeddings]) for t in markers}
    ctx = {t: np.concatenate([c[t] for c in contexts]) for t in markers}
    x = {t: design_matrix(emb[t], ctx[t], dataset_index, len(names)) for t in markers}
    train, validation = split_dataset_rows(rows, markers=markers, dataset_index=dataset_index,
                                           datasets=len(names), test_size=validation_cells, seed=seed)
    rows['split'] = 'training'
    rows.loc[validation, 'split'] = 'validation'

    joint = JOINT_MODEL.stamp(
        {**reference, 'datasets': names, 'context_features': list(context_features),
         'feature_layout': ['embedding'] + [f'{f}|{n}' for n in names for f in context_features]
         + [f'intercept|{n}' for n in names],
         'intensity_neighbors': INTENSITY_NEIGHBORS, 'split_seed': seed,
         'validation_split': f'random, stratified by dataset and joint {"/".join(markers)} label, '
                             'falling back to dataset and each marker in order',
         'train_cells': rows.cell.iloc[train].tolist(), 'validation_cells': rows.cell.iloc[validation].tolist(),
         'datasets_provenance': provenance, 'created_at_utc': datetime.now(timezone.utc).isoformat()},
        contract)
    metrics: dict[str, object] = {'datasets': {n: {'labels': int((rows.dataset == n).sum()),
                                                   'training': int(((rows.dataset == n) & (rows.split == 'training')).sum()),
                                                   'validation': int(((rows.dataset == n) & (rows.split == 'validation')).sum())}
                                               for n in names},
                                  'split_seed': seed, 'targets': {}}
    for target in markers:
        y = rows[target].to_numpy()
        best_c, cv_score, folds = _best_logreg_c(x[target][train], y[train], cv_folds=CV_FOLDS, seed=seed,
                                                 n_jobs=workers)
        # Out-of-fold probabilities per dataset: each fold of that dataset's labels is scored by a
        # classifier fitted at the selected C on every other labeled cell of all datasets.
        held_out = []
        for d in range(len(names)):
            members = np.flatnonzero(dataset_index == d)
            splitter = StratifiedKFold(n_splits=min(CV_FOLDS, int(np.bincount(y[members]).min())),
                                       shuffle=True, random_state=seed)
            held_out.extend(members[held] for _, held in splitter.split(members, y[members]))
        fits = [(train, np.arange(len(rows)))] + [
            (np.setdiff1d(np.arange(len(rows)), held), held) for held in held_out]
        (scaler, model, p), *oof_fits = joblib.Parallel(n_jobs=workers)(
            joblib.delayed(fit_and_score)(x[target][keep], y[keep], best_c, x[target][scored])
            for keep, scored in fits)
        cutoff, _ = balanced_accuracy_cutoff(y[validation], p[validation])
        rows[probability_column(target)] = p
        rows[prediction_column(target)] = p >= cutoff
        oof = np.full(len(rows), np.nan)
        for held, (_, _, held_p) in zip(held_out, oof_fits):
            oof[held] = held_p
        rows[oof_probability_column(target)] = oof
        joint.update({scaler_key(target): scaler, model_key(target): model, cutoff_key(target): cutoff})
        metrics['targets'][target] = {
            'best_c': best_c, 'cv_folds': folds, 'cv_balanced_accuracy': cv_score, 'cutoff': cutoff,
            'validation': summarize(y[validation], p[validation], cutoff),
            'validation_by_dataset': {n: summarize(y[validation][dataset_index[validation] == d],
                                                   p[validation][dataset_index[validation] == d], cutoff)
                                      for d, n in enumerate(names)},
            'out_of_fold_by_dataset': {n: summarize(y[dataset_index == d], oof[dataset_index == d], cutoff)
                                       for d, n in enumerate(names)}}
        print(f'{target}: {json.dumps(metrics["targets"][target])}', flush=True)

    _dump_joblib_atomic(joint, output / 'model.joblib')
    joint_sha256 = file_sha256(output / 'model.joblib')
    rows.to_csv(output / 'labels.csv', columns=['cell', *markers, 'tranche', 'dataset'], index=False)
    rows.to_parquet(output / 'labeled_predictions.parquet', index=False)
    for d, dataset in enumerate(datasets):
        deployed = {key: value for key, value in bundles[dataset.name].items()
                    if key not in STALE_KEYS and key not in ('schema', CONTRACT_FIELD)}
        members = np.flatnonzero(dataset_index == d)
        for target in markers:
            scaler, model = collapse(joint[scaler_key(target)], joint[model_key(target)], index=d,
                                     datasets=len(names), embedding_width=EMBEDDING_WIDTH,
                                     context_width=len(context_features))
            np.testing.assert_allclose(
                model.predict_proba(scaler.transform(np.column_stack([emb[target][members], ctx[target][members]])))[:, 1],
                rows[probability_column(target)].to_numpy()[members], rtol=0, atol=1e-10)
            deployed.update({scaler_key(target): scaler, model_key(target): model,
                             cutoff_key(target): joint[cutoff_key(target)],
                             metric_key(target, 'best_c'): metrics['targets'][target]['best_c']})
        deployed.update({
            'classifier_feature_mode': 'marker',
            'context_features': list(context_features), 'intensity_neighbors': INTENSITY_NEIGHBORS,
            'intensity_scale_zyx_um': provenance[dataset.name]['intensity_scale_zyx_um'],
            'source_sha256': provenance[dataset.name]['source_sha256'],
            'intensity_props': provenance[dataset.name]['intensity_props'],
            'intensity_props_sha256': provenance[dataset.name]['intensity_props_sha256'],
            'intensity_measurement': provenance[dataset.name]['intensity_measurement'],
            'dapi_props': provenance[dataset.name]['dapi_props'],
            'dapi_props_sha256': provenance[dataset.name]['dapi_props_sha256'],
            'embedding_run': provenance[dataset.name]['embedding_run'],
            'embedding_model': provenance[dataset.name]['embedding_model'],
            'training_intensity_labels': dict(zip(rows.cell.iloc[members], rows.label.iloc[members].tolist())),
            'train_cells': rows.cell.iloc[members][rows.split.iloc[members] == 'training'].tolist(),
            'validation_cells': rows.cell.iloc[members][rows.split.iloc[members] == 'validation'].tolist(),
            'split_seed': seed, 'created_at_utc': joint['created_at_utc'],
            'joint_model': {'path': str((output / 'model.joblib').resolve()), 'sha256': joint_sha256,
                            'schema': joint['schema'], 'dataset': dataset.name, 'datasets': names}})
        deployed = INTENSITY_MODEL.stamp(deployed, contract)
        destination = output / 'deploy' / dataset.name
        _dump_joblib_atomic(deployed, destination / 'model.joblib')
        rows.iloc[members].to_csv(destination / 'labels.csv', columns=['cell', *markers, 'tranche'], index=False)
        metrics['datasets'][dataset.name]['deployed_model'] = str((destination / 'model.joblib').resolve())
        metrics['datasets'][dataset.name]['deployed_model_sha256'] = file_sha256(destination / 'model.joblib')
    metrics.update({'joint_model': str((output / 'model.joblib').resolve()), 'joint_model_sha256': joint_sha256,
                    'classifier_features': {t: int(joint[model_key(t)].n_features_in_) for t in markers},
                    'feature_layout': joint['feature_layout'], 'tranches': tranches})
    write_json_atomic(metrics, output / 'metrics.json')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, required=True,
                        help='Processes for the independent single-threaded logistic fits')
    args = parser.parse_args()
    fit(args.config, args.output, workers=args.workers)


if __name__ == '__main__':
    main()
