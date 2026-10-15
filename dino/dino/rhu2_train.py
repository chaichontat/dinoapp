"""Refit cortical RHU2 labels with the deployed marker-specific Cell-DINO recipe."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tomllib
from typing import Literal

import anndata as ad
import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
import zarr

from dino.intensity_context import IntensityContext
from dino.zarr_model import (
    balanced_accuracy_cutoff, build_cell_transform, extract_normalized_features, feature_dtype,
    load_cell_dino_model, prepare_model_batch,
)
from dino.zarr_source import CellCrop, cell_dino_images
from dino.zarr_train import _best_logreg_c

TARGETS = ('brdu', 'edu')
CONTEXT_FEATURES = ('log1p_cell_mean', 'log_cell_over_local_mean', 'log1p_pi_mean')
ENCODER_KEYS = (
    'feature_extractor', 'input_channels', 'input_channel_names_by_target',
    'weights', 'weights_sha256', 'dinov2_repo', 'crop_size', 'resize_size',
    'crop_eval_size', 'highpass_sigma_px', 'mask_erosion_px', 'feature_precision',
    'feature_execution',
)


def resolve_spacing(rois: list[dict], z_step_um: float | None) -> dict:
    """Resolve physical spacing once, before selecting cells for native measurement."""
    spacing = {}
    for source in rois:
        acquisition = Path(source['fused']).parents[3] / 'pi_edu.toml'
        if not acquisition.exists():
            acquisition = acquisition.with_name('all.toml')
        if acquisition.exists():
            step = float(tomllib.loads(acquisition.read_text())['params']['step'])
            provenance = str(acquisition)
        elif z_step_um is not None:
            step, provenance = z_step_um, 'user-confirmed RHU2 Z spacing'
        else:
            raise ValueError(f'Missing recorded Z spacing for {source["key"]}; confirmation required')
        scale = np.array([step, source['pixel_size_um'], source['pixel_size_um']])
        if not np.isfinite(scale).all() or (scale <= 0).any() or (z_step_um is not None and not np.isclose(step, z_step_um)):
            raise ValueError(f'Invalid or conflicting Z spacing for {source["key"]}')
        spacing[source['key']] = {'scale_zyx_um': scale.tolist(), 'z_source': provenance,
                                  'xy_source': source['geometry_source']}
    return spacing


def prepare(review: Path, output: Path, reference: Path, z_step_um: float | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'prepared.json').exists():
        raise FileExistsError('This run is already prepared')
    if not (output / 'labels_snapshot.csv').exists():
        shutil.copyfile(review / 'labels.csv', output / 'labels_snapshot.csv')
    labels = pd.read_csv(output / 'labels_snapshot.csv')
    if labels.cell.duplicated().any() or not labels[list(TARGETS)].isin([0, 1]).all().all():
        raise ValueError('Labels must have unique identities and binary marker calls')
    spatial = json.loads((review / 'spatial.json').read_text())
    spacing = resolve_spacing(spatial['rois'], z_step_um)
    data = ad.read_h5ad(spatial['cohort'], backed='r')
    columns = ['dataset', 'roi', 'z', 'y', 'x', 'brdu_mean', 'edu_mean', 'pi_mean']
    cohort = data.obs[columns].copy()
    data.file.close()
    cohort.index.name = 'cell'
    cohort = cohort.reset_index()
    if cohort.cell.duplicated().any() or len(cohort) != spatial['cells']:
        raise ValueError('Cortical cohort identity/count mismatch')
    cohort['roi'] = cohort.roi.astype(str)
    cohort['dataset'] = cohort.dataset.astype(str)
    cohort['label'] = cohort.cell.str.rsplit('|', n=1).str[-1].astype(int)
    numeric = cohort[['z', 'y', 'x', 'brdu_mean', 'edu_mean', 'pi_mean']].to_numpy()
    if not np.isfinite(numeric[:, :3]).all() or (numeric < 0).any():
        raise ValueError('Cohort coordinates must be finite; coordinates/intensities must be nonnegative')
    rows = labels.merge(cohort, on='cell', validate='one_to_one').sort_values('cell').reset_index(drop=True)
    labels.loc[~labels.cell.isin(rows.cell)].to_csv(output / 'excluded_noncohort_labels.csv', index=False)
    rows.to_parquet(output / 'training_rows.parquet', index=False)
    cohort.to_parquet(output / 'cortical_context.parquet', index=False)
    reference_bundle = joblib.load(reference)
    metadata = {
        'schema': 'rhu2-dino-knn-intensity-v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'review': str(review), 'cohort': spatial['cohort'], 'cohort_cells': len(cohort),
        'reference_model': str(reference), 'label_snapshot_count': len(labels),
        'training_label_count': len(rows), 'excluded_noncohort_labels': len(labels) - len(rows),
        'cohort_missing_intensity_cells': int((~np.isfinite(numeric[:, 3:]).all(axis=1)).sum()),
        'encoder': {key: reference_bundle[key] for key in ENCODER_KEYS},
        'pixel_correction': None,
        'pixel_correction_note': 'Prod2 correction is acquisition-specific; RHU2 uses native fused pixels.',
        'rois': spatial['rois'], 'spacing_by_roi': spacing,
    }
    (output / 'prepared.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Prepared {len(rows)} cortical labels from {len(labels)} annotations', flush=True)


def native_crop(row: pd.Series, fused: zarr.Array, masks: zarr.Array, keys: list[str], size: int) -> tuple[CellCrop, int]:
    x0, y0 = int(np.rint(row.x)) - size // 2, int(np.rint(row.y)) - size // 2
    x1, y1 = x0 + size, y0 + size
    xs, ys = slice(max(0, x0), min(masks.shape[2], x1)), slice(max(0, y0), min(masks.shape[1], y1))
    volume = np.asarray(masks[:, ys, xs])
    areas = (volume == row.label).sum(axis=(1, 2))
    if not areas.any():
        raise ValueError(f'No native segmentation at centroid for {row.cell}')
    plane = int(areas.argmax())
    padding = ((max(0, -y0), max(0, y1 - masks.shape[1])), (max(0, -x0), max(0, x1 - masks.shape[2])))
    labels = np.pad(volume[plane], padding)
    pixels = np.pad(np.asarray(fused[plane, ys, xs, :]), (*padding, (0, 0)))
    return CellCrop(labels, labels == row.label, (labels > 0) & (labels != row.label),
                    {name: pixels[..., i] for i, name in enumerate(keys)}), plane


def extract(output: Path) -> None:
    metadata = json.loads((output / 'prepared.json').read_text())
    config = metadata['encoder']
    rows = pd.read_parquet(output / 'training_rows.parquet')
    device = torch.device('cuda:0')
    torch.set_num_threads(4)
    model = load_cell_dino_model(repo=Path(config['dinov2_repo']), hub_model=config['feature_extractor'],
                                 weights=Path(config['weights']), device=device, precision=config['feature_precision'])
    transform = build_cell_transform(repo=Path(config['dinov2_repo']), resize_size=config['resize_size'],
                                     crop_size=config['crop_eval_size'])
    features = {target: np.empty((len(rows), 1024), np.float32) for target in TARGETS}
    planes = np.full(len(rows), -1, dtype=np.int64)
    rois = {item['key']: item for item in metadata['rois']}
    for (dataset, roi), group in rows.groupby(['dataset', 'roi'], sort=True):
        source = rois[f'{dataset}:{roi}']
        fused = zarr.open_array(source['fused'], mode='r')
        masks = zarr.open_array(source['segmentation'], mode='r')
        keys = list(fused.attrs['key'])
        if keys != source['channel_keys'] or set(keys) != {'brdu', 'edu', 'pi'} or masks.shape != fused.shape[:3]:
            raise ValueError(f'Native source contract changed for {dataset}:{roi}')

        def images(item: tuple[int, pd.Series]) -> tuple[int, dict[str, np.ndarray], int]:
            index, row = item
            crop, plane = native_crop(row, fused, masks, keys, config['crop_size'])
            tensors = cell_dino_images(crop, input_channels=config['input_channels'],
                                       channel_names_by_target=config['input_channel_names_by_target'],
                                       highpass_sigma_px=config['highpass_sigma_px'], mask_erosion_px=config['mask_erosion_px'])
            return index, tensors, plane

        with ThreadPoolExecutor(max_workers=4) as pool:
            iterator = pool.map(images, group.iterrows())
            batch = []
            for item in iterator:
                batch.append(item)
                if len(batch) == 32:
                    encode_batch(batch, features, planes, model, transform, device, config['feature_precision'])
                    batch = []
            if batch:
                encode_batch(batch, features, planes, model, transform, device, config['feature_precision'])
        print(f'Extracted {dataset}:{roi}: {len(group)} cells; total {(planes >= 0).sum()}/{len(rows)}', flush=True)
    if (planes < 0).any() or any(not np.isfinite(x).all() for x in features.values()):
        raise ValueError('Incomplete embedding extraction')
    np.savez_compressed(output / 'embeddings.npz', cell=rows.cell.to_numpy(dtype=str), plane_z=planes, **features)
    (output / 'embeddings_complete.json').write_text(json.dumps({'cells': len(rows), 'dimensions': 1024}) + '\n')


def encode_batch(batch, features, planes, model, transform, device, precision) -> None:
    indices = [item[0] for item in batch]
    planes[indices] = [item[2] for item in batch]
    for target in TARGETS:
        tensor = prepare_model_batch([item[1][target] for item in batch], transform=transform,
                                      device=device, dtype=feature_dtype(precision))
        features[target][indices] = extract_normalized_features(model, tensor, precision=precision)


def cortical_context(cohort: pd.DataFrame, cells: pd.Series, scale: np.ndarray) -> dict[str, np.ndarray]:
    """Map sparse segmentation identities to dense internal kNN rows explicitly."""
    points = cohort[['z', 'y', 'x']].to_numpy(dtype=float) * scale
    if len(cohort) < 17 or cohort.cell.duplicated().any():
        raise ValueError('Context requires 17 unique cortical cells per ROI')
    indices = pd.Index(cohort.cell).get_indexer(cells)
    if (indices < 0).any():
        raise ValueError('Training cells absent from cortical neighbor population')
    context = IntensityContext(points, {t: cohort[f'{t}_mean'].to_numpy(dtype=float) for t in TARGETS},
                               cKDTree(points), scale)
    relative = context.features_for_labels(indices + 1, neighbors=16)
    nuclear = np.log1p(cohort.pi_mean.to_numpy(dtype=float)[indices])
    features = {t: np.column_stack([relative[t], nuclear]) for t in TARGETS}
    if any(not np.isfinite(values).all() for values in features.values()):
        raise ValueError('Required training cell or neighbor intensity is missing')
    return features


def context_feature_indices(names: list[str]) -> list[int]:
    """Resolve the two supported classifier recipes in cortical_context column order."""
    if names not in [list(CONTEXT_FEATURES), list(CONTEXT_FEATURES[1:])]:
        raise ValueError(f'Unsupported RHU2 context features: {names}')
    return [CONTEXT_FEATURES.index(name) for name in names]


def offset_groups(rows: pd.DataFrame, identity: str) -> pd.DataFrame:
    """Resolve the recorded acquisition grouping independently of CCF branches."""
    if identity == 'dataset':
        return pd.DataFrame({'dataset': rows.dataset.astype(str)})
    if identity == 'roi_key':
        return pd.DataFrame({'roi_key': rows.dataset.astype(str) + ':' + rows.roi.astype(str)})
    if identity == 'dataset_roi':
        return pd.DataFrame({
            'dataset': 'dataset:' + rows.dataset.astype(str),
            'roi_key': 'roi:' + rows.dataset.astype(str) + ':' + rows.roi.astype(str),
        })
    raise ValueError(f'Unsupported offset identity: {identity}')


def section_features(datasets: pd.Series | pd.DataFrame, categories: list[str]) -> np.ndarray:
    """Encode additive intercepts in the model's recorded group order."""
    groups = pd.DataFrame(datasets).to_numpy()
    indices = pd.Index(categories).get_indexer(groups.ravel()).reshape(groups.shape)
    if (indices < 0).any():
        raise ValueError(f'Unknown section: {sorted(set(groups[indices < 0]))}')
    values = np.zeros((len(datasets), len(categories)), dtype=float)
    values[np.arange(len(datasets))[:, None], indices] = 1.
    return values


def fit(output: Path, seed: int, z_step_um: float | None, *, include_absolute_intensity: bool = True,
        section_offsets: Literal['none', 'brdu', 'both'] = 'none',
        section_identity: Literal['dataset', 'roi_key', 'dataset_roi'] = 'dataset') -> None:
    if section_offsets not in ['none', 'brdu', 'both']:
        raise ValueError(f'Unsupported section-offset targets: {section_offsets}')
    offset_targets = {'none': (), 'brdu': ('brdu',), 'both': TARGETS}[section_offsets]
    metadata = json.loads((output / 'prepared.json').read_text())
    completed = json.loads((output / 'embeddings_complete.json').read_text())
    rows = pd.read_parquet(output / 'training_rows.parquet')
    intensity_complete = json.loads((output / 'intensity_complete.json').read_text())
    if intensity_complete['status'] != 'complete':
        raise ValueError('Native intensity measurement is incomplete')
    cohort = pd.read_parquet(output / 'intensity_context.parquet')
    embeddings = np.load(output / 'embeddings.npz')
    if completed['cells'] != len(rows) or not np.array_equal(embeddings['cell'], rows.cell):
        raise ValueError('Embedding identities must exactly match training rows')
    context_features = list(CONTEXT_FEATURES if include_absolute_intensity else CONTEXT_FEATURES[1:])
    context_columns = context_feature_indices(context_features)
    contexts = {t: np.empty((len(rows), len(context_features)), dtype=float) for t in TARGETS}
    rois = {item['key']: item for item in metadata['rois']}
    if intensity_complete.get('spacing_by_roi') != metadata['spacing_by_roi']:
        raise ValueError('Native intensity measurement spacing disagrees with preparation')
    for (dataset, roi), group in rows.groupby(['dataset', 'roi'], sort=True):
        source = rois[f'{dataset}:{roi}']
        scale = np.asarray(metadata['spacing_by_roi'][source['key']]['scale_zyx_um'], dtype=float)
        if z_step_um is not None and (not np.isfinite(z_step_um) or z_step_um <= 0 or not np.isclose(scale[0], z_step_um)):
            raise ValueError(f'Invalid or conflicting Z spacing for {source["key"]}')
        local = cohort.loc[(cohort.dataset == dataset) & (cohort.roi == roi)]
        values = cortical_context(local, group.cell, scale)
        for target in TARGETS:
            contexts[target][group.index] = values[target][:, context_columns]
    train, validation = train_test_split(np.arange(len(rows)), test_size=192, random_state=seed,
                                         stratify=rows.brdu + 2 * rows.edu)
    train, validation = np.sort(train), np.sort(validation)
    sections = {target: np.empty((len(rows), 0), dtype=float) for target in TARGETS}
    section_metadata = {}
    if offset_targets:
        groups = offset_groups(rows, section_identity)
        categories = np.unique(groups.iloc[train].to_numpy()).tolist()
        # Every acquisition group to be scored must have training annotations.
        section_features(offset_groups(pd.DataFrame(metadata['rois']), section_identity), categories)
        encoded = section_features(groups, categories)
        for target in offset_targets:
            sections[target] = encoded
        section_metadata = {'schema': 'rhu2-dino-section-offset-v2', 'section_categories': categories,
                            'section_identity': section_identity, 'section_offset_targets': list(offset_targets)}
    bundle = {**metadata['encoder'], 'schema': metadata['schema'], 'context_features': context_features,
              'intensity_neighbors': 16,
              'neighbor_population': 'cortical cells within the same ROI', 'spacing_by_roi': metadata['spacing_by_roi'],
              'split_seed': seed, 'train_cells': rows.cell.iloc[train].tolist(),
              'validation_cells': rows.cell.iloc[validation].tolist(), 'pixel_correction': None,
              'intensity_context': str(output / 'intensity_context.parquet'),
              'intensity_measurement': intensity_complete,
              'provenance': str(output / 'prepared.json'), **section_metadata}
    metrics = {'train_cells': len(train), 'validation_cells': len(validation), 'split_seed': seed}
    predictions = rows[['cell', 'brdu', 'edu']].copy()
    predictions['split'] = 'train'
    predictions.loc[validation, 'split'] = 'validation'
    with threadpool_limits(limits=1):
        for target in TARGETS:
            x = np.column_stack([embeddings[target], contexts[target], sections[target]])
            y = rows[target].to_numpy(dtype=int)
            best_c, cv_score, folds = _best_logreg_c(x[train], y[train], cv_folds=5, seed=seed)
            scaler = StandardScaler().fit(x[train])
            model = LogisticRegression(solver='lbfgs', C=best_c, max_iter=1000, tol=1e-12)
            model.fit(scaler.transform(x[train]), y[train])
            probabilities = model.predict_proba(scaler.transform(x))[:, 1]
            cutoff, _ = balanced_accuracy_cutoff(y[validation], probabilities[validation])
            bundle.update({f'{target}_feature_scaler': scaler, f'{target}_model': model, f'{target}_cutoff': cutoff})
            predictions[f'{target}_prob'] = probabilities
            predictions[f'{target}_call'] = (probabilities >= cutoff).astype(int)
            metrics[target] = {'best_c': best_c, 'cv_folds': folds, 'cv_balanced_accuracy': cv_score,
                               'cutoff': cutoff, 'validation_auc': roc_auc_score(y[validation], probabilities[validation]),
                               'validation_balanced_accuracy': balanced_accuracy_score(y[validation], probabilities[validation] >= cutoff),
                               'validation_brier': brier_score_loss(y[validation], probabilities[validation])}
            print(f'{target}: {metrics[target]}', flush=True)
    np.savez_compressed(output / 'context_features.npz', cell=rows.cell.to_numpy(dtype=str), **contexts)
    predictions.to_parquet(output / 'labeled_predictions.parquet', index=False)
    joblib.dump(bundle, output / 'model.joblib')
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    # Verify the persisted classifiers reproduce the saved identity-aligned probabilities.
    restored = joblib.load(output / 'model.joblib')
    for target in TARGETS:
        x = np.column_stack([embeddings[target], contexts[target], sections[target]])
        actual = restored[f'{target}_model'].predict_proba(restored[f'{target}_feature_scaler'].transform(x))[:, 1]
        np.testing.assert_allclose(actual, predictions[f'{target}_prob'], rtol=0, atol=1e-12)
    (output / 'complete.json').write_text(json.dumps({'status': 'complete', 'cells': len(rows),
                                                     'classifier_features': {
                                                         target: bundle[f'{target}_model'].n_features_in_
                                                         for target in TARGETS}}) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'extract', 'fit'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--seed', type=int, default=20260928)
    parser.add_argument('--z-step-um', type=float)
    parser.add_argument('--exclude-absolute-intensity', action='store_true',
                        help='Omit absolute marker intensity; retain local relative intensity and PI intensity')
    parser.add_argument('--section-offsets', nargs='?', const='both', default='none', choices=['none', 'brdu', 'both'],
                        help='Fit regularized section intercepts for BrdU only or both markers')
    parser.add_argument('--section-identity', choices=['dataset', 'roi_key', 'dataset_roi'], default='dataset',
                        help='Offset grouping: dataset, acquisition ROI, or dataset plus nested acquisition ROI')
    args = parser.parse_args()
    if args.stage == 'prepare':
        if args.review is None or args.reference is None:
            parser.error('prepare requires --review and --reference')
        prepare(args.review, args.output, args.reference, args.z_step_um)
    elif args.stage == 'extract':
        extract(args.output)
    else:
        fit(args.output, args.seed, args.z_step_um, include_absolute_intensity=not args.exclude_absolute_intensity,
            section_offsets=args.section_offsets, section_identity=args.section_identity)


if __name__ == '__main__':
    main()
