"""Measure and score the complete cortical RHU2 cohort for reviewer activation."""
import argparse
import json
import os
from pathlib import Path
import time

import anndata as ad
import joblib
import numba
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import zarr

from dino.rhu2_embed import partition_rows, read_tile
from dino.rhu2_train import context_feature_indices, cortical_context, offset_groups, section_features, TARGETS
from dino.zarr_source import write_json_atomic


@numba.njit(cache=True)
def mask_sums(labels, pixels, wanted):
    """Accumulate only the requested labels in one pass over native voxel values."""
    lookup = np.full(wanted.max() + 1, -1, np.int64)
    for i in range(len(wanted)):
        lookup[wanted[i]] = i
    counts = np.zeros(len(wanted), np.int64)
    sums = np.zeros((len(wanted), pixels.shape[3]), np.float64)
    for z in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            for x in range(labels.shape[2]):
                label = labels[z, y, x]
                if label < len(lookup):
                    row = lookup[label]
                    if row >= 0:
                        counts[row] += 1
                        for channel in range(pixels.shape[3]):
                            sums[row, channel] += pixels[z, y, x, channel]
    return counts, sums


def measure(output: Path, shard: int, shards: int) -> None:
    meta = json.loads((output / 'prepared.json').read_text())
    cohort = pd.read_parquet(output / 'cortical_context.parquet')
    release = ad.read_h5ad(meta['cohort'], backed='r')
    cohort['area'] = release.obs.area.reindex(cohort.cell).to_numpy()
    release.file.close()
    known = pd.read_parquet(output / 'intensity_context.parquet').set_index('cell')
    root = output / 'inference/intensity'
    root.mkdir(parents=True, exist_ok=True)
    sources = {r['key']: r for r in meta['rois']}
    for ordinal, ((dataset, roi), group) in enumerate(cohort.groupby(['dataset', 'roi'], sort=True)):
        if ordinal % shards != shard:
            continue
        path = root / f'{dataset}__{roi}.parquet'
        if path.exists():
            saved = pd.read_parquet(path)
            if not saved.cell.equals(group.cell.reset_index(drop=True)) or not np.isfinite(saved[['brdu_mean', 'edu_mean', 'pi_mean']]).all().all():
                raise ValueError(f'Invalid saved intensity ROI: {path}')
            continue
        source = sources[f'{dataset}:{roi}']
        fused = zarr.open_array(source['fused'], mode='r')
        masks = zarr.open_array(source['segmentation'], mode='r')
        keys = list(fused.attrs['key'])
        if keys != source['channel_keys'] or set(keys) != {'pi', 'brdu', 'edu'} or masks.shape != fused.shape[:3]:
            raise ValueError(f'Native source contract changed: {source["key"]}')
        result = group.copy()
        result[['brdu_mean', 'edu_mean', 'pi_mean']] = np.nan
        start = time.monotonic()
        for (ty, tx), tile in partition_rows(group):
            _, pixels, labels = read_tile(tile, fused, masks, 256)
            counts, sums = mask_sums(labels, pixels, tile.label.to_numpy(dtype=np.int64))
            if not np.array_equal(counts, tile.area.to_numpy(dtype=np.int64)) or (counts == 0).any():
                bad = tile.loc[counts != tile.area.to_numpy(dtype=np.int64), 'cell'].tolist()
                raise ValueError(f'Incomplete full-cell mask in {source["key"]} tile {ty},{tx}: {bad[:5]}')
            for channel in ['brdu', 'edu', 'pi']:
                result.loc[tile.index, f'{channel}_mean'] = sums[:, keys.index(channel)] / counts
            del pixels, labels
        prior = known.loc[result.cell, ['brdu_mean', 'edu_mean', 'pi_mean']]
        selected = prior.notna().all(axis=1).to_numpy()
        np.testing.assert_allclose(result.loc[selected, prior.columns].to_numpy(), prior.loc[selected].to_numpy(), rtol=0, atol=1e-10)
        temporary = path.with_suffix('.partial.parquet')
        result.to_parquet(temporary, index=False)
        os.replace(temporary, path)
        print(f'Measured {source["key"]} cells={len(result)} seconds={time.monotonic() - start:.1f}', flush=True)
    write_json_atomic({'status': 'complete', 'shard_index': shard, 'shard_count': shards}, root / f'complete-{shard}.json')


def score(output: Path, shards: int) -> None:
    for i in range(shards):
        complete = json.loads((output / f'inference/intensity/complete-{i}.json').read_text())
        if complete != {'status': 'complete', 'shard_index': i, 'shard_count': shards}:
            raise ValueError('Intensity shard completion contract failed')
    bundle = joblib.load(output / 'model.joblib')
    if bundle['schema'] == 'rhu2-dino-knn-intensity-v1':
        offset_targets = ()
    elif bundle['schema'] == 'rhu2-dino-section-offset-v1':
        offset_targets = TARGETS
    elif bundle['schema'] == 'rhu2-dino-section-offset-v2':
        offset_targets = tuple(bundle['section_offset_targets'])
        if offset_targets not in [('brdu',), TARGETS]:
            raise ValueError(f'Unsupported section-offset targets: {offset_targets}')
    else:
        raise ValueError(f'Unsupported RHU2 model schema: {bundle["schema"]}')
    context_columns = context_feature_indices(bundle['context_features'])
    metadata = json.loads((output / 'prepared.json').read_text())
    embeddings = output / 'cortical_embeddings'
    complete = json.loads((embeddings / 'complete.json').read_text())
    manifest = json.loads((embeddings / 'manifest.json').read_text())
    if manifest['encoder']['input_channel_names_by_target'] != bundle['input_channel_names_by_target']:
        raise ValueError('Classifier/embedding input channels disagree')
    write_json_atomic({'status': 'incomplete'}, output / 'inference/complete.json')
    root = output / 'inference/predictions'
    root.mkdir(exist_ok=True)
    count = 0
    audit = pd.read_parquet(output / 'labeled_predictions.parquet').set_index('cell')
    with threadpool_limits(limits=1):
        for source in metadata['rois']:
            name = f'{source["dataset"]}__{source["roi"]}'
            rows = pd.read_parquet(output / f'inference/intensity/{name}.parquet')
            if len(rows) != source['cells'] or rows.cell.duplicated().any():
                raise ValueError(f'Intensity identity/count mismatch: {name}')
            context = cortical_context(rows, rows.cell, np.asarray(bundle['spacing_by_roi'][source['key']]['scale_zyx_um'], dtype=float))
            sections = {target: np.empty((len(rows), 0)) for target in TARGETS}
            if offset_targets:
                encoded = section_features(offset_groups(rows, bundle['section_identity']),
                                           bundle['section_categories'])
                for target in offset_targets:
                    sections[target] = encoded
            predictions = rows[['cell', 'dataset', 'roi', 'x', 'y', 'z', 'label']].copy()
            for target in TARGETS:
                predictions[f'{target}_prob'] = np.nan
            seen = np.zeros(len(rows), dtype=bool)
            index = pd.Index(rows.cell)
            parts = [p for p in complete['partitions'] if Path(p['path']).parent.name == name]
            for part in parts:
                with np.load(embeddings / part['path']) as saved:
                    indices = index.get_indexer(saved['cell'])
                    if (indices < 0).any() or seen[indices].any():
                        raise ValueError(f'Embedding partition identity mismatch: {part["path"]}')
                    for target in TARGETS:
                        x = np.column_stack([saved[target], context[target][indices][:, context_columns], sections[target][indices]])
                        probability = bundle[f'{target}_model'].predict_proba(bundle[f'{target}_feature_scaler'].transform(x))[:, 1]
                        predictions.loc[indices, f'{target}_prob'] = probability
                    seen[indices] = True
            probs = predictions[['brdu_prob', 'edu_prob']]
            if not seen.all() or not np.isfinite(probs).all().all() or not ((probs >= 0) & (probs <= 1)).all().all():
                raise ValueError(f'Incomplete cortical predictions: {name}')
            check = predictions.set_index('cell').loc[predictions.cell.isin(audit.index).to_numpy(), probs.columns]
            np.testing.assert_allclose(check.to_numpy(), audit.loc[check.index, probs.columns].to_numpy(), rtol=0, atol=1e-10)
            path = root / f'{name}.parquet'
            temporary = path.with_suffix('.partial.parquet')
            predictions.to_parquet(temporary, index=False)
            os.replace(temporary, path)
            count += len(predictions)
            print(f'Scored {source["key"]} cells={len(predictions)} total={count}', flush=True)
    if count != metadata['cohort_cells']:
        raise ValueError('Full cortical prediction coverage failed')
    write_json_atomic({'status': 'complete', 'cells': count, 'model': str(output / 'model.joblib'),
                       'decision_thresholds': {t: bundle[f'{t}_cutoff'] for t in TARGETS}},
                      output / 'inference/complete.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['measure', 'score'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=2)
    args = parser.parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error('Invalid shard index/count')
    if args.stage == 'measure':
        measure(args.output, args.shard_index, args.shard_count)
    else:
        score(args.output, args.shard_count)
