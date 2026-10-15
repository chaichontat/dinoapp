"""Persist PI+marker Cell-DINO embeddings for every RHU2 cortical cell."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import zarr

from dino.rhu2_train import TARGETS, native_crop
from dino.zarr_model import (
    build_cell_transform, extract_normalized_features, feature_dtype, load_cell_dino_model,
    prepare_model_batch,
)
from dino.zarr_source import cell_dino_images, write_json_atomic

CHANNELS = {'brdu': ['pi', 'brdu'], 'edu': ['pi', 'edu']}
TILE_SIZE = 2048


def partition_rows(rows: pd.DataFrame):
    """Keep each cell in exactly one tile, with rounding identical to crop centering."""
    rows = rows.copy()
    rows['_tile_y'] = np.rint(rows.y).astype(np.int64) // TILE_SIZE
    rows['_tile_x'] = np.rint(rows.x).astype(np.int64) // TILE_SIZE
    yield from rows.groupby(['_tile_y', '_tile_x'], sort=True)


def read_tile(group: pd.DataFrame, fused, masks, size: int):
    """Read a bounded native volume shared by all centered crops in one tile."""
    half = size // 2
    cy, cx = np.rint(group.y).astype(int), np.rint(group.x).astype(int)
    y0, x0 = max(0, int(cy.min()) - half), max(0, int(cx.min()) - half)
    y1 = min(masks.shape[1], int(cy.max()) - half + size)
    x1 = min(masks.shape[2], int(cx.max()) - half + size)
    labels = np.asarray(masks[:, y0:y1, x0:x1])
    pixels = np.asarray(fused[:, y0:y1, x0:x1, :])
    local = group.copy()
    local['x'], local['y'] = cx - x0, cy - y0
    return local, pixels, labels


def check_partition(path: Path, group: pd.DataFrame) -> None:
    with np.load(path) as saved:
        if not np.array_equal(saved['cell'], group.cell.to_numpy(dtype=str)):
            raise ValueError(f'Partition identities changed: {path}')
        if saved['plane_z'].shape != (len(group),) or (saved['plane_z'] < 0).any():
            raise ValueError(f'Invalid crop planes: {path}')
        for target in TARGETS:
            if saved[target].shape != (len(group), 1024) or not np.isfinite(saved[target]).all():
                raise ValueError(f'Incomplete {target} embeddings: {path}')


def run(output: Path, batch_size: int, workers: int, max_partitions: int | None,
        shard_index: int, shard_count: int) -> None:
    prepared = json.loads((output / 'prepared.json').read_text())
    config = {**prepared['encoder'], 'input_channel_names_by_target': CHANNELS}
    rows = pd.read_parquet(output / 'cortical_context.parquet')
    root = output / 'cortical_embeddings'
    root.mkdir(exist_ok=True)
    manifest = {'artifact': 'rhu2-cortical-pi-marker-embeddings-v1', 'cohort': prepared['cohort'],
                'cells': len(rows), 'encoder': config, 'pixel_correction': None,
                'crop_plane': 'maximum target-mask area inside centered native crop; earliest plane breaks ties',
                'tile_size': TILE_SIZE, 'rois': prepared['rois']}
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Existing embedding run has a different source/encoder contract')
    else:
        write_json_atomic(manifest, manifest_path)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    model = load_cell_dino_model(repo=Path(config['dinov2_repo']), hub_model=config['feature_extractor'],
                                 weights=Path(config['weights']), device=device, precision=config['feature_precision'])
    transform = build_cell_transform(repo=Path(config['dinov2_repo']), resize_size=config['resize_size'],
                                     crop_size=config['crop_eval_size'])
    print(f'CUDA worker PID={os.getpid()} device={torch.cuda.get_device_name(device)}', flush=True)
    rois = {item['key']: item for item in prepared['rois']}
    count, written, ordinal = 0, 0, 0
    parts = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (dataset, roi), roi_rows in rows.groupby(['dataset', 'roi'], sort=True):
            source = rois[f'{dataset}:{roi}']
            fused = zarr.open_array(source['fused'], mode='r')
            masks = zarr.open_array(source['segmentation'], mode='r')
            keys = list(fused.attrs['key'])
            if keys != source['channel_keys'] or set(keys) != {'pi', 'brdu', 'edu'} or masks.shape != fused.shape[:3]:
                raise ValueError(f'Native source contract changed for {dataset}:{roi}')
            roi_dir = root / f'{dataset}__{roi}'
            roi_dir.mkdir(exist_ok=True)
            for (ty, tx), group in partition_rows(roi_rows):
                owned = ordinal % shard_count == shard_index
                ordinal += 1
                if not owned:
                    continue
                path = roi_dir / f'y{ty:04d}_x{tx:04d}.npz'
                if path.exists():
                    check_partition(path, group)
                else:
                    tick = time.monotonic()
                    local, pixels, labels = read_tile(group, fused, masks, config['crop_size'])
                    read_seconds = time.monotonic() - tick

                    def images(item):
                        crop, plane = native_crop(item, pixels, labels, keys, config['crop_size'])
                        tensors = cell_dino_images(crop, input_channels=config['input_channels'],
                                                   channel_names_by_target=CHANNELS,
                                                   highpass_sigma_px=config['highpass_sigma_px'],
                                                   mask_erosion_px=config['mask_erosion_px'])
                        return tensors, plane

                    features = {target: np.empty((len(group), 1024), np.float32) for target in TARGETS}
                    planes = np.empty(len(group), dtype=np.int16)
                    for offset in range(0, len(local), batch_size):
                        batch = list(pool.map(images, local.iloc[offset:offset + batch_size].itertuples(index=False)))
                        planes[offset:offset + len(batch)] = [item[1] for item in batch]
                        for target in TARGETS:
                            tensor = prepare_model_batch([item[0][target] for item in batch], transform=transform,
                                                          device=device, dtype=feature_dtype(config['feature_precision']))
                            features[target][offset:offset + len(batch)] = extract_normalized_features(
                                model, tensor, precision=config['feature_precision'])
                    temporary = path.with_suffix('.partial.npz')
                    np.savez_compressed(temporary, cell=group.cell.to_numpy(dtype=str), plane_z=planes, **features)
                    check_partition(temporary, group)
                    os.replace(temporary, path)
                    written += 1
                    print(f'{dataset}:{roi} tile={ty},{tx} cells={len(group)} read_s={read_seconds:.2f} '
                          f'total_s={time.monotonic() - tick:.2f} completed={count + len(group)}/{len(rows)}', flush=True)
                    del pixels, labels, features
                count += len(group)
                parts.append({'path': str(path.relative_to(root)), 'cells': len(group)})
                if max_partitions is not None and written >= max_partitions:
                    print(f'Bounded run finished after {written} new partitions; {count} cells verified', flush=True)
                    return
    if rows.cell.duplicated().any():
        raise ValueError('Cortical shard coverage failed')
    write_json_atomic({'status': 'complete', 'cells': count, 'partitions': parts,
                       'shard_index': shard_index, 'shard_count': shard_count,
                       'elapsed_seconds': time.monotonic() - started}, root / f'complete-shard-{shard_index}.json')
    print(f'Shard {shard_index}: {count} cortical cells embedded in {len(parts)} verified partitions', flush=True)


def finalize(output: Path, shard_count: int) -> None:
    root = output / 'cortical_embeddings'
    manifest = json.loads((root / 'manifest.json').read_text())
    parts = []
    for index in range(shard_count):
        shard = json.loads((root / f'complete-shard-{index}.json').read_text())
        if shard['status'] != 'complete' or shard['shard_index'] != index or shard['shard_count'] != shard_count:
            raise ValueError('Inconsistent cortical shard completion record')
        if sum(part['cells'] for part in shard['partitions']) != shard['cells']:
            raise ValueError('Cortical shard cell count mismatch')
        parts.extend(shard['partitions'])
    if len({part['path'] for part in parts}) != len(parts):
        raise ValueError('Overlapping cortical shard partitions')
    count = sum(part['cells'] for part in parts)
    if count != manifest['cells']:
        raise ValueError('Full cortical embedding coverage failed')
    write_json_atomic({'status': 'complete', 'cells': count, 'partitions': parts}, root / 'complete.json')


def collect_training(output: Path) -> None:
    """Select frozen annotation identities from the completed cortical embedding run."""
    root = output / 'cortical_embeddings'
    manifest = json.loads((root / 'manifest.json').read_text())
    complete = json.loads((root / 'complete.json').read_text())
    prepared = json.loads((output / 'prepared.json').read_text())
    rows = pd.read_parquet(output / 'training_rows.parquet')
    if complete['status'] != 'complete' or complete['cells'] != prepared['cohort_cells']:
        raise ValueError('Full cortical embedding run is not complete')
    index = pd.Index(rows.cell)
    features = {target: np.empty((len(rows), 1024), np.float32) for target in TARGETS}
    planes = np.full(len(rows), -1, dtype=np.int16)
    for part in complete['partitions']:
        with np.load(root / part['path']) as saved:
            selected = index.get_indexer(saved['cell'])
            keep = selected >= 0
            positions = selected[keep]
            if (planes[positions] >= 0).any():
                raise ValueError('Duplicated training identity across embedding partitions')
            planes[positions] = saved['plane_z'][keep]
            for target in TARGETS:
                features[target][positions] = saved[target][keep]
    if (planes < 0).any():
        raise ValueError('Training identities missing from cortical embeddings')
    np.savez_compressed(output / 'embeddings.npz', cell=rows.cell.to_numpy(dtype=str), plane_z=planes, **features)
    prepared['encoder'] = manifest['encoder']
    prepared['cortical_embeddings'] = str(root)
    write_json_atomic(prepared, output / 'prepared.json')
    write_json_atomic({'cells': len(rows), 'dimensions': 1024, 'source': str(root)}, output / 'embeddings_complete.json')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['embed', 'finalize', 'collect-training'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--max-partitions', type=int)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers <= 0 or (args.max_partitions is not None and args.max_partitions <= 0):
        parser.error('Batch size, workers and optional partition count must be positive')
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error('Shard index must be in [0, shard count)')
    if args.stage == 'embed':
        run(args.output, args.batch_size, args.workers, args.max_partitions, args.shard_index, args.shard_count)
    elif args.stage == 'finalize':
        finalize(args.output, args.shard_count)
    else:
        collect_training(args.output)


if __name__ == '__main__':
    main()
