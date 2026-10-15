"""Snapshot RHU2 review coordinates and existing probabilities, without image copies."""
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from dino_markers import DEFAULT_CONTRACT, SPATIAL_REVIEW, rhu2_probability_column

from dino.review_contracts import combined_uncertainty


def prepare_branches(review_root: Path, release: Path, rows: pd.DataFrame) -> dict:
    """Persist cortical branch membership and the complete release branch catalog."""
    membership = rows[['cell', 'dataset', 'roi', 'ccf_adjusted']].copy()
    if membership.cell.duplicated().any() or membership.ccf_adjusted.isna().any():
        raise ValueError('Branch membership requires unique cells and recorded ccf_adjusted values')
    membership['roi_key'] = membership.dataset.astype(str) + ':' + membership.roi.astype(str)
    counts = membership.groupby(['roi_key', 'ccf_adjusted'], observed=True).size()
    release_manifest = json.loads((release / 'manifest.json').read_text())
    branches = []
    for roi in release_manifest['rois']:
        roi_key = f'{roi["dataset"]}:{roi["roi"]}'
        for branch in roi['branches']:
            name = branch['ccf_adjusted']
            branches.append(dict(key=f'{roi_key}:{name}', roi_key=roi_key,
                                 dataset=roi['dataset'], roi=roi['roi'], ccf_adjusted=name,
                                 cells=int(counts.get((roi_key, name), 0))))
    keys = [branch['key'] for branch in branches]
    if len(set(keys)) != len(keys) or sum(branch['cells'] for branch in branches) != len(rows):
        raise ValueError('Release branch catalog does not match cortical membership')
    relative = 'ccf_branches.parquet'
    temporary = review_root / 'ccf_branches.partial.parquet'
    membership[['cell', 'roi_key', 'ccf_adjusted']].sort_values('cell').to_parquet(temporary, index=False)
    temporary.replace(review_root / relative)
    return {'branch_table': relative, 'branches': sorted(branches, key=lambda branch: branch['key'])}


def prepare(review_root: Path, release: Path) -> None:
    sources = json.loads((review_root / 'complete.json').read_text())['sources']
    target = review_root / 'spatial'
    if target.exists() or (review_root / 'spatial.json').exists():
        raise FileExistsError(f'Spatial snapshot already exists: {review_root}')
    cohort = release / 'cohort/cells.h5ad'
    probability_columns = [rhu2_probability_column(marker) for marker in DEFAULT_CONTRACT.markers]
    data = ad.read_h5ad(cohort, backed='r')
    try:
        rows = data.obs[['dataset', 'roi', 'ccf_adjusted', 'x', 'y', 'z', *probability_columns]].copy()
    finally:
        data.file.close()
    rows.index.name = 'cell'
    rows = rows.reset_index()
    rows['cell'] = rows.cell.astype(str)
    rows['label'] = rows.cell.str.rsplit('|', n=1).str[-1].astype('int64')
    if rows.cell.duplicated().any():
        raise ValueError('Duplicate RHU2 cell identifiers')
    if not np.isfinite(rows[['x', 'y', 'z', *probability_columns]].to_numpy()).all():
        raise ValueError('RHU2 coordinates and probabilities must be finite')
    if not rows[probability_columns].apply(lambda x: x.between(0, 1)).all().all():
        raise ValueError('RHU2 probabilities must be in [0, 1]')
    metadata = []
    groups = []
    for (dataset, roi), group in rows.groupby(['dataset', 'roi'], observed=True, sort=True):
        key = f'{dataset}:{roi}'
        source = sources[key]
        roi_dir = release / 'datasets' / dataset / 'rois' / roi
        landmarks = roi_dir / 'ccf/landmarks.json'
        pixel_size = float(json.loads(landmarks.read_text())['sample_voxel_xy_um'])
        fused = json.loads((Path(source['fused']) / 'zarr.json').read_text())
        segmentation = json.loads((Path(source['segmentation']) / 'zarr.json').read_text())
        shape = segmentation['shape']
        keys = fused['attributes']['key']
        if fused['shape'][:3] != shape or set(keys) != set(DEFAULT_CONTRACT.channel_keys):
            raise ValueError(f'Invalid channel/segmentation geometry: {key}')
        if pixel_size <= 0 or not np.isfinite(pixel_size):
            raise ValueError(f'Invalid physical XY spacing: {key}')
        if not group.cell.equals(key + '|' + group.label.astype(str)):
            raise ValueError(f'Cell identities disagree with ROI: {key}')
        coords = group[['z', 'y', 'x']].to_numpy()
        if not ((coords >= 0) & (coords < np.array(shape))).all():
            raise ValueError(f'Coordinates outside native image: {key}')
        relative = f'spatial/{dataset}__{roi}.parquet'
        metadata.append(dict(key=key, dataset=dataset, roi=roi, cells=len(group),
                             table=relative, fused=source['fused'],
                             segmentation=source['segmentation'], shape=shape,
                             channel_keys=keys, pixel_size_um=pixel_size,
                             geometry_source=str(landmarks)))
        groups.append(group.sort_values('cell', kind='stable').reset_index(drop=True))
    target.mkdir()
    for meta, group in zip(metadata, groups, strict=True):
        group.to_parquet(review_root / meta['table'], index=False)
        pd.testing.assert_frame_equal(group, pd.read_parquet(review_root / meta['table']))
    thresholds = dict.fromkeys(DEFAULT_CONTRACT.markers, 0.5)
    uncertainty = combined_uncertainty(
        {marker: rows[column].to_numpy() for marker, column in zip(DEFAULT_CONTRACT.markers, probability_columns)},
        thresholds,
    )
    log_vmax = float(np.quantile(np.log1p(uncertainty[np.isfinite(uncertainty)]), 0.999))
    manifest = SPATIAL_REVIEW.stamp(
        dict(uncertainty_log_vmax=log_vmax, cohort=str(cohort),
             cells=len(rows), probability_columns=probability_columns,
             decision_thresholds=thresholds,
             sampling='uniform ROI, then uniform cell, with SeedSequence([seed, position])',
             z_coordinates='native image plane; no unrecorded physical Z spacing assumed',
             display_max=32768.0, rois=metadata, **prepare_branches(review_root, release, rows)),
        DEFAULT_CONTRACT,
    )
    (review_root / 'spatial.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Snapshotted {len(rows):,} cells in {len(metadata)} ROIs; probabilities unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-root', required=True, type=Path)
    parser.add_argument('--release', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.review_root, args.release)
