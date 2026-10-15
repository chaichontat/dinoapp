"""Measure native full-mask intensities required by the RHU2 logistic fit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import zarr

from dino.intensity_context import IntensityContext
from dino.rhu2_embed import partition_rows, read_tile
from dino.zarr_source import write_json_atomic


def run(output: Path) -> None:
    metadata = json.loads((output / 'prepared.json').read_text())
    cohort = pd.read_parquet(output / 'cortical_context.parquet')
    training = pd.read_parquet(output / 'training_rows.parquet')
    release = ad.read_h5ad(metadata['cohort'], backed='r')
    cohort['area'] = release.obs.area.reindex(cohort.cell).to_numpy()
    release.file.close()
    means = ['brdu_mean', 'edu_mean', 'pi_mean']
    cohort[means] = np.nan
    rois = {row['key']: row for row in metadata['rois']}
    count = 0
    for (dataset, roi), group in cohort.groupby(['dataset', 'roi'], sort=True):
        source = rois[f'{dataset}:{roi}']
        scale = np.asarray(metadata['spacing_by_roi'][source['key']]['scale_zyx_um'], dtype=float)
        points = group[['z', 'y', 'x']].to_numpy(dtype=float) * scale
        context = IntensityContext(points, {}, cKDTree(points), scale)
        selected = training.loc[(training.dataset == dataset) & (training.roi == roi), 'cell']
        indices = pd.Index(group.cell).get_indexer(selected)
        if (indices < 0).any():
            raise ValueError('Training identity absent from cortical context')
        neighbors = context.neighbor_indices_for_labels(indices + 1, neighbors=16)
        needed = group.iloc[np.unique(np.concatenate([indices, neighbors.ravel()]))]
        fused = zarr.open_array(source['fused'], mode='r')
        masks = zarr.open_array(source['segmentation'], mode='r')
        keys = list(fused.attrs['key'])
        if keys != source['channel_keys'] or set(keys) != {'pi', 'brdu', 'edu'}:
            raise ValueError('Intensity source channel identity mismatch')
        for _, tile in partition_rows(needed):
            local, pixels, labels = read_tile(tile, fused, masks, 256)

            def measure(row):
                x, y = int(row.x), int(row.y)
                ys, xs = slice(max(0, y - 128), y + 128), slice(max(0, x - 128), x + 128)
                mask = labels[:, ys, xs] == row.label
                count = int(mask.sum())
                if count != int(row.area) or not count:
                    raise ValueError(f'Incomplete native cell mask for {row.cell}: {count} != {row.area}')
                return row.Index, [float(pixels[:, ys, xs, keys.index(key)][mask].mean())
                                   for key in ['brdu', 'edu', 'pi']]

            with ThreadPoolExecutor(max_workers=4) as pool:
                for index, values in pool.map(measure, local.itertuples()):
                    cohort.loc[index, means] = values
        count += len(needed)
        print(f'Measured {dataset}:{roi}: {len(needed)} required cells; total {count}', flush=True)
    temporary = output / 'intensity_context.partial.parquet'
    cohort.to_parquet(temporary, index=False)
    os.replace(temporary, output / 'intensity_context.parquet')
    write_json_atomic({'status': 'complete', 'measured_cells': count, 'cohort_cells': len(cohort),
                       'method': 'arithmetic mean of native fused pixels over the complete 3D cell mask',
                       'population': 'all training/validation cells and their 16 nearest cortical neighbors',
                       'spacing_by_roi': metadata['spacing_by_roi'],
                       'note': 'Unneeded rows have unmeasured intensities (NaN); coordinates retain the complete cohort.'},
                      output / 'intensity_complete.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
