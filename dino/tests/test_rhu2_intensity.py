import json

import anndata as ad
import numpy as np
import pandas as pd
import zarr
from scipy.spatial import cKDTree

from dino.rhu2_intensity import run
from dino.rhu2_predict import measure


def test_native_measurement_covers_exact_required_neighbors(tmp_path):
    count = 20
    cells = [f'd:r|{7 + 3 * i}' for i in range(count)]
    cohort = pd.DataFrame({'cell': cells, 'dataset': 'd', 'roi': 'r', 'label': 7 + 3 * np.arange(count),
                           'x': 2 + 4 * np.arange(count), 'y': 2., 'z': 1.,
                           'brdu_mean': 999., 'edu_mean': 999., 'pi_mean': 999.})
    cohort.loc[1:16, 'z'] = 2.
    cohort.loc[17:, 'z'] = 0.
    cohort.loc[0, 'z'] = 0.
    cohort.to_parquet(tmp_path / 'cortical_context.parquet')
    cohort.iloc[[0]].to_parquet(tmp_path / 'training_rows.parquet')
    release = tmp_path / 'cohort.h5ad'
    ad.AnnData(obs=pd.DataFrame({'area': np.ones(count)}, index=cells)).write_h5ad(release)
    labels = zarr.open_array(tmp_path / 'labels.zarr', mode='w', shape=(3, 5, 85), dtype='uint32')
    pixels = zarr.open_array(tmp_path / 'fused.zarr', mode='w', shape=(3, 5, 85, 3), dtype='uint16')
    pixels.attrs['key'] = ['edu', 'pi', 'brdu']
    for i, row in enumerate(cohort.itertuples()):
        labels[int(row.z), 2, row.x] = row.label
        pixels[int(row.z), 2, row.x] = [10 + i, 20 + i, 30 + i]
    metadata = {'spacing_by_roi': {'d:r': {'scale_zyx_um': [10., .216, .216], 'z_source': 'test', 'xy_source': 'test'}}, 'cohort': str(release), 'rois': [{'key': 'd:r', 'dataset': 'd', 'roi': 'r', 'cells': count, 'fused': str(tmp_path / 'fused.zarr'),
                'segmentation': str(tmp_path / 'labels.zarr'), 'channel_keys': ['edu', 'pi', 'brdu'],
                'pixel_size_um': .216}]}
    (tmp_path / 'prepared.json').write_text(json.dumps(metadata))
    run(tmp_path)
    measured = pd.read_parquet(tmp_path / 'intensity_context.parquet')
    points = cohort[['z', 'y', 'x']].to_numpy()
    selected = np.sort(cKDTree(points * [10., .216, .216]).query(points[0] * [10., .216, .216], k=17)[1])
    old = np.sort(cKDTree(points * [.4, .216, .216]).query(points[0] * [.4, .216, .216], k=17)[1])
    assert not np.array_equal(selected, old)
    np.testing.assert_array_equal(measured.brdu_mean.iloc[selected], 30 + selected)
    np.testing.assert_array_equal(measured.edu_mean.iloc[selected], 10 + selected)
    np.testing.assert_array_equal(measured.pi_mean.iloc[selected], 20 + selected)
    assert measured.brdu_mean.drop(index=selected).isna().all()
    assert json.loads((tmp_path / 'intensity_complete.json').read_text())['measured_cells'] == 17
    measure(tmp_path, 0, 1)
    full = pd.read_parquet(tmp_path / 'inference/intensity/d__r.parquet')
    np.testing.assert_array_equal(full.brdu_mean, 30 + np.arange(count))
    np.testing.assert_array_equal(full.edu_mean, 10 + np.arange(count))
    np.testing.assert_array_equal(full.pi_mean, 20 + np.arange(count))
