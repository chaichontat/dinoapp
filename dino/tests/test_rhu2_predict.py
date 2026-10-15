import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from dino.rhu2_predict import score
from dino.rhu2_train import cortical_context


def test_failed_rescoring_invalidates_completion_before_first_roi_write(tmp_path):
    rng = np.random.default_rng(124)
    n = 40
    rows = pd.DataFrame({'cell': [f'd:r|{i+1}' for i in range(n)], 'label': np.arange(1, n+1),
                         'dataset': 'd', 'roi': 'r', 'z': rng.uniform(0, 40, n),
                         'y': rng.uniform(0, 40, n), 'x': rng.uniform(0, 40, n),
                         'brdu_mean': np.arange(n)*30., 'edu_mean': np.arange(n)*10., 'pi_mean': 5.})
    intensity = tmp_path / 'inference/intensity'
    intensity.mkdir(parents=True)
    rows.to_parquet(intensity / 'd__r.parquet', index=False)
    (intensity / 'complete-0.json').write_text(json.dumps({'status': 'complete', 'shard_index': 0, 'shard_count': 1}))
    channels = {'brdu': ['pi', 'brdu'], 'edu': ['pi', 'edu']}
    bundle = {'schema': 'rhu2-dino-knn-intensity-v1', 'context_features':
              ['log1p_cell_mean', 'log_cell_over_local_mean', 'log1p_pi_mean'],
              'input_channel_names_by_target': channels,
              'spacing_by_roi': {'d:r': {'scale_zyx_um': [.4, .2, .2]}}}
    context = cortical_context(rows, rows.cell, np.array([.4, .2, .2]))
    embeddings = {t: rng.normal(size=(n, 2)) for t in ['brdu', 'edu']}
    audit = rows[['cell']].copy()
    for target in embeddings:
        x = np.column_stack([embeddings[target], context[target]])
        scaler = StandardScaler().fit(x)
        model = LogisticRegression().fit(scaler.transform(x), np.arange(n) % 2)
        bundle.update({f'{target}_feature_scaler': scaler, f'{target}_model': model, f'{target}_cutoff': .5})
        audit[f'{target}_prob'] = model.predict_proba(scaler.transform(x))[:, 1]
    joblib.dump(bundle, tmp_path / 'model.joblib')
    audit.to_parquet(tmp_path / 'labeled_predictions.parquet', index=False)
    rois = [
        {'dataset': 'd', 'roi': 'r', 'key': 'd:r', 'cells': n, 'pixel_size_um': .2},
        {'dataset': 'd', 'roi': 'missing', 'key': 'd:missing', 'cells': n, 'pixel_size_um': .2},
    ]
    (tmp_path / 'prepared.json').write_text(json.dumps({'cohort_cells': n * len(rois), 'rois': rois}))
    e = tmp_path / 'cortical_embeddings'
    (e / 'd__r').mkdir(parents=True)
    np.savez(e / 'd__r/part.npz', cell=rows.cell.to_numpy(dtype=str), **embeddings)
    (e / 'manifest.json').write_text(json.dumps({'encoder': {'input_channel_names_by_target': channels}}))
    (e / 'complete.json').write_text(json.dumps({'status': 'complete', 'cells': n,
                                               'partitions': [{'path': 'd__r/part.npz', 'cells': n}]}))
    completion = tmp_path / 'inference/complete.json'
    completion.write_text(json.dumps({'status': 'complete', 'cells': 80}))
    with pytest.raises(FileNotFoundError):
        score(tmp_path, 1)
    assert (tmp_path / 'inference/predictions/d__r.parquet').exists()
    assert json.loads(completion.read_text())['status'] != 'complete'
