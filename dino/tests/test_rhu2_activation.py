import json

import pandas as pd

from dino.review_activation import activate


def test_activation_switches_complete_tables_and_preserves_labels(tmp_path):
    model, review = tmp_path / 'model', tmp_path / 'review'
    (model / 'inference/predictions').mkdir(parents=True)
    (review / 'spatial').mkdir(parents=True)
    rows = pd.DataFrame({'cell': ['d:r|7', 'd:r|9'], 'label': [7, 9], 'x': [1., 2.], 'y': [3., 4.],
                         'z': [0., 1.], 'brdu_prob': [.1, .9], 'edu_prob': [.8, .2]})
    rows.to_parquet(review / 'spatial/d__r.parquet', index=False)
    changed = rows.copy()
    changed.brdu_prob = [.7, .3]
    changed.to_parquet(model / 'inference/predictions/d__r.parquet', index=False)
    metadata = {'cells': 2, 'rois': [{'dataset': 'd', 'roi': 'r', 'cells': 2, 'table': 'spatial/d__r.parquet'}]}
    (review / 'spatial.json').write_text(json.dumps(metadata))
    original = (review / 'spatial.json').read_bytes()
    labels = b'cell,brdu,edu\nd:r|7,1,0\n'
    (review / 'labels.csv').write_bytes(labels)
    (model / 'inference/complete.json').write_text(json.dumps(
        {'status': 'complete', 'cells': 2, 'model': str(model / 'model.joblib'),
         'decision_thresholds': {'brdu': .45, 'edu': .13}}))
    activate(model, review)
    active = json.loads((review / 'spatial.json').read_text())
    assert active['decision_thresholds'] == {'brdu': .45, 'edu': .13}
    assert (review / 'labels.csv').read_bytes() == labels
    assert (review / 'spatial.before-model-v1.json').read_bytes() == original
    pd.testing.assert_frame_equal(pd.read_parquet(review / active['rois'][0]['table']), changed)


def test_failed_second_activation_preserves_every_active_table(tmp_path):
    import pytest

    model, review = tmp_path / 'model', tmp_path / 'review'
    (model / 'inference/predictions').mkdir(parents=True)
    (review / 'spatial-model-v1').mkdir(parents=True)
    rois = []
    for roi in ('a', 'b'):
        rows = pd.DataFrame({'cell': [f'd:{roi}|1'], 'label': [1], 'x': [1.], 'y': [2.],
                             'z': [0.], 'brdu_prob': [.1], 'edu_prob': [.2]})
        table = f'spatial-model-v1/d__{roi}.parquet'
        rows.to_parquet(review / table, index=False)
        rows['brdu_prob'] = .9
        if roi == 'b':
            rows['cell'] = 'd:b|99'
        rows.to_parquet(model / f'inference/predictions/d__{roi}.parquet', index=False)
        rois.append({'dataset': 'd', 'roi': roi, 'cells': 1, 'table': table})
    (review / 'spatial.json').write_text(json.dumps({'cells': 2, 'rois': rois}))
    (review / 'labels.csv').write_text('cell,brdu,edu\n')
    (model / 'inference/complete.json').write_text(json.dumps(
        {'status': 'complete', 'cells': 2, 'model': str(model / 'model.joblib'),
         'decision_thresholds': {'brdu': .5, 'edu': .5}}))
    paths = [review / 'spatial.json', *(review / source['table'] for source in rois)]
    before = {path: path.read_bytes() for path in paths}
    with pytest.raises(AssertionError):
        activate(model, review)
    assert {path: path.read_bytes() for path in paths} == before
