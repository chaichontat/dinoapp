import json

import joblib
import numpy as np
import pandas as pd
import pytest

from dino.rhu2_predict import score
from dino.rhu2_train import cortical_context, fit, native_crop, offset_groups, section_features


def test_roi_offsets_share_ccf_branches_but_distinguish_acquisition_rois():
    rows = pd.DataFrame({'dataset': ['d', 'd', 'd', 'e'], 'roi': ['1', '1', '2', '1'],
                         'ccf_adjusted': ['cortex', 'cortex2', 'cortex', 'cortex']})
    groups = offset_groups(rows, 'roi_key')
    assert groups.roi_key.tolist() == ['d:1', 'd:1', 'd:2', 'e:1']
    np.testing.assert_array_equal(section_features(groups, ['d:1', 'd:2', 'e:1']),
                                  [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match='Unsupported offset identity'):
        offset_groups(rows, 'ccf_adjusted')


def test_nested_offsets_add_dataset_and_roi_terms_without_splitting_ccf_branches():
    rows = pd.DataFrame({'dataset': ['d', 'd', 'd', 'e'], 'roi': ['1', '1', '2', '1'],
                         'ccf_adjusted': ['cortex', 'cortex2', 'cortex', 'cortex']})
    groups = offset_groups(rows, 'dataset_roi')
    categories = ['dataset:d', 'dataset:e', 'roi:d:1', 'roi:d:2', 'roi:e:1']
    design = section_features(groups, categories)
    np.testing.assert_array_equal(design, [[1, 0, 1, 0, 0], [1, 0, 1, 0, 0],
                                          [1, 0, 0, 1, 0], [0, 1, 0, 0, 1]])
    np.testing.assert_allclose(design @ np.array([.4, -.2, .1, -.3, .6]), [.5, .5, .1, .4])
    with pytest.raises(ValueError, match='Unknown section'):
        section_features(groups, categories[:-1])


def test_section_features_use_saved_order_and_reject_unseen_sections():
    actual = section_features(pd.Series(['e', 'd', 'e']), ['d', 'e'])
    np.testing.assert_array_equal(actual, [[0, 1], [1, 0], [0, 1]])
    with pytest.raises(ValueError, match='Unknown section'):
        section_features(pd.Series(['unseen']), ['d', 'e'])


def test_context_uses_qualified_identity_and_excludes_self():
    table = pd.DataFrame({'cell': [f'roi|{i * 10 + 7}' for i in range(17)],
                          'z': np.arange(17), 'y': 0., 'x': 0.,
                          'brdu_mean': np.arange(17) * 2., 'edu_mean': 3., 'pi_mean': 9.})
    selected = pd.Series(['roi|167', 'roi|7'])
    actual = cortical_context(table, selected, np.array([.4, .216, .216]))
    np.testing.assert_allclose(actual['brdu'], [[np.log1p(32), np.log(33 / 16), np.log(10)],
                                               [0, np.log(1 / 18), np.log(10)]])
    np.testing.assert_allclose(actual['edu'][:, 1], 0)
    with pytest.raises(ValueError, match='absent'):
        cortical_context(table, pd.Series(['other|7']), np.ones(3))
    table.loc[1, 'edu_mean'] = np.nan
    with pytest.raises(ValueError, match='intensity is missing'):
        cortical_context(table, selected, np.ones(3))


def test_native_crop_uses_maximum_area_plane_and_channel_metadata():
    masks = np.zeros((3, 6, 6), dtype=int)
    masks[0, 0, 0] = 9
    masks[2, :3, :3] = 9
    raw = np.zeros((3, 6, 6, 3), dtype=np.uint16)
    raw[..., 0], raw[..., 1], raw[..., 2] = 100, 200, 300
    row = pd.Series({'cell': 'roi|9', 'label': 9, 'x': 1, 'y': 1})
    crop, plane = native_crop(row, raw, masks, ['edu', 'pi', 'brdu'], 4)
    assert plane == 2 and crop.mask.shape == (4, 4) and crop.mask.sum() == 9
    assert (crop.channels['brdu'][crop.mask] == 300).all()
    assert not crop.mask[0].any()


@pytest.mark.parametrize('include_absolute_intensity', [True, False])
@pytest.mark.parametrize('section_offsets,section_identity',
                         [('none', 'dataset'), ('brdu', 'dataset'), ('both', 'dataset'),
                          ('brdu', 'roi_key'), ('brdu', 'dataset_roi')])
def test_fit_persists_disjoint_split_and_reproducible_predictions(tmp_path, include_absolute_intensity,
                                                                section_offsets, section_identity):
    rng = np.random.default_rng(42)
    count = 256
    rows = pd.DataFrame({'cell': [f'd:r|{i}' for i in range(count)], 'dataset': 'd', 'roi': 'r',
                         'label': np.arange(count), 'brdu': np.arange(count) % 2, 'edu': np.arange(count) // 2 % 2,
                         'x': rng.uniform(0, 100, count), 'y': rng.uniform(0, 100, count),
                         'z': rng.uniform(0, 30, count), 'brdu_mean': rng.uniform(0, 500, count),
                         'edu_mean': rng.uniform(0, 500, count), 'pi_mean': 300.})
    rows.loc[count // 2:, 'dataset'] = 'e'
    rows.loc[rows.index % (count // 2) >= count // 4, 'roi'] = 's'
    rows['cell'] = rows.dataset + ':' + rows.roi + '|' + rows.label.astype(str)
    rows.to_parquet(tmp_path / 'training_rows.parquet')
    rows.to_parquet(tmp_path / 'intensity_context.parquet')
    metadata = {'encoder': {'input_channel_names_by_target': {'brdu': ['pi', 'brdu'], 'edu': ['pi', 'edu']}},
                'cohort_cells': count, 'schema': 'rhu2-dino-knn-intensity-v1', 'rois': [
        {'key': f'{dataset}:{roi}', 'dataset': dataset, 'roi': roi, 'cells': count // 4,
         'fused': str(tmp_path / 'acquisition/analysis/deconv/stitch/fused.zarr'),
         'pixel_size_um': .216, 'geometry_source': 'test'} for dataset in ['d', 'e'] for roi in ['r', 's']]}
    metadata['spacing_by_roi'] = {source['key']: {'scale_zyx_um': [1., .216, .216], 'z_source': 'test', 'xy_source': 'test'} for source in metadata['rois']}
    (tmp_path / 'intensity_complete.json').write_text(json.dumps({'status': 'complete', 'spacing_by_roi': metadata['spacing_by_roi']}))
    (tmp_path / 'prepared.json').write_text(json.dumps(metadata))
    (tmp_path / 'embeddings_complete.json').write_text(json.dumps({'cells': count}))
    np.savez_compressed(tmp_path / 'embeddings.npz', cell=rows.cell.to_numpy(dtype=str),
                        brdu=rng.normal(size=(count, 1024)), edu=rng.normal(size=(count, 1024)))
    fit(tmp_path, 42, 1., include_absolute_intensity=include_absolute_intensity,
        section_offsets=section_offsets, section_identity=section_identity)
    if include_absolute_intensity and section_offsets == 'none':
        with pytest.raises(ValueError, match='conflicting Z spacing'):
            fit(tmp_path, 42, .4)
    bundle = joblib.load(tmp_path / 'model.joblib')
    assert len(bundle['train_cells']) == 64 and len(bundle['validation_cells']) == 192
    assert set(bundle['train_cells']).isdisjoint(bundle['validation_cells'])
    saved = pd.read_parquet(tmp_path / 'labeled_predictions.parquet')
    assert saved.cell.equals(rows.cell)
    assert saved[['brdu_prob', 'edu_prob']].ge(0).all().all()
    assert saved[['brdu_prob', 'edu_prob']].le(1).all().all()
    assert json.loads((tmp_path / 'complete.json').read_text())['status'] == 'complete'
    names = ['log1p_cell_mean', 'log_cell_over_local_mean', 'log1p_pi_mean']
    expected_names = names if include_absolute_intensity else names[1:]
    assert bundle['context_features'] == expected_names
    contexts = np.load(tmp_path / 'context_features.npz')
    expected_context = {target: np.concatenate([
        cortical_context(group, group.cell, np.array([1., .216, .216]))[target]
        for _, group in rows.groupby(['dataset', 'roi'])
    ]) for target in ['brdu', 'edu']}
    for target in ['brdu', 'edu']:
        expected = expected_context[target] if include_absolute_intensity else expected_context[target][:, 1:]
        np.testing.assert_allclose(contexts[target], expected, rtol=0, atol=1e-12)
        has_offsets = section_offsets == 'both' or (section_offsets == 'brdu' and target == 'brdu')
        categories = {
            'dataset': ['d', 'e'],
            'roi_key': ['d:r', 'd:s', 'e:r', 'e:s'],
            'dataset_roi': ['dataset:d', 'dataset:e', 'roi:d:r', 'roi:d:s', 'roi:e:r', 'roi:e:s'],
        }[section_identity]
        n_groups = len(categories)
        assert bundle[f'{target}_model'].n_features_in_ == 1024 + len(expected_names) + n_groups * has_offsets
        if has_offsets:
            assert bundle['section_categories'] == categories
            assert bundle['section_identity'] == section_identity
            with np.load(tmp_path / 'embeddings.npz') as features:
                same_cell = np.tile(np.concatenate([features[target][0], contexts[target][0]]), (3, 1))
            group_rows = pd.DataFrame({'dataset': ['d', 'd', 'e'], 'roi': ['r', 's', 'r']})
            design = section_features(offset_groups(group_rows, section_identity), categories)
            x = np.column_stack([same_cell, design])
            scaler = bundle[f'{target}_feature_scaler']
            model = bundle[f'{target}_model']
            logits = model.decision_function(scaler.transform(x))
            offsets = model.coef_[0, -n_groups:] / scaler.scale_[-n_groups:]
            np.testing.assert_allclose(logits[1:] - logits[0], (design[1:] - design[0]) @ offsets, atol=1e-12)
    if section_offsets != 'none':
        assert bundle['section_offset_targets'] == (['brdu'] if section_offsets == 'brdu' else ['brdu', 'edu'])

    intensity = tmp_path / 'inference/intensity'
    intensity.mkdir(parents=True)
    (intensity / 'complete-0.json').write_text(json.dumps({'status': 'complete', 'shard_index': 0, 'shard_count': 1}))
    embeddings = tmp_path / 'cortical_embeddings'
    parts = []
    for (dataset, roi), group in rows.groupby(['dataset', 'roi']):
        name = f'{dataset}__{roi}'
        group.to_parquet(intensity / f'{name}.parquet', index=False)
        (embeddings / name).mkdir(parents=True)
        with np.load(tmp_path / 'embeddings.npz') as features:
            np.savez_compressed(embeddings / f'{name}/part.npz',
                                **{key: features[key][group.index][::-1] for key in features.files})
        parts.append({'path': f'{name}/part.npz', 'cells': len(group)})
    (embeddings / 'manifest.json').write_text(json.dumps({'encoder': metadata['encoder']}))
    (embeddings / 'complete.json').write_text(json.dumps({'status': 'complete', 'cells': count,
                                                        'partitions': parts}))
    score(tmp_path, 1)
    assert json.loads((tmp_path / "inference/complete.json").read_text())["status"] == "complete"
    predicted = pd.concat([pd.read_parquet(tmp_path / f'inference/predictions/{dataset}__{roi}.parquet')
                           for dataset in ['d', 'e'] for roi in ['r', 's']], ignore_index=True)
    np.testing.assert_array_equal(predicted.cell, saved.cell)
    np.testing.assert_allclose(predicted[['brdu_prob', 'edu_prob']], saved[['brdu_prob', 'edu_prob']], rtol=0, atol=1e-10)
    bundle['context_features'] = ['log1p_pi_mean']
    joblib.dump(bundle, tmp_path / 'model.joblib')
    with pytest.raises(ValueError, match='Unsupported RHU2 context features'):
        score(tmp_path, 1)


def test_rejects_unsupported_offset_recipe(tmp_path):
    with pytest.raises(ValueError, match='Unsupported section-offset targets'):
        fit(tmp_path, 42, .4, section_offsets='edu')


@pytest.mark.parametrize('precision', ['float32', 'bfloat16'])
def test_encode_batch_uses_recorded_precision(precision):
    import torch
    from dino.rhu2_train import encode_batch
    from dino.zarr_model import feature_dtype
    class Encoder(torch.nn.Module):
        def forward(self, batch):
            assert batch.dtype == feature_dtype(precision)
            return batch.flatten(1)
    images = {t: np.ones((1, 2, 2), dtype=np.float32) for t in ['brdu', 'edu']}
    features = {t: np.empty((1, 4), dtype=np.float32) for t in images}
    encode_batch([(0, images, 2)], features, np.empty(1), Encoder(), lambda x: x, torch.device('cpu'), precision)
    for values in features.values():
        np.testing.assert_allclose(values, .5)


def test_prepare_records_acquisition_spacing_and_rejects_conflicting_confirmation(tmp_path):
    import anndata as ad
    from dino.rhu2_train import ENCODER_KEYS, prepare
    review = tmp_path / 'review'
    review.mkdir()
    cell = 'd:r|1'
    pd.DataFrame({'cell': [cell], 'brdu': [0], 'edu': [1]}).to_csv(review / 'labels.csv', index=False)
    cohort = tmp_path / 'cohort.h5ad'
    ad.AnnData(obs=pd.DataFrame({'dataset': ['d'], 'roi': ['r'], 'z': [1.], 'y': [2.], 'x': [3.],
                                'brdu_mean': [10.], 'edu_mean': [20.], 'pi_mean': [30.]}, index=[cell])).write_h5ad(cohort)
    acquisition = tmp_path / 'acquisition'
    acquisition.mkdir()
    (acquisition / 'all.toml').write_text('[params]\nstep = 1.25\n')
    source = {'key': 'd:r', 'fused': str(acquisition / 'analysis/deconv/stitch/fused.zarr'),
              'pixel_size_um': .216, 'geometry_source': 'landmarks.json'}
    (review / 'spatial.json').write_text(json.dumps({'cohort': str(cohort), 'cells': 1, 'rois': [source]}))
    reference = tmp_path / 'reference.joblib'
    joblib.dump(dict.fromkeys(ENCODER_KEYS, 'fixture'), reference)
    prepare(review, tmp_path / 'run', reference, 1.25)
    meta = json.loads((tmp_path / 'run/prepared.json').read_text())
    assert meta['spacing_by_roi']['d:r']['scale_zyx_um'] == [1.25, .216, .216]
    assert meta['spacing_by_roi']['d:r']['z_source'] == str(acquisition / 'all.toml')
    with pytest.raises(ValueError, match='conflicting'):
        prepare(review, tmp_path / 'conflicting', reference, .4)
    assert not (tmp_path / 'conflicting/prepared.json').exists()


@pytest.mark.parametrize('step', [None, 0., -1., float('nan'), float('inf')])
def test_spacing_requires_positive_recorded_or_explicit_value(tmp_path, step):
    from dino.rhu2_train import resolve_spacing
    source = {'key': 'd:r', 'fused': str(tmp_path / 'a/b/c/fused.zarr'),
              'pixel_size_um': .216, 'geometry_source': 'test'}
    with pytest.raises(ValueError, match='spacing'):
        resolve_spacing([source], step)
    assert resolve_spacing([source], 1.)['d:r']['scale_zyx_um'] == [1., .216, .216]
