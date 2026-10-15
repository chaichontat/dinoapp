import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from dino.artifacts import ordered_cell_sha256, write_embedding_atomic
from dino.joint_train import collapse, design_matrix, fit_and_score, labeled_embeddings
from dino_markers import DEFAULT_CONTRACT, ContractMismatchError, MarkerContract


def synthetic(rng, rows=300, width=4, context=3, datasets=2):
    embedding = rng.normal(size=(rows, width))
    dataset = rng.integers(0, datasets, size=rows)
    brightness = rng.normal(size=(rows, context)) + 2.0 * dataset[:, None]
    logit = embedding[:, 0] + np.where(dataset == 1, -1.5 * brightness[:, 0], brightness[:, 0]) + dataset
    labels = (rng.random(rows) < 1 / (1 + np.exp(-logit))).astype(int)
    return embedding, brightness, dataset, labels


def test_design_matrix_masks_context_and_encodes_intercepts():
    embedding = np.arange(6.).reshape(3, 2)
    context = np.array([[1., 2.], [3., 4.], [5., 6.]])
    x = design_matrix(embedding, context, np.array([0, 1, 1]), 2)
    np.testing.assert_array_equal(x, [
        [0., 1., 1., 2., 0., 0., 1., 0.],
        [2., 3., 0., 0., 3., 4., 0., 1.],
        [4., 5., 0., 0., 5., 6., 0., 1.],
    ])


def test_collapsed_classifier_reproduces_joint_probabilities():
    rng = np.random.default_rng(0)
    embedding, brightness, dataset, labels = synthetic(rng)
    x = design_matrix(embedding, brightness, dataset, 2)
    scaler, model, joint_probability = fit_and_score(x, labels, 0.3, x)
    for index in range(2):
        reduced_scaler, reduced_model = collapse(scaler, model, index=index, datasets=2,
                                                 embedding_width=4, context_width=3)
        members = dataset == index
        reduced = reduced_model.predict_proba(reduced_scaler.transform(
            np.column_stack([embedding[members], brightness[members]])))[:, 1]
        assert reduced_model.n_features_in_ == 7
        np.testing.assert_allclose(reduced, joint_probability[members], rtol=0, atol=1e-12)


def test_brightness_slopes_differ_between_datasets():
    rng = np.random.default_rng(1)
    embedding, brightness, dataset, labels = synthetic(rng, rows=4000)
    x = design_matrix(embedding, brightness, dataset, 2)
    scaler, model, _ = fit_and_score(x, labels, 10., x)
    slopes = [collapse(scaler, model, index=d, datasets=2, embedding_width=4, context_width=3)
              for d in (0, 1)]
    first = [m.coef_[0, 4] / s.scale_[4] for s, m in slopes]
    assert first[0] > 0.5 and first[1] < -0.9


def test_collapse_rejects_mismatched_layout():
    rng = np.random.default_rng(2)
    embedding, brightness, dataset, labels = synthetic(rng)
    x = design_matrix(embedding, brightness, dataset, 2)
    scaler, model, _ = fit_and_score(x, labels, 1., x)
    with pytest.raises(ValueError, match='features'):
        collapse(scaler, model, index=0, datasets=3, embedding_width=4, context_width=3)


def _embedding_run(tmp_path, partitions):
    for shard, cells_by_part in enumerate(partitions):
        shard_dir = tmp_path / f'shard-{shard:03d}'
        (shard_dir / 'predictions').mkdir(parents=True)
        (shard_dir / 'complete.json').write_text(
            json.dumps({'artifact': 'dino-zarr-infer-v2', 'stores_embeddings': True, 'model_sha256': 'm',
                        'source_sha256': 's'}))
        for part, cells in enumerate(cells_by_part):
            pd.DataFrame({'cell': cells, 'label': [int(c.rsplit('|', 1)[1]) for c in cells]}).to_parquet(
                shard_dir / 'predictions' / f'part-{part:06d}.parquet')
            values = torch.tensor([[float(c.rsplit('|', 1)[1])] * 4 for c in cells]).to(torch.bfloat16)
            write_embedding_atomic({'brdu': values, 'edu': -values, 'cell_sha256': ordered_cell_sha256(cells)},
                                   shard_dir / 'embeddings' / f'part-{part:06d}.pt')
    return tmp_path


def test_labeled_embeddings_returns_requested_order_across_partitions(tmp_path):
    run = _embedding_run(tmp_path, [[['r|1', 'r|2'], ['r|3']], [['r|4', 'r|5']]])
    labels, values = labeled_embeddings(['r|5', 'r|1', 'r|3'], run, model_sha256='m', source_sha256='s', workers=2,
                                      contract=DEFAULT_CONTRACT)
    np.testing.assert_array_equal(labels, [5, 1, 3])
    np.testing.assert_array_equal(values['brdu'][:, 0], [5., 1., 3.])
    np.testing.assert_array_equal(values['edu'][:, 0], [-5., -1., -3.])


def test_labeled_embeddings_rejects_duplicate_cells(tmp_path):
    run = _embedding_run(tmp_path, [[['r|1', 'r|2']], [['r|2']]])
    with pytest.raises(ValueError, match='duplicate cell r\\|2'):
        labeled_embeddings(['r|2'], run, model_sha256='m', source_sha256='s', workers=2, contract=DEFAULT_CONTRACT)


def test_encoder_contract_rejects_different_backbone_source_before_fitting(tmp_path):
    import joblib
    from dino.artifacts import file_sha256
    from dino.joint_train import ENCODER_KEYS, fit

    datasets = []
    for name in ['first', 'second']:
        root = tmp_path / name
        root.mkdir()
        cells = [f'r|{i}' for i in range(1, 21)]
        pd.DataFrame({'cell': cells, 'brdu': np.arange(20) % 2,
                      'edu': np.arange(20) // 2 % 2}).to_csv(root / 'labels.csv', index=False)
        bundle = {**dict.fromkeys(ENCODER_KEYS, 'same'), 'schema': 'dino-zarr-model-v6', 'source_sha256': 's',
                  'model_source_sha256': name}
        model_path = root / 'model.joblib'
        joblib.dump(bundle, model_path)
        run = _embedding_run(root / 'run', [[cells]])
        (run / 'shard-000/complete.json').write_text(json.dumps({
            'artifact': 'dino-zarr-infer-v2', 'stores_embeddings': True, 'model_sha256': file_sha256(model_path), 'source_sha256': 's'}))
        props = root / 'props.parquet'
        pd.DataFrame({'label': np.arange(1, 21), 'area': 1, 'centroid_z': 0., 'centroid_y': 0.,
                      'centroid_x': np.arange(20, dtype=float),
                      'intensity_brdu_mean': np.arange(20, dtype=float),
                      'intensity_edu_mean': np.arange(20, dtype=float)}).to_parquet(props)
        dapi = root / 'dapi.parquet'
        pd.DataFrame({'label': np.arange(1, 21), 'area': 1, 'intensity_dapi_mean': 10.}).to_parquet(dapi)
        provenance = root / 'intensity.json'
        provenance.write_text(json.dumps({'props_sha256': file_sha256(props)}))
        datasets.append({'name': name, 'tranches': [str(root)], 'embedding_run': str(run),
                         'embedding_model': str(model_path), 'intensity_props': str(props),
                         'intensity_provenance': str(provenance), 'dapi_props': str(dapi),
                         'scale_zyx_um': [1., 1., 1.]})
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'datasets': datasets, 'seed': 42, 'validation_cells': 8,
                                  'context_features': ['log1p_cell_mean', 'log_cell_over_local_mean', 'log1p_dapi_mean']}))
    with pytest.raises(ValueError, match='different encoder'):
        fit(config, tmp_path / 'output', workers=1)
    assert not (tmp_path / 'output').exists()


# --- Marker contract. ---

CUSTOM = MarkerContract('dapi', ('edu', 'brdu', 'pcna'))
ENCODER = dict.fromkeys(
    ('feature_extractor', 'weights_sha256', 'model_source_sha256', 'crop_size', 'resize_size', 'crop_eval_size',
     'highpass_sigma_px', 'mask_erosion_px', 'input_channels', 'input_channel_names_by_target',
     'feature_precision', 'feature_execution'), 'same')


def _declare(directory, contract):
    from dino_markers import parse_config, save_config

    directory.mkdir(parents=True, exist_ok=True)
    save_config(parse_config({
        'version': 1, 'reference': {'key': contract.reference, 'label': contract.reference.upper()},
        'markers': [{'key': m, 'label': m.upper(), 'color': '#ff0000'} for m in contract.markers],
        'presentation': {'crop_panels': list(contract.channel_keys), 'negative_color': [.25, .25, .25],
                         'uncalled_color': '#999999'}}, source='test'), directory)


def _dataset(root, name, contract, *, cells=32, seed=0, declare=True, current=True):
    """One dataset's inputs for a fit: tranche labels, embedding run, embedding bundle and intensities."""
    import joblib
    from dino.artifacts import file_sha256

    rng = np.random.default_rng(seed)
    root.mkdir()
    tranche = root / 'tranche'
    names = [f'{name}|{i}' for i in range(1, cells + 1)]
    calls = {m: (np.arange(cells) // 2 ** bit) % 2 for bit, m in enumerate(contract.markers)}
    tranche.mkdir()
    if declare:
        _declare(tranche, contract)
    pd.DataFrame({'cell': names, **calls}).to_csv(tranche / 'labels.csv', index=False)
    model_path = root / 'embedding-model.joblib'
    versioned = ({'schema': 'dino-zarr-model-v7', 'marker_contract': contract.to_json()} if current
                 else {'schema': 'dino-zarr-model-v6'})
    joblib.dump({**versioned, **ENCODER, 'source_sha256': f'source-{name}'}, model_path)
    shard = root / 'run' / 'shard-000'
    (shard / 'predictions').mkdir(parents=True)
    pd.DataFrame({'cell': names, 'label': range(1, cells + 1)}).to_parquet(shard / 'predictions' / 'part-000000.parquet')
    values = {}
    for m in contract.markers:
        e = rng.normal(size=(cells, 1024)) * .3
        e[:, 0] += 2. * (calls[m] - .5)
        values[m] = torch.from_numpy(e).to(torch.bfloat16)
    write_embedding_atomic({**values, 'cell_sha256': ordered_cell_sha256(names)}, shard / 'embeddings' / 'part-000000.pt')
    record = ({'artifact': 'dino-zarr-infer-v3', 'marker_contract': contract.to_json()} if current
              else {'artifact': 'dino-zarr-infer-v2'})
    (shard / 'complete.json').write_text(json.dumps({
        **record, 'stores_embeddings': True, 'model_sha256': file_sha256(model_path), 'source_sha256': f'source-{name}'}))
    props = root / 'props.parquet'
    pd.DataFrame({'label': np.arange(1, cells + 1), 'area': 1, 'centroid_z': 0., 'centroid_y': 0.,
                  'centroid_x': np.arange(cells, dtype=float),
                  **{f'intensity_{m}_mean': rng.uniform(10, 500, cells) for m in contract.markers}}).to_parquet(props)
    dapi = root / 'dapi.parquet'
    pd.DataFrame({'label': np.arange(1, cells + 1), 'area': 1,
                  'intensity_dapi_mean': rng.uniform(100, 300, cells)}).to_parquet(dapi)
    provenance = root / 'intensity.json'
    provenance.write_text(json.dumps({'props_sha256': file_sha256(props)}))
    return {'name': name, 'tranches': [str(tranche)], 'embedding_run': str(root / 'run'),
            'embedding_model': str(model_path), 'intensity_props': str(props),
            'intensity_provenance': str(provenance), 'dapi_props': str(dapi), 'scale_zyx_um': [1., 1., 1.]}


def _config(tmp_path, datasets, *, validation_cells=24):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'datasets': datasets, 'seed': 3, 'validation_cells': validation_cells,
                                  'context_features': ['log1p_cell_mean', 'log_cell_over_local_mean',
                                                       'log1p_dapi_mean']}))
    return config


def test_three_marker_fit_stamps_v2_bundles_and_names_every_marker(tmp_path):
    import joblib
    from dino.joint_train import fit

    datasets = [_dataset(tmp_path / name, name, CUSTOM, seed=seed) for seed, name in enumerate(['first', 'second'])]
    output = tmp_path / 'output'

    fit(_config(tmp_path, datasets), output, workers=1)

    joint = joblib.load(output / 'model.joblib')
    assert joint['schema'] == 'dino-joint-dataset-v2'
    assert joint['marker_contract'] == CUSTOM.to_json()
    for m in CUSTOM.markers:
        assert joint[f'{m}_model'].n_features_in_ == 1024 + 2 * (3 + 1)
        assert f'{m}_feature_scaler' in joint and f'{m}_cutoff' in joint
    for name in ('first', 'second'):
        deployed = joblib.load(output / 'deploy' / name / 'model.joblib')
        assert deployed['schema'] == 'dino-knn-intensity-v2'
        assert deployed['marker_contract'] == CUSTOM.to_json()
        assert deployed['joint_model']['schema'] == 'dino-joint-dataset-v2'
        assert deployed['classifier_feature_mode'] == 'marker'
        for m in CUSTOM.markers:
            assert deployed[f'{m}_feature_scaler'].n_features_in_ == 1024 + 3
            assert deployed[f'{m}_cutoff'] == joint[f'{m}_cutoff']
        assert list(pd.read_csv(output / 'deploy' / name / 'labels.csv').columns) == [
            'cell', 'edu', 'brdu', 'pcna', 'tranche']
    assert list(pd.read_csv(output / 'labels.csv').columns) == ['cell', 'edu', 'brdu', 'pcna', 'tranche', 'dataset']
    predictions = pd.read_parquet(output / 'labeled_predictions.parquet')
    for m in CUSTOM.markers:
        assert {f'p_{m}', f'pred_{m}', f'oof_p_{m}', m} <= set(predictions.columns)
    metrics = json.loads((output / 'metrics.json').read_text())
    assert set(metrics['targets']) == set(CUSTOM.markers)
    assert metrics['classifier_features'] == {m: 1024 + 2 * 4 for m in CUSTOM.markers}


def test_fit_rejects_a_tranche_with_reordered_markers_before_reading_its_columns(tmp_path):
    from dino.joint_train import fit

    default = DEFAULT_CONTRACT
    datasets = [_dataset(tmp_path / name, name, default, cells=24, seed=seed, current=False, declare=False)
                for seed, name in enumerate(['first', 'second'])]
    _declare(Path(datasets[1]['tranches'][0]), MarkerContract('pi', ('edu', 'brdu')))
    # Without its marker columns, a read of this tranche's labels would fail with a different error.
    Path(datasets[1]['tranches'][0], 'labels.csv').write_text('cell\nsecond|1\n')

    with pytest.raises(ContractMismatchError, match='marker order is part of the contract'):
        fit(_config(tmp_path, datasets, validation_cells=12), tmp_path / 'output', workers=1)
    assert not (tmp_path / 'output').exists()


def test_fit_rejects_a_shard_inference_record_with_another_contract(tmp_path):
    from dino.joint_train import fit

    datasets = [_dataset(tmp_path / name, name, DEFAULT_CONTRACT, cells=24, seed=seed, current=False)
                for seed, name in enumerate(['first', 'second'])]
    complete = Path(datasets[0]['embedding_run'], 'shard-000', 'complete.json')
    record = json.loads(complete.read_text())
    complete.write_text(json.dumps({**record, 'artifact': 'dino-zarr-infer-v3',
                                    'marker_contract': MarkerContract('pi', ('edu', 'brdu')).to_json()}))

    with pytest.raises(ContractMismatchError, match='complete.json'):
        fit(_config(tmp_path, datasets, validation_cells=12), tmp_path / 'output', workers=1)


def test_fit_rejects_datasets_with_different_contracts(tmp_path):
    from dino.joint_train import fit

    datasets = [_dataset(tmp_path / 'first', 'first', DEFAULT_CONTRACT, cells=24, current=False),
                _dataset(tmp_path / 'second', 'second', MarkerContract('pi', ('edu', 'brdu')), cells=24, seed=1)]

    with pytest.raises(ContractMismatchError, match='second'):
        fit(_config(tmp_path, datasets, validation_cells=12), tmp_path / 'output', workers=1)


def test_labeled_embeddings_rejects_a_marker_missing_from_the_partition(tmp_path):
    run = _embedding_run(tmp_path, [[['r|1', 'r|2']]])
    three = MarkerContract('pi', ('brdu', 'edu', 'pcna'))
    (run / 'shard-000' / 'complete.json').write_text(json.dumps({
        'artifact': 'dino-zarr-infer-v3', 'marker_contract': three.to_json(), 'stores_embeddings': True,
        'model_sha256': 'm', 'source_sha256': 's'}))

    with pytest.raises(ValueError, match='pcna'):
        labeled_embeddings(['r|1'], run, model_sha256='m', source_sha256='s', workers=1, contract=three)
