"""Galleries consume model-bound review rows with their actual fit split."""
import json

import pandas as pd
import pytest
from dino_markers import DEFAULT_CONTRACT

from dino.artifacts import file_sha256
from dinoapp.review_predictions import load_review_predictions


@pytest.fixture
def review_artifacts(tmp_path):
    rows = pd.DataFrame([
        ['prod3|10', 10, 'vz_svz', 'training', .1, False, .9, True],
        ['prod3|20', 20, 'vz_svz', 'training', .8, True, .2, False],
    ], columns=['cell', 'label', 'tranche', 'split', 'p_brdu', 'pred_brdu', 'p_edu', 'pred_edu'])
    path = tmp_path / 'predictions.parquet'
    rows.to_parquet(path)
    manifest = dict(artifact='dino-labeled-review-v1', model_sha256='selected', cells=2,
                    training_cells=2, validation_cells=0, brdu_cutoff=.5, edu_cutoff=.3,
                    predictions=str(path), predictions_sha256=file_sha256(path))
    manifest_path = tmp_path / 'complete.json'
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, path, manifest, rows


def test_final_fit_review_has_all_training_rows_and_no_invented_holdout(review_artifacts):
    manifest_path, _, _, _ = review_artifacts
    rows = load_review_predictions(
        manifest_path, cell_prefix='prod3', model_sha256='selected', contract=DEFAULT_CONTRACT
    )
    assert rows.cell.tolist() == ['prod3|10', 'prod3|20']
    assert rows.split.tolist() == ['training', 'training']
    assert rows.p_edu.tolist() == [.9, .2]


@pytest.mark.parametrize('invalid', ['column', 'duplicate', 'split', 'probability', 'identity', 'call'])
def test_invalid_review_rows_fail_at_load_boundary(review_artifacts, invalid):
    manifest_path, path, manifest, rows = review_artifacts
    if invalid == 'column':
        rows = rows.drop(columns=['pred_edu'])
    elif invalid == 'duplicate':
        rows.loc[1, 'cell'] = rows.loc[0, 'cell']
    elif invalid == 'split':
        rows.loc[0, 'split'] = 'validation'
    elif invalid == 'probability':
        rows.loc[0, 'p_edu'] = 1.5
    elif invalid == 'identity':
        rows.loc[0, 'cell'] = 'prod3|11'
    else:
        rows.loc[0, 'pred_edu'] = False
    rows.to_parquet(path)
    manifest['predictions_sha256'] = file_sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_review_predictions(
            manifest_path, cell_prefix='prod3', model_sha256='selected', contract=DEFAULT_CONTRACT
        )


@pytest.mark.parametrize('artifact', ['model_binding', 'predictions'])
def test_mutated_binding_or_predictions_are_rejected(review_artifacts, artifact):
    manifest_path, path, manifest, _ = review_artifacts
    if artifact == 'predictions':
        path.write_bytes(b'changed')
    else:
        manifest['model_sha256'] = 'different'
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='classifier|identity'):
        load_review_predictions(
            manifest_path, cell_prefix='prod3', model_sha256='selected', contract=DEFAULT_CONTRACT
        )
