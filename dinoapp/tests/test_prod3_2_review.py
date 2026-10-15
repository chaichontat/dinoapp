from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from dino_markers import default_config

from dinoapp.sources import NativeSource


def test_saved_labels_can_be_reviewed_without_becoming_training_rows(tmp_path: Path):
    frame = pd.DataFrame({'cell': ['235-4-edu-workflow|10'], 'label': [10],
                          'split': ['untrained'], 'tranche': ['barrage-cortex-v1'],
                          'p_brdu': [.8], 'p_edu': [.2],
                          'pred_brdu': [True], 'pred_edu': [False]})
    source = SimpleNamespace(
        annotation_paths=(tmp_path / 'labels.csv',), default_annotation_path=tmp_path / 'labels.csv',
        training_predictions=frame.iloc[:0], gallery_predictions=frame,
        cell_id=lambda label: f'235-4-edu-workflow|{label}',
        display_name='Prod3-2 native', model_version='baseline', origin=[0.], scale=[.6],
        config=default_config(),
        marker_crop=lambda label, marker: (b'corrected crop', 4),
        cell_crop=lambda label: (b'corrected crop', 4),
    )
    adapter = NativeSource(source, label_root=tmp_path)
    labels = {'235-4-edu-workflow|10': (0, 1)}
    rows = adapter.correction_rows(labels, identity=(0, 1), subset='All labeled', dataset='All datasets')
    assert rows.cell.tolist() == ['235-4-edu-workflow|10']
    assert adapter.correction_rows(labels, identity=(0, 1), subset='Training', dataset='All datasets').empty
    assert source.training_predictions.empty
    assert adapter.crop('235-4-edu-workflow|10', 'edu').image == b'corrected crop'
