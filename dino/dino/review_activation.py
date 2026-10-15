"""Atomically activate complete RHU2 cortical predictions in the copied reviewer."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from dino_markers import DEFAULT_CONTRACT, SPATIAL_REVIEW, require_same_contract

from dino.zarr_source import write_json_atomic
from dino.review_contracts import combined_uncertainty


def activate(model_root: Path, review: Path) -> None:
    complete = json.loads((model_root / 'inference/complete.json').read_text())
    if complete['status'] != 'complete':
        raise ValueError('Inference is incomplete')
    manifest_path = review / 'spatial.json'
    manifest = json.loads(manifest_path.read_text())
    require_same_contract(DEFAULT_CONTRACT, SPATIAL_REVIEW.contract(manifest, where=str(manifest_path)),
                          expected_from='RHU2 activation (BrdU/EdU)', actual_from=str(manifest_path))
    if complete['cells'] != manifest['cells']:
        raise ValueError('Cortical prediction count differs from the review cohort')
    thresholds = complete['decision_thresholds']
    label_snapshot = (review / 'labels.csv').read_bytes()
    old_manifest = manifest_path.read_bytes()
    destination = Path(tempfile.mkdtemp(prefix='spatial-model-v1-', dir=review))
    uncertainty = []
    total = 0
    for source in manifest['rois']:
        name = f'{source["dataset"]}__{source["roi"]}.parquet'
        old = pd.read_parquet(review / source['table']).sort_values('cell').reset_index(drop=True)
        predictions = pd.read_parquet(model_root / 'inference/predictions' / name).sort_values('cell').reset_index(drop=True)
        pd.testing.assert_frame_equal(old[['cell', 'label', 'x', 'y', 'z']], predictions[['cell', 'label', 'x', 'y', 'z']], check_dtype=False)
        if len(predictions) != source['cells'] or predictions.cell.duplicated().any():
            raise ValueError(f'Invalid prediction identity/count: {name}')
        scores = combined_uncertainty({'brdu': predictions.brdu_prob.to_numpy(), 'edu': predictions.edu_prob.to_numpy()},
                                      {'brdu': thresholds['brdu'], 'edu': thresholds['edu']})
        uncertainty.append(np.log1p(scores[np.isfinite(scores)]))
        path = destination / name
        predictions.to_parquet(path, index=False)
        pd.testing.assert_frame_equal(predictions, pd.read_parquet(path))
        source['table'] = str(path.relative_to(review))
        total += len(predictions)
    if total != manifest['cells']:
        raise ValueError('Activation coverage incomplete')
    backup = review / 'spatial.before-model-v1.json'
    if not backup.exists():
        shutil.copyfile(manifest_path, backup)
    manifest.update(decision_thresholds=thresholds, model_id='rhu2-logistic-model-v1',
                    model=complete['model'], model_predictions=str(model_root / 'inference/predictions'),
                    activated_at_utc=datetime.now(timezone.utc).isoformat(),
                    uncertainty_log_vmax=float(np.quantile(np.concatenate(uncertainty), .999)))
    if manifest_path.read_bytes() != old_manifest:
        raise RuntimeError('Active spatial manifest changed while staging activation')
    if (review / 'labels.csv').read_bytes() != label_snapshot:
        print('Live annotations changed during activation; their edits were preserved.', flush=True)
    write_json_atomic(manifest, manifest_path)
    print(f'Activated {manifest["model_id"]} for {total} cortical cells; thresholds={thresholds}', flush=True)
