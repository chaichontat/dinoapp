"""Prod3-2 adapter for the shared native ventricular-field reviewer."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from dino.artifacts import file_sha256, read_parquet_artifact
from dino.classifier_features import validate_feature_contract
from dino.zarr_source import CellCropReader
from dinoapp.display_pixels import DisplayPixels
from dinoapp.native import AnnotationFiles as AnnotationFiles, combined_uncertainty
from dinoapp.native import load_prediction_pointer
from dinoapp.native import render_field as render_field
from dinoapp.review_predictions import load_review_predictions
from dinoapp.prod3 import FieldSource as Prod3FieldSource
from squisher.jpegxr_zarr import register_jpegxr_codec

ROOT = Path('/working/eduseg/prod3-2')
REVIEW_ROOT = ROOT / 'dino-prod3-2'


class FieldSource(Prod3FieldSource):
    """Use the reviewed ventricular surface and validated saved model calls."""

    display_name = 'Prod3-2 native'
    cell_prefix = '235-4-edu-workflow-recovered'
    anatomy_channel = 'DAPI'

    dataset_name = 'Prod3-2'
    dataset_root = ROOT
    review_directory = 'dino-prod3-2'
    reembedding_run = Path('/warm/prod3-2-514-local3d-ls-z128-core80/dino')
    surface_coordinates = reembedding_run / 'vz-coordinates/cells.parquet'
    prediction_pointer: Path | None = REVIEW_ROOT / 'cell-review/current.json'

    def __init__(self, root: Path | None = None):
        root = self.dataset_root if root is None else root
        self.root = root
        review_root = root / self.review_directory
        run = self.reembedding_run
        complete = json.loads((run / 'complete.json').read_text())
        if not complete['passed'] or not complete['baseline_classifier_preserved']:
            raise ValueError(f'{self.dataset_name} corrected baseline inference is not validated')
        model_path = Path(complete['model'])
        if file_sha256(model_path) != complete['model_sha256']:
            raise ValueError(f'{self.dataset_name} baseline inference model identity changed')
        embedding_model_path = model_path
        bundle = joblib.load(model_path)
        prediction_contract = complete
        prediction_manifest = run / 'complete.json'
        scored_directory = None
        if self.prediction_pointer is not None:
            _, scored_directory, prediction_manifest, prediction_contract = load_prediction_pointer(self.prediction_pointer)
            if (prediction_contract['source_sha256'] != complete['source_sha256']
                    or prediction_contract['embedding_model_sha256'] != complete['model_sha256']
                    or Path(prediction_contract['embedding_run']).resolve() != (run / 'infer').resolve()
                    or prediction_contract['cells'] != complete['cells']):
                raise ValueError(f'{self.dataset_name} selected predictions differ from the corrected embeddings')
            model_path = Path(prediction_contract['model'])
            if file_sha256(model_path) != prediction_contract['model_sha256']:
                raise ValueError(f'{self.dataset_name} selected classifier identity changed')
            embedding_bundle = bundle
            bundle = joblib.load(model_path)
            validate_feature_contract(bundle, embedding_bundle)
        register_jpegxr_codec()
        self.display_pixels = DisplayPixels(embedding_model_path, run / 'source')
        source = self.display_pixels.source
        if source.manifest['source_sha256'] != complete['source_sha256']:
            raise ValueError(f'{self.dataset_name} display and inference source identities differ')
        self.scale = np.asarray(self.display_pixels.spatial.scale_zyx_um)
        self.origin = np.asarray(self.display_pixels.spatial.origin_zyx_um)
        self.labels = source.labels
        self.shape = tuple(self.labels.shape)
        self.display_pixels.validate_native_grid(
            labels=source.manifest['labels']['root'], shape=self.shape,
            scale=self.scale, origin=self.origin,
        )
        self.arrays = {'DAPI': source.channels['pi'], 'BrdU': source.channels['brdu'],
                       'EdU': source.channels['edu']}
        self.crop_reader = CellCropReader(
            labels=self.labels,
            channels={'pi': self.arrays['DAPI'], 'dapi': self.arrays['DAPI'],
                      'edu': self.arrays['EdU'], 'brdu': self.arrays['BrdU']},
            offset_zyx=(0, 0, 0),
            cache_bytes=4 * int(np.prod(self.labels.chunks)) * self.labels.dtype.itemsize
            + 64 * 1024**2,
        )
        self.crop_lock = Lock()
        self.pixel_version = self.display_pixels.identity
        self.display_name = f'{self.display_name} · {self.display_pixels.description}'
        self.props_path = run / 'source/cells.parquet'
        self.crop_contract = json.loads((review_root / 'barrage-cortex-v1/run.json').read_text())
        self.contrast = np.asarray([[0., self.crop_contract['display_max'][key]]
                                    for key in ('pi', 'brdu', 'edu')])

        spatial = root / 'spatial-workflow-v1'
        surface_path = spatial / 'vz-surface/surface.npz'
        surface_record = json.loads((surface_path.parent / 'metadata.json').read_text())
        if file_sha256(surface_path) != surface_record['outputs']['surface.npz']:
            raise ValueError(f'{self.dataset_name} reviewed ventricular surface identity changed')
        with np.load(surface_path) as mesh:
            vertices = np.asarray(mesh['vertices_xyz_um'], dtype=np.float64)
            faces = np.asarray(mesh['faces'], dtype=np.int64)
        self.boundaries = [(vertices, faces)]
        self.layer_path = self.surface_coordinates
        coordinates = json.loads((self.layer_path.parent / 'metadata.json').read_text())
        if (coordinates['surface_sha256'] != surface_record['outputs']['surface.npz']
                or coordinates['rows'] != complete['cells']
                or not np.allclose(coordinates['source_scale_zyx_um'], self.scale, rtol=0, atol=1e-12)
                or not np.allclose(coordinates['source_origin_zyx_um'], self.origin, rtol=0, atol=1e-12)):
            raise ValueError(f'{self.dataset_name} ventricular coordinates differ from the native grid')
        zi = int(round((float(np.median(vertices[:, 2])) - self.origin[0]) / self.scale[0]))
        if not 0 <= zi < self.shape[0]:
            raise ValueError(f'{self.dataset_name} ventricular surface is outside the native image')
        self.default_z = float(self.origin[0] + zi * self.scale[0])

        parts = []
        if scored_directory is None:
            for shard in complete['shards']:
                directory = run / 'infer' / f"shard-{shard['shard']:03d}"
                contract = json.loads((directory / 'complete.json').read_text())
                if (contract['model_sha256'] != complete['model_sha256']
                        or contract['source_sha256'] != complete['source_sha256']
                        or contract['cells'] != shard['cells']):
                    raise ValueError(f'{self.dataset_name} inference shard identity differs')
                files = sorted((directory / 'predictions').glob('*.parquet'))
                if len(files) != shard['partitions']:
                    raise ValueError(f'{self.dataset_name} inference partition coverage differs')
                parts.extend(files)
        else:
            parts = sorted(scored_directory.glob('*.parquet'))
            if len(parts) != prediction_contract['prediction_partitions']:
                raise ValueError(f'{self.dataset_name} selected prediction partition coverage differs')
        for path in parts:
            metadata, _ = read_parquet_artifact(path)
            if (metadata['model_sha256'] != prediction_contract['model_sha256']
                    or metadata['source_sha256'] != complete['source_sha256']
                    or any(metadata[f'{marker}_cutoff'] != bundle[f'{marker}_cutoff']
                           for marker in ('brdu', 'edu'))):
                raise ValueError(f'{self.dataset_name} prediction provenance differs: {path}')
        parts = [str(path) for path in parts]
        frame = ds.dataset(parts, format='parquet').to_table(
            columns=['cell', 'label', 'brdu', 'edu', 'p_brdu', 'p_edu'],
        ).to_pandas()
        if (len(frame) != complete['cells'] or frame.label.duplicated().any()
                or not frame.cell.equals(self.cell_prefix + '|' + frame.label.astype(str))
                or not frame[['brdu', 'edu']].isin([0, 1]).all().all()):
            raise ValueError(f'{self.dataset_name} prediction cell identities or calls differ')
        self.calls = np.full(int(frame.label.max()) + 1, 4, dtype=np.uint8)
        self.calls[frame.label] = frame.edu.to_numpy(dtype=np.uint8) + 2 * frame.brdu.to_numpy(dtype=np.uint8)
        uncertainty = combined_uncertainty(
            frame.p_brdu.to_numpy(), frame.p_edu.to_numpy(),
            brdu_threshold=float(bundle['brdu_cutoff']), edu_threshold=float(bundle['edu_cutoff']),
        )
        self.uncertainty = np.full(len(self.calls), np.nan, dtype=np.float32)
        self.uncertainty[frame.label] = uncertainty
        self.uncertainty_log_vmax = float(np.quantile(np.log1p(uncertainty[np.isfinite(uncertainty)]), .999))
        self.model_version = prediction_contract['model_sha256'][:12]
        self.prediction_manifest = str(prediction_manifest)
        self.classifier_caption = (
            f"Classifier {self.model_version} · BrdU cutoff {bundle['brdu_cutoff']:.10f} · "
            f"EdU cutoff {bundle['edu_cutoff']:.10f}"
        )
        train = frame.cell.isin(bundle['train_cells'])
        validation = frame.cell.isin(bundle['validation_cells'])
        self._gallery_predictions = frame.rename(
            columns={'brdu': 'pred_brdu', 'edu': 'pred_edu'},
        ).copy()
        self._gallery_predictions['split'] = 'untrained'
        self._gallery_predictions.loc[train, 'split'] = 'training'
        self._gallery_predictions.loc[validation, 'split'] = 'validation'
        self._gallery_predictions['tranche'] = 'barrage-cortex-v1'
        self.training_predictions = self._gallery_predictions.loc[train | validation].copy()
        if self.prediction_pointer is not None:
            self.training_predictions = load_review_predictions(
                Path(prediction_contract['review_manifest']),
                model_sha256=prediction_contract['model_sha256'], cell_prefix=self.cell_prefix,
            )
        self.annotation_paths = (review_root / 'barrage-cortex-v1/labels.csv',)
        self.default_annotation_path = self.annotation_paths[0]

    @property
    def gallery_predictions(self):
        return self._gallery_predictions

    def _field_boundary_info(self, labels: list[int]) -> dict[int, dict[str, object]]:
        return {
            row['label']: {'r_vz_um': row['r_vz_um']}
            for row in pq.read_table(self.layer_path, columns=['label', 'r_vz_um'],
                                     filters=[('label', 'in', labels)]).to_pylist()
        } if labels else {}
