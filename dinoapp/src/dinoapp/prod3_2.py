"""Prod3-2 adapter for the shared native ventricular-field reviewer."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from dino_markers import INFERENCE, cutoff_key, prediction_column, probability_column

from dino.artifacts import file_sha256, read_parquet_artifact
from dino.classifier_features import validate_feature_contract
from dino.zarr_source import CellCropReader
from dinoapp.display_pixels import DisplayPixels
from dinoapp.native import AnnotationFiles as AnnotationFiles, adapter_config, classifier_caption
from dinoapp.native import load_prediction_pointer, prediction_contract
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
    raw_channels = {'DAPI': 'pi', 'BrdU': 'brdu', 'EdU': 'edu'}

    dataset_name = 'Prod3-2'
    dataset_root = ROOT
    review_directory = 'dino-prod3-2'
    reembedding_run = Path('/warm/prod3-2-514-local3d-ls-z128-core80/dino')
    surface_coordinates = reembedding_run / 'vz-coordinates/cells.parquet'
    prediction_pointer: Path | None = REVIEW_ROOT / 'cell-review/current.json'

    @classmethod
    def annotation_directories(cls, root: Path | None = None) -> tuple[Path, ...]:
        root = cls.dataset_root if root is None else root
        return (root / cls.review_directory / 'barrage-cortex-v1',)

    def __init__(self, root: Path | None = None):
        root = self.dataset_root if root is None else root
        self.root = root
        review_root = root / self.review_directory
        self.annotation_paths = tuple(directory / 'labels.csv' for directory in self.annotation_directories(root))
        self.default_annotation_path = self.annotation_paths[0]
        self.config = adapter_config(
            self.default_annotation_path.parent, self.raw_channels, adapter=self.dataset_name
        )
        markers = self.config.contract.markers
        run = self.reembedding_run
        complete = json.loads((run / 'complete.json').read_text())
        self._require_contract(
            prediction_contract(complete, where=str(run / 'complete.json')),
            actual_from=str(run / 'complete.json'),
        )
        if not complete['passed'] or not complete['baseline_classifier_preserved']:
            raise ValueError(f'{self.dataset_name} corrected baseline inference is not validated')
        model_path = Path(complete['model'])
        if file_sha256(model_path) != complete['model_sha256']:
            raise ValueError(f'{self.dataset_name} baseline inference model identity changed')
        embedding_model_path = model_path
        bundle = joblib.load(model_path)
        prediction_record = complete
        prediction_manifest = run / 'complete.json'
        scored_directory = None
        if self.prediction_pointer is not None:
            _, scored_directory, prediction_manifest, prediction_record = load_prediction_pointer(self.prediction_pointer)
            self._require_contract(
                prediction_contract(prediction_record, where=str(prediction_manifest)),
                actual_from=str(prediction_manifest),
            )
            if (prediction_record['source_sha256'] != complete['source_sha256']
                    or prediction_record['embedding_model_sha256'] != complete['model_sha256']
                    or Path(prediction_record['embedding_run']).resolve() != (run / 'infer').resolve()
                    or prediction_record['cells'] != complete['cells']):
                raise ValueError(f'{self.dataset_name} selected predictions differ from the corrected embeddings')
            model_path = Path(prediction_record['model'])
            if file_sha256(model_path) != prediction_record['model_sha256']:
                raise ValueError(f'{self.dataset_name} selected classifier identity changed')
            embedding_bundle = bundle
            bundle = joblib.load(model_path)
            self._require_contract(validate_feature_contract(bundle, embedding_bundle), actual_from=str(model_path))
        register_jpegxr_codec()
        self.display_pixels = DisplayPixels(embedding_model_path, run / 'source')
        self._require_contract(self.display_pixels.contract, actual_from=str(embedding_model_path))
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
        self.arrays = {name: source.channels[key] for name, key in self.raw_channels.items()}
        self.crop_reader = CellCropReader(
            labels=self.labels,
            channels={**{key: self.arrays[name] for name, key in self.raw_channels.items()},
                      'dapi': self.arrays['DAPI']},
            offset_zyx=(0, 0, 0),
            cache_bytes=4 * int(np.prod(self.labels.chunks)) * self.labels.dtype.itemsize
            + 64 * 1024**2,
        )
        self.crop_lock = Lock()
        self.pixel_version = self.display_pixels.identity
        self.display_name = f'{self.display_name} · {self.display_pixels.description}'
        self.props_path = run / 'source/cells.parquet'
        self.crop_contract = json.loads((review_root / 'barrage-cortex-v1/run.json').read_text())
        self.contrast = {key: (0., self.crop_contract['display_max'][key])
                         for key in self.config.contract.channel_keys}

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
                shard_record = json.loads((directory / 'complete.json').read_text())
                self._require_contract(
                    INFERENCE.contract(shard_record, where=str(directory / 'complete.json')),
                    actual_from=str(directory / 'complete.json'),
                )
                if (shard_record['model_sha256'] != complete['model_sha256']
                        or shard_record['source_sha256'] != complete['source_sha256']
                        or shard_record['cells'] != shard['cells']):
                    raise ValueError(f'{self.dataset_name} inference shard identity differs')
                files = sorted((directory / 'predictions').glob('*.parquet'))
                if len(files) != shard['partitions']:
                    raise ValueError(f'{self.dataset_name} inference partition coverage differs')
                parts.extend((path, shard_record) for path in files)
        else:
            files = sorted(scored_directory.glob('*.parquet'))
            if len(files) != prediction_record['prediction_partitions']:
                raise ValueError(f'{self.dataset_name} selected prediction partition coverage differs')
            parts.extend((path, prediction_record) for path in files)
        for path, run_record in parts:
            metadata, _ = read_parquet_artifact(path)
            if metadata.get('artifact') != run_record.get('artifact'):
                raise ValueError(f'{self.dataset_name} prediction artifact differs from its run record: {path}')
            self._require_contract(prediction_contract(metadata, where=str(path)), actual_from=str(path))
            if (metadata['model_sha256'] != prediction_record['model_sha256']
                    or metadata['source_sha256'] != complete['source_sha256']
                    or any(metadata[cutoff_key(marker)] != bundle[cutoff_key(marker)] for marker in markers)):
                raise ValueError(f'{self.dataset_name} prediction provenance differs: {path}')
        columns = ['cell', 'label', *markers, *(probability_column(marker) for marker in markers)]
        frame = ds.dataset([str(path) for path, _ in parts], format='parquet').to_table(columns=columns).to_pandas()
        if len(frame) != complete['cells']:
            raise ValueError(f'{self.dataset_name} prediction cell count differs from its inference')
        cutoffs = {marker: float(bundle[cutoff_key(marker)]) for marker in markers}
        self._adopt_model_calls(frame, cutoffs=cutoffs, cell_prefix=self.cell_prefix)
        self.model_version = prediction_record['model_sha256'][:12]
        self.prediction_manifest = str(prediction_manifest)
        self.classifier_caption = classifier_caption(self.model_version, self.config, cutoffs)
        train = frame.cell.isin(bundle['train_cells'])
        validation = frame.cell.isin(bundle['validation_cells'])
        self._gallery_predictions = frame.rename(
            columns={marker: prediction_column(marker) for marker in markers},
        ).copy()
        self._gallery_predictions['split'] = 'untrained'
        self._gallery_predictions.loc[train, 'split'] = 'training'
        self._gallery_predictions.loc[validation, 'split'] = 'validation'
        self._gallery_predictions['tranche'] = 'barrage-cortex-v1'
        self.training_predictions = self._gallery_predictions.loc[train | validation].copy()
        if self.prediction_pointer is not None:
            self.training_predictions = load_review_predictions(
                Path(prediction_record['review_manifest']),
                model_sha256=prediction_record['model_sha256'], cell_prefix=self.cell_prefix,
                contract=self.config.contract,
            )

    @property
    def gallery_predictions(self):
        return self._gallery_predictions

    def _field_boundary_info(self, labels: list[int]) -> dict[int, dict[str, object]]:
        return {
            row['label']: {'r_vz_um': row['r_vz_um']}
            for row in pq.read_table(self.layer_path, columns=['label', 'r_vz_um'],
                                     filters=[('label', 'in', labels)]).to_pylist()
        } if labels else {}
