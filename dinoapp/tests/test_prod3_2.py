import base64
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr
from dino_markers import (
    CORTEX_RESCORE,
    DEFAULT_CONTRACT,
    INFERENCE,
    ContractMismatchError,
    UnsupportedArtifactError,
    config_to_json,
    default_config,
    parse_config,
)
from PIL import Image

from dino.artifacts import PARQUET_METADATA_KEY
from dinoapp import prod3_2
from dinoapp.app import get_native_modules
from dinoapp.native import Field, render_field


def test_prod3_2_is_a_native_dataset():
    native, shared_app = get_native_modules('Prod3-2')
    assert native.__name__ == 'dinoapp.prod3_2'
    assert shared_app.__name__ == 'dinoapp.native_app'


def test_single_ventricular_surface_and_true_dapi_render():
    mask = np.ones((40, 40), dtype=np.uint32)
    field = Field(
        {'pi': np.full(mask.shape, 64, dtype=np.uint16),
         'brdu': np.zeros(mask.shape), 'edu': np.zeros(mask.shape)},
        mask, np.array([1], dtype=np.uint32), np.zeros(mask.shape, dtype=int),
        np.zeros(mask.shape, dtype=bool), np.zeros(2), np.ones(2),
        np.full(2, 20.), 12., 20, 0, 0,
        [np.array([[[10., 5.], [10., 35.]]])], {},
    )
    result = render_field(field, np.array([0]), config=default_config(), mode='pi', selected=None,
                          border_alpha=0., boundary_alpha=1.,
                          contrast=dict.fromkeys(('pi', 'brdu', 'edu'), (0., 255.)))
    image = Image.open(io.BytesIO(base64.b64decode(result.split(',', 1)[1])))
    assert image.size == (40, 40)
    assert image.getpixel((1, 1))[:3] == (64, 64, 64)


CELLS = [1, 2, 4, 5]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Fixture:
    """A Prod3-2 dataset root whose pixel readers are replaced by small stand-ins."""

    def __init__(self, tmp_path: Path, monkeypatch, *, pointer=False):
        self.root = tmp_path / 'prod3-2'
        self.run = tmp_path / 'reembedding'
        self.review = self.root / 'dino-test'
        for directory in (self.run / 'infer/shard-000/predictions', self.review / 'barrage-cortex-v1',
                          self.root / 'spatial-workflow-v1/vz-surface', self.run / 'vz-coordinates'):
            directory.mkdir(parents=True, exist_ok=True)
        (self.review / 'barrage-cortex-v1/run.json').write_text(
            json.dumps({'display_max': {'pi': 10., 'brdu': 20., 'edu': 30.}}))
        surface = self.root / 'spatial-workflow-v1/vz-surface/surface.npz'
        np.savez(surface, vertices_xyz_um=np.array([[0., 0, 0], [0, 1, 0], [0, 0, 1]]), faces=np.array([[0, 1, 2]]))
        (surface.parent / 'metadata.json').write_text(json.dumps({'outputs': {'surface.npz': _digest(surface)}}))
        (self.run / 'vz-coordinates/metadata.json').write_text(json.dumps({
            'surface_sha256': _digest(surface), 'rows': len(CELLS),
            'source_scale_zyx_um': [1., 1., 1.], 'source_origin_zyx_um': [0., 0., 0.]}))
        self.model = tmp_path / 'embedding-model.joblib'
        self.selected = tmp_path / 'selected-model.joblib'
        cells = [f'tst|{label}' for label in CELLS]
        for path in (self.model, self.selected):
            joblib.dump({'brdu_cutoff': .5, 'edu_cutoff': .5, 'train_cells': cells[:2],
                         'validation_cells': cells[2:3]}, path)
        self.complete = {
            'passed': True, 'baseline_classifier_preserved': True, 'model': str(self.model),
            'model_sha256': _digest(self.model), 'source_sha256': 'source-hash', 'cells': len(CELLS),
            'shards': [{'shard': 0, 'cells': len(CELLS), 'partitions': 1}],
        }
        self.shard = {'model_sha256': _digest(self.model), 'source_sha256': 'source-hash', 'cells': len(CELLS),
                      'artifact': 'dino-zarr-infer-v2'}
        self.rows = pd.DataFrame({
            'cell': [f'tst|{label}' for label in CELLS], 'label': CELLS,
            'brdu': [False, False, True, True], 'edu': [False, True, False, True],
            'p_brdu': [.1, .2, .8, .9], 'p_edu': [.3, .7, .2, .6],
        })
        self.partition = {'artifact': 'dino-zarr-infer-v2', 'model_sha256': _digest(self.model),
                          'source_sha256': 'source-hash', 'brdu_cutoff': .5, 'edu_cutoff': .5}
        self.display_contract = DEFAULT_CONTRACT
        self.pointer = None
        monkeypatch.setattr(prod3_2, 'register_jpegxr_codec', lambda: None)
        monkeypatch.setattr(prod3_2, 'CellCropReader', lambda **kwargs: SimpleNamespace(**kwargs))
        labels = zarr.open_array(tmp_path / 'labels', mode='w', shape=(1, 4, 4), chunks=(1, 4, 4), dtype='u4')
        channels = {key: zarr.open_array(tmp_path / key, mode='w', shape=(1, 4, 4), chunks=(1, 4, 4), dtype='u2')
                    for key in ('pi', 'brdu', 'edu')}
        fixture = self

        def display(model, source_dir):
            assert source_dir == fixture.run / 'source'
            return SimpleNamespace(
                source=SimpleNamespace(manifest={'source_sha256': 'source-hash', 'labels': {'root': str(tmp_path / 'labels')}},
                                       labels=labels, channels=channels),
                spatial=SimpleNamespace(scale_zyx_um=(1., 1., 1.), origin_zyx_um=(0., 0., 0.)),
                validate_native_grid=lambda **kwargs: None, identity='display-id', description='stub pixels',
                contract=fixture.display_contract, model_path=model,
            )

        monkeypatch.setattr(prod3_2, 'DisplayPixels', display)
        if pointer:
            scored = tmp_path / 'scored'
            scored.mkdir()
            self.record = {'artifact': 'dino-cortex-rescore-v1', 'source_sha256': 'source-hash',
                           'embedding_model_sha256': self.complete['model_sha256'],
                           'embedding_run': str(self.run / 'infer'), 'cells': len(CELLS), 'model': str(self.selected),
                           'model_sha256': _digest(self.selected), 'prediction_partitions': 1,
                           'review_manifest': str(tmp_path / 'review.json')}
            self.scored = scored
            self.partition = {**self.partition, 'artifact': 'dino-cortex-rescore-v1', 'model_sha256': _digest(self.selected)}
            self.pointer = tmp_path / 'current.json'
            monkeypatch.setattr(prod3_2, 'load_prediction_pointer',
                                lambda path: ('selected', scored, tmp_path / 'scored-complete.json', dict(self.record)))
            monkeypatch.setattr(prod3_2, 'validate_feature_contract', lambda bundle, embedding: self.selected_contract)
            monkeypatch.setattr(prod3_2, 'load_review_predictions', lambda path, **kwargs: self.review_frame)
            self.selected_contract = DEFAULT_CONTRACT
            self.review_frame = self.rows.assign(split='training', tranche='barrage-cortex-v1')

    def build(self) -> type:
        (self.run / 'complete.json').write_text(json.dumps(self.complete))
        (self.run / 'infer/shard-000/complete.json').write_text(json.dumps(self.shard))
        directory = self.scored if self.pointer is not None else self.run / 'infer/shard-000/predictions'
        table = pa.Table.from_pandas(self.rows, preserve_index=False)
        table = table.replace_schema_metadata({PARQUET_METADATA_KEY: json.dumps(self.partition).encode()})
        pq.write_table(table, directory / 'part-0.parquet')
        fixture = self

        class Adapter(prod3_2.FieldSource):
            dataset_name = 'Test3-2'
            dataset_root = fixture.root
            review_directory = 'dino-test'
            reembedding_run = fixture.run
            surface_coordinates = fixture.run / 'vz-coordinates/cells.parquet'
            cell_prefix = 'tst'
            prediction_pointer = fixture.pointer

        return Adapter

    def write_markers(self, config) -> None:
        (self.review / 'barrage-cortex-v1/markers.json').write_text(json.dumps(config_to_json(config)))


REORDERED = parse_config({
    'version': 1, 'reference': {'key': 'pi', 'label': 'PI'},
    'markers': [{'key': 'edu', 'label': 'EdU', 'color': '#00ff00'},
                {'key': 'brdu', 'label': 'BrdU', 'color': '#ff00ff'}],
    'presentation': {'crop_panels': ['pi', 'edu', 'brdu'], 'negative_color': [.25, .25, .25],
                     'uncalled_color': '#999999'},
}, source='test')


def test_prod3_2_reads_calls_provenance_contrast_and_gallery_splits(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)

    source = fixture.build()()

    assert source.config == default_config()
    np.testing.assert_array_equal(source.calls, [4, 0, 1, 4, 2, 3])
    assert source.model_version == fixture.complete['model_sha256'][:12]
    assert source.classifier_caption == (
        f'Classifier {source.model_version} · BrdU cutoff 0.5000000000 · EdU cutoff 0.5000000000')
    assert source.contrast == {'pi': (0., 10.), 'brdu': (0., 20.), 'edu': (0., 30.)}
    assert source.annotation_paths == (fixture.review / 'barrage-cortex-v1/labels.csv',)
    gallery = source.gallery_predictions
    assert gallery.split.tolist() == ['training', 'training', 'validation', 'untrained']
    assert {'pred_brdu', 'pred_edu', 'p_brdu', 'p_edu'} <= set(gallery.columns)
    assert source.training_predictions.cell.tolist() == ['tst|1', 'tst|2', 'tst|4']
    assert set(source.crop_reader.channels) == {'pi', 'dapi', 'edu', 'brdu'}
    assert set(source.arrays) == {'DAPI', 'BrdU', 'EdU'}


def test_prod3_2_reads_the_selected_pointer_predictions_and_classifier(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch, pointer=True)

    source = fixture.build()()

    assert source.model_version == fixture.record['model_sha256'][:12]
    assert source.training_predictions.equals(fixture.review_frame)
    np.testing.assert_array_equal(source.calls, [4, 0, 1, 4, 2, 3])


def test_prod3_2_rejects_an_annotation_directory_whose_markers_do_not_fit_its_channels(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    three = parse_config({
        'version': 1, 'reference': {'key': 'dapi', 'label': 'DAPI'},
        'markers': [{'key': key, 'label': key.upper(), 'color': color}
                    for key, color in (('edu', '#00ff00'), ('brdu', '#ff00ff'), ('pcna', '#0000ff'))],
        'presentation': {'crop_panels': ['dapi', 'pcna', 'edu', 'brdu'], 'negative_color': [.25, .25, .25],
                         'uncalled_color': '#999999'},
    }, source='test')
    fixture.write_markers(three)

    with pytest.raises(ValueError, match='do not match the marker configuration'):
        fixture.build()()


def test_prod3_2_rejects_reordered_annotations_against_its_inference_run(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.write_markers(REORDERED)

    with pytest.raises(ContractMismatchError, match='marker order is part of the contract'):
        fixture.build()()


def test_prod3_2_rejects_a_reembedding_manifest_of_another_contract(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.complete = INFERENCE.stamp(fixture.complete, REORDERED.contract)

    with pytest.raises(ContractMismatchError, match='marker order is part of the contract'):
        fixture.build()()


def test_prod3_2_rejects_a_shard_inference_record_of_another_contract(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.shard = INFERENCE.stamp({k: v for k, v in fixture.shard.items() if k != 'artifact'},
                                    REORDERED.contract)
    fixture.partition = {**fixture.partition, 'artifact': 'dino-zarr-infer-v3'}

    with pytest.raises(ContractMismatchError, match='marker order is part of the contract'):
        fixture.build()()


def test_prod3_2_requires_a_versioned_shard_inference_record(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.shard = {k: v for k, v in fixture.shard.items() if k != 'artifact'}

    with pytest.raises(UnsupportedArtifactError, match='unsupported inference record'):
        fixture.build()()


def test_prod3_2_rejects_a_display_bundle_of_another_contract(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.display_contract = REORDERED.contract

    with pytest.raises(ContractMismatchError, match='marker order is part of the contract'):
        fixture.build()()


def test_prod3_2_rejects_a_selected_classifier_of_another_contract(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch, pointer=True)
    fixture.selected_contract = REORDERED.contract

    with pytest.raises(ContractMismatchError, match='marker order is part of the contract'):
        fixture.build()()


def test_prod3_2_rejects_a_pointer_manifest_of_another_contract(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch, pointer=True)
    fixture.record = CORTEX_RESCORE.stamp({k: v for k, v in fixture.record.items() if k != 'artifact'},
                                          REORDERED.contract)

    with pytest.raises(ContractMismatchError, match='marker order is part of the contract'):
        fixture.build()()


def test_prod3_2_rejects_a_partition_of_another_artifact_than_its_run_record(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.partition = INFERENCE.stamp({k: v for k, v in fixture.partition.items() if k != 'artifact'},
                                        DEFAULT_CONTRACT)

    with pytest.raises(ValueError, match='artifact differs from its run record'):
        fixture.build()()


def test_prod3_2_rejects_a_partition_of_another_contract(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.shard = INFERENCE.stamp({k: v for k, v in fixture.shard.items() if k != 'artifact'}, DEFAULT_CONTRACT)
    fixture.partition = INFERENCE.stamp({k: v for k, v in fixture.partition.items() if k != 'artifact'},
                                        REORDERED.contract)

    with pytest.raises(ContractMismatchError, match='marker order is part of the contract'):
        fixture.build()()


def test_prod3_2_accepts_current_inference_records_of_its_contract(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.shard = INFERENCE.stamp({k: v for k, v in fixture.shard.items() if k != 'artifact'}, DEFAULT_CONTRACT)
    fixture.partition = INFERENCE.stamp({k: v for k, v in fixture.partition.items() if k != 'artifact'},
                                        DEFAULT_CONTRACT)

    assert fixture.build()().config.contract == DEFAULT_CONTRACT


@pytest.mark.parametrize('marker', ['brdu', 'edu'])
def test_prod3_2_rejects_a_partition_cutoff_that_differs_from_the_classifier_for_each_marker(
        tmp_path, monkeypatch, marker):
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.partition = {**fixture.partition, f'{marker}_cutoff': .6}

    with pytest.raises(ValueError, match='prediction provenance differs'):
        fixture.build()()
