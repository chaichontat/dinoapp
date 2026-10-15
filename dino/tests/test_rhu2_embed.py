import json

import numpy as np
import pandas as pd
import pytest

from dino.rhu2_embed import CHANNELS, check_partition, finalize, partition_rows, read_tile
from dino.rhu2_train import native_crop
from dino.zarr_source import cell_dino_images


def test_tiled_inputs_match_direct_crops_with_borders_and_half_pixel_centroids():
    rng = np.random.default_rng(42)
    masks = np.zeros((4, 2200, 2200), dtype=np.uint32)
    pixels = rng.integers(0, 20000, (4, 2200, 2200, 3), dtype=np.uint16)
    rows = pd.DataFrame({'cell': ['r|3', 'r|4', 'r|5'], 'label': [3, 4, 5],
                         'x': [1.5, 2051.5, 2198.5], 'y': [2.5, 2052.5, 2198.5]})
    for row in rows.itertuples():
        x, y = int(np.rint(row.x)), int(np.rint(row.y))
        masks[2, max(0, y - 3):y + 4, max(0, x - 3):x + 4] = row.label
    keys = ['edu', 'pi', 'brdu']
    seen = []
    for _, group in partition_rows(rows):
        local, tile_pixels, tile_masks = read_tile(group, pixels, masks, 100)
        for original, shifted in zip(group.itertuples(), local.itertuples(), strict=True):
            direct, plane = native_crop(original, pixels, masks, keys, 100)
            tiled, tile_plane = native_crop(shifted, tile_pixels, tile_masks, keys, 100)
            assert plane == tile_plane
            np.testing.assert_array_equal(direct.labels, tiled.labels)
            a = cell_dino_images(direct, input_channels=4, channel_names_by_target=CHANNELS, highpass_sigma_px=20)
            b = cell_dino_images(tiled, input_channels=4, channel_names_by_target=CHANNELS, highpass_sigma_px=20)
            for target in CHANNELS:
                np.testing.assert_array_equal(a[target], b[target])
                assert not a[target][2:].any()
            np.testing.assert_array_equal(a['brdu'][0], a['edu'][0])
            seen.append(original.cell)
    assert sorted(seen) == sorted(rows.cell)


def test_resume_rejects_wrong_identity_or_incomplete_features(tmp_path):
    path = tmp_path / 'part.npz'
    group = pd.DataFrame({'cell': ['r|9', 'r|15']})
    data = {'cell': group.cell.to_numpy(dtype=str), 'plane_z': np.array([1, 2]),
            'brdu': np.ones((2, 1024)), 'edu': np.ones((2, 1024))}
    np.savez(path, **data)
    check_partition(path, group)
    with pytest.raises(ValueError, match='identities changed'):
        check_partition(path, group.iloc[::-1])
    data['edu'][1, 1] = np.nan
    np.savez(path, **data)
    with pytest.raises(ValueError, match='Incomplete edu'):
        check_partition(path, group)


def test_finalize_requires_both_shards_and_rejects_overlap(tmp_path):
    root = tmp_path / 'cortical_embeddings'
    root.mkdir()
    (root / 'manifest.json').write_text(json.dumps({'cells': 3}))
    shards = [{'status': 'complete', 'cells': count, 'shard_index': i, 'shard_count': 2,
               'partitions': [{'path': f'{i}.npz', 'cells': count}]} for i, count in enumerate([1, 2])]
    (root / 'complete-shard-0.json').write_text(json.dumps(shards[0]))
    with pytest.raises(FileNotFoundError):
        finalize(tmp_path, 2)
    (root / 'complete-shard-1.json').write_text(json.dumps(shards[1]))
    finalize(tmp_path, 2)
    assert json.loads((root / 'complete.json').read_text())['cells'] == 3
    shards[1]['partitions'][0]['path'] = '0.npz'
    (root / 'complete-shard-1.json').write_text(json.dumps(shards[1]))
    with pytest.raises(ValueError, match='Overlapping'):
        finalize(tmp_path, 2)


@pytest.mark.parametrize('precision', ['float32', 'bfloat16'])
def test_embedding_run_uses_recorded_precision(tmp_path, monkeypatch, precision):
    import torch
    import zarr
    import dino.rhu2_embed as module
    from dino.zarr_model import feature_dtype

    class Encoder(torch.nn.Module):
        def forward(self, batch):
            assert batch.dtype == feature_dtype(precision)
            return torch.ones((len(batch), 1024), dtype=batch.dtype)

    cpu = torch.device('cpu')
    monkeypatch.setattr(module.torch, 'device', lambda _: cpu)
    monkeypatch.setattr(module.torch.cuda, 'get_device_name', lambda _: 'CPU fixture')
    monkeypatch.setattr(module, 'load_cell_dino_model', lambda **_: Encoder())
    monkeypatch.setattr(module, 'build_cell_transform', lambda **_: lambda x: x)
    masks = zarr.open_array(tmp_path / 'labels.zarr', mode='w', shape=(1, 8, 8), dtype='uint32')
    masks[0, 3:5, 3:5] = 1
    fused = zarr.open_array(tmp_path / 'fused.zarr', mode='w', shape=(1, 8, 8, 3), dtype='uint16')
    fused[:] = 10
    fused.attrs['key'] = ['pi', 'brdu', 'edu']
    rows = pd.DataFrame({'cell': ['d:r|1'], 'dataset': 'd', 'roi': 'r', 'label': [1], 'x': [4.], 'y': [4.]})
    rows.to_parquet(tmp_path / 'cortical_context.parquet')
    config = {'dinov2_repo': 'unused', 'feature_extractor': 'fixture', 'weights': 'unused',
              'feature_precision': precision, 'resize_size': 4, 'crop_eval_size': 4, 'crop_size': 4,
              'input_channels': 4, 'highpass_sigma_px': 0, 'mask_erosion_px': 0}
    (tmp_path / 'prepared.json').write_text(json.dumps({'cohort': 'fixture', 'encoder': config, 'rois': [
        {'key': 'd:r', 'fused': str(tmp_path / 'fused.zarr'), 'segmentation': str(tmp_path / 'labels.zarr'),
         'channel_keys': ['pi', 'brdu', 'edu']}]}))
    module.run(tmp_path, 1, 1, None, 0, 1)
    with np.load(tmp_path / 'cortical_embeddings/d__r/y0000_x0000.npz') as saved:
        np.testing.assert_allclose(saved['brdu'], 1 / 32)
        np.testing.assert_allclose(saved['edu'], 1 / 32)
