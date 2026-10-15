"""RHU2 seeded sampling, ROI identity, and native pixel contracts."""
import base64
from io import BytesIO
import json

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import zarr

from dinoapp.native import pick_cell, render_field
from dinoapp.rhu2 import RoiSource, SpatialSource, sample_location
from dinoapp.sources import AnnotationFiles


@pytest.fixture
def catalog(tmp_path):
    rois, membership, branches = [], [], []
    for roi in ('1', '2'):
        key = f'section:{roi}'
        raw = np.zeros((3, 120, 120, 3), dtype=np.uint16)
        raw[..., 0], raw[..., 1], raw[..., 2] = 1000, 2000, 3000
        labels = np.zeros((3, 120, 120), dtype=np.uint32)
        labels[:, 40:65, 40:65] = 1
        labels[:, 65:85, 65:85] = 2
        labels[:, 85:95, 85:95] = 3  # Segmented, outside the cortical cohort.
        fused_path, labels_path = tmp_path / f'fused-{roi}', tmp_path / f'labels-{roi}'
        fused = zarr.create_array(str(fused_path), data=raw)
        fused.attrs['key'] = ['edu', 'pi', 'brdu']  # Deliberately permuted.
        zarr.create_array(str(labels_path), data=labels)
        rows = pd.DataFrame(dict(cell=[f'{key}|1', f'{key}|2'], label=[1, 2],
                                 x=[50., 75.], y=[50., 75.], z=[1., 1.],
                                 brdu_prob=[.9, .2], edu_prob=[.1, .8]))
        rows.to_parquet(tmp_path / f'{roi}.parquet', index=False)
        for cell, name in zip(rows.cell, ['cortex', 'cortex2'], strict=True):
            membership.append(dict(cell=cell, roi_key=key, ccf_adjusted=name))
            branches.append(dict(key=f'{key}:{name}', roi_key=key, dataset='section', roi=roi,
                                 ccf_adjusted=name, cells=1))
        rois.append(dict(key=key, dataset='section', roi=roi, cells=2, table=f'{roi}.parquet',
                         fused=str(fused_path), segmentation=str(labels_path), shape=[3, 120, 120],
                         channel_keys=['edu', 'pi', 'brdu'], pixel_size_um=.5))
    pd.DataFrame(columns=['cell', 'brdu', 'edu']).to_csv(tmp_path / 'labels.csv', index=False)
    pd.DataFrame(membership).to_parquet(tmp_path / 'ccf_branches.parquet', index=False)
    branches.append(dict(key='outside:1:cortex', roi_key='outside:1', dataset='outside', roi='1',
                         ccf_adjusted='cortex', cells=0))
    manifest = dict(artifact='rhu2-spatial-review-v1', rois=rois, branches=branches,
                    branch_table='ccf_branches.parquet',
                    decision_thresholds={'brdu': .5, 'edu': .5},
                    display_max=32768., uncertainty_log_vmax=4.)
    (tmp_path / 'spatial.json').write_text(json.dumps(manifest))
    return SpatialSource(tmp_path)


def test_seed_replays_roi_and_cell_without_weighting_by_roi_size():
    first = [sample_location((1, 1000), 23, n) for n in range(1000)]
    assert first == [sample_location((1, 1000), 23, n) for n in range(1000)]
    assert first != [sample_location((1, 1000), 24, n) for n in range(1000)]
    assert 450 < sum(roi == 0 for roi, _ in first) < 550
    assert all(0 <= cell < (1, 1000)[roi] for roi, cell in first)


@pytest.mark.parametrize('counts,seed,position', [((), 0, 0), ((0,), 0, 0), ((2,), -1, 0), ((2,), 0, -1)])
def test_rejects_invalid_sampling(counts, seed, position):
    with pytest.raises(ValueError):
        sample_location(counts, seed, position)


def test_fields_use_recorded_channels_geometry_and_distinct_roi_tokens(catalog):
    first, second = catalog.roi_source(0), catalog.roi_source(1)
    a, b = [source.sampled_field(0, 7, 0, 50.) for source in (first, second)]
    assert a.token != b.token
    assert a.mask.shape == (100, 100)
    assert a.z_index == 1 and a.z_um is None
    np.testing.assert_array_equal(a.scale_xy_um, [.5, .5])
    assert np.all(a.images['BrdU'] == 3000)
    assert np.all(a.images['EdU'] == 1000)
    assert np.all(a.images['561'] == 2000)
    assert first.sampled_field(0, 7, 0, 40.).token != a.token
    annotations = AnnotationFiles(first.annotation_paths, first.default_annotation_path)
    calls = first.field_calls(a, annotations)
    assert dict(zip(a.ids, calls)) == {0: 4, 1: 2, 2: 1}
    for mode in ('Cell identities', 'Model uncertainty'):
        encoded = render_field(a, calls, mode=mode, selected=None, border_alpha=.35,
                               boundary_alpha=0, contrast=first.contrast,
                               uncertainty=first.field_uncertainty(a), uncertainty_log_vmax=4.)
        with Image.open(BytesIO(base64.b64decode(encoded.split(',', 1)[1]))) as image:
            image.load()
            assert image.size == (100, 100)


def test_annotation_override_and_undo_cannot_cross_rois(catalog):
    first, second = catalog.roi_source(0), catalog.roi_source(1)
    a, b = [source.sampled_field(0, 0, 0, 50.) for source in (first, second)]
    annotations = AnnotationFiles(first.annotation_paths, first.default_annotation_path)
    edit = annotations.assign(first.cell_id(1), (0, 1), expected=None)
    assert first.field_calls(a, annotations)[np.flatnonzero(a.ids == 1)[0]] == 1
    assert second.field_calls(b, annotations)[np.flatnonzero(b.ids == 1)[0]] == 2
    assert set(annotations.values) == {'section:1|1'}
    annotations.undo(edit)
    assert first.field_calls(a, annotations)[np.flatnonzero(a.ids == 1)[0]] == 2
    assert not AnnotationFiles(first.annotation_paths, first.default_annotation_path).values


@pytest.mark.parametrize('problem', ['probability', 'identity', 'coordinates', 'channels'])
def test_rejects_invalid_source_contract(catalog, problem):
    meta = catalog.rois[0]
    path = catalog.root / meta['table']
    rows = pd.read_parquet(path)
    if problem == 'probability':
        rows.loc[0, 'brdu_prob'] = 1.1
    elif problem == 'identity':
        rows.loc[0, 'cell'] = 'other:1|1'
    elif problem == 'coordinates':
        rows.loc[0, 'x'] = 1000
    else:
        zarr.open_array(meta['fused'], mode='a').attrs['key'] = ['brdu', 'edu', 'pi']
    rows.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        RoiSource(catalog.root, meta, catalog.manifest)


def test_cell_detail_reads_native_mask_and_probability(catalog):
    source = catalog.roi_source(0)
    crop, plane = source.cell_crop(1)
    with Image.open(BytesIO(crop)) as image:
        image.load()
        assert image.width > image.height
    assert plane == 0
    assert source.probability_caption(1) == 'BrdU probability 0.9000 · EdU probability 0.1000'
    assert 'No saved call' in source.probability_caption(3)


def test_non_cortical_cells_cannot_be_selected_or_colored_from_saved_labels(catalog):
    source = catalog.roi_source(0)
    annotations = AnnotationFiles(source.annotation_paths, source.default_annotation_path)
    annotations.assign(source.cell_id(3), (1, 1), expected=None)
    field = source.sampled_field(0, 0, 0, 50.)
    assert pick_cell(field.mask, .9, .9) is None
    assert 3 not in field.ids
    assert pick_cell(field.mask, .5, .5) == 1
    assert np.isfinite(source.field_uncertainty(field)[field.ids > 0]).all()
    assert source.cell_id(3) in annotations.values  # Preserve old labels without exposing them.


def test_explicit_branch_sampling_never_crosses_ccf_adjusted_or_roi(catalog):
    for index, branch in enumerate(catalog.branches):
        if branch['cells'] == 0:
            with pytest.raises(ValueError, match='no cells'):
                catalog.sample_branch(index, 0)
            continue
        for position in [0, 1, 5]:
            roi, cell_index = catalog.sample_branch(index, position)
            assert roi.key == branch['roi_key']
            assert roi.rows.iloc[cell_index].ccf_adjusted == branch['ccf_adjusted']
            assert catalog.sample_branch(index, position) == (roi, cell_index)
    with pytest.raises(ValueError, match='Invalid branch index'):
        catalog.sample_branch(-1, 0)
    with pytest.raises(ValueError, match='nonnegative'):
        catalog.sample_branch(1, -1)


def test_branch_membership_rejects_mismatched_cell_identities(catalog):
    path = catalog.root / 'ccf_branches.parquet'
    rows = pd.read_parquet(path)
    rows.loc[0, 'cell'] = 'section:1|999'
    rows.to_parquet(path, index=False)
    with pytest.raises(ValueError, match='Branch membership identities disagree'):
        catalog.roi_source(0)


def test_annotation_crop_uses_independent_marker_contrast_and_cache_keys(catalog):
    source = catalog.roi_source(0)
    original, plane = source.cell_crop(1, (32768., 32768., 32768.))
    brdu, brdu_plane = source.cell_crop(1, (32768., 3000., 32768.))
    edu, edu_plane = source.cell_crop(1, (32768., 32768., 1000.))
    assert plane == brdu_plane == edu_plane
    images = [np.asarray(Image.open(BytesIO(data))) for data in [original, brdu, edu]]
    width = images[0].shape[1]
    # The annotation plot orders its panels PI, EdU, BrdU.
    np.testing.assert_array_equal(images[0][:, :2 * width // 3], images[1][:, :2 * width // 3])
    assert not np.array_equal(images[0][:, 2 * width // 3:], images[1][:, 2 * width // 3:])
    np.testing.assert_array_equal(images[0][:, :width // 3], images[2][:, :width // 3])
    np.testing.assert_array_equal(images[0][:, 2 * width // 3:], images[2][:, 2 * width // 3:])
    assert not np.array_equal(images[0][:, width // 3:2 * width // 3], images[2][:, width // 3:2 * width // 3])
    assert source.cell_crop(1, (32768., 32768., 32768.)) == (original, plane)
    assert source.cell_crop(1) == (original, plane)
    with pytest.raises(ValueError, match='finite positive'):
        source.cell_crop(1, (32768., 0., 1000.))


def test_branch_annotation_counts_track_saved_cells_within_each_cortical_branch(catalog):
    labels = catalog.root / 'labels.csv'
    annotations = AnnotationFiles((labels,), labels)
    original_branches = catalog.branches
    assert catalog.branch_annotation_counts(()) == (0, 0, 0, 0, 0)
    for cell in ['section:1|1', 'section:2|2', 'section:1|3', 'outside:1|1']:
        annotations.assign(cell, (0, 0), expected=None)
    assert catalog.branch_annotation_counts(tuple(sorted(annotations.values))) == (0, 1, 0, 0, 1)
    edit = annotations.assign('section:1|2', (1, 1), expected=None)
    assert catalog.branch_annotation_counts(tuple(sorted(annotations.values))) == (0, 1, 1, 0, 1)
    annotations.undo(edit)
    assert catalog.branch_annotation_counts(tuple(sorted(annotations.values))) == (0, 1, 0, 0, 1)
    assert catalog.branches == original_branches


@pytest.mark.parametrize("cell", ["other:1|1", "section:1|01", "section:1|3", "section:1|0"])
def test_spatial_resolution_rejects_wrong_identity_and_excluded_cells(catalog, cell):
    with pytest.raises(ValueError, match="identity|cohort"):
        catalog.resolve_cell(cell)


def test_roi_crop_queue_preserves_contrast_and_locates_actual_crop_plane(catalog):
    from dinoapp.rhu2 import FieldCropSource
    first, label = catalog.resolve_cell("section:1|1")
    second, second_label = catalog.resolve_cell("section:2|1")
    assert first is not second and label == second_label == 1
    limits = (32768., 3000., 1000.)
    source = FieldCropSource(first, limits, "opened-target", "snapshot")
    crop = source.crop("section:1|1", "combined")
    assert crop.image == first.cell_crop(1, limits)[0]
    located = first.field_at_cell(label, 50.)
    assert located.z_index == first.cell_crop(1)[1] == 0
    assert located.seed is None and located.position is None
    assert located.roi_key == "section:1"
    assert 3 not in located.ids
    with pytest.raises(ValueError, match="cohort"):
        source.crop("section:2|1", "combined")
    assert not (catalog.root / "cell-review").exists()


def test_rhu2_workspace_round_trip_restores_origin_roi_and_contrast(catalog, monkeypatch):
    from PIL import Image
    from streamlit.testing.v1 import AppTest
    from dinoapp import app, rhu2_app

    for metadata in catalog.rois:
        metadata["pixel_size_um"] = 2.
    Image.new("RGB", (8, 8)).save(catalog.root / "crop.png")
    (catalog.root / "index.csv").write_text("cell,rank,image_path\nsection:1|1,0,crop.png\n")
    monkeypatch.setenv("DINO_REVIEW_SOURCE", "Barrage")
    monkeypatch.setenv("DINO_REVIEW_BARRAGE_DIR", str(catalog.root))
    app.st.cache_resource.clear()
    monkeypatch.setattr(rhu2_app, "get_source", lambda root, mtime: catalog)
    review = AppTest.from_file(app.__file__).run(timeout=20)
    assert not review.exception
    context, = review.session_state.review_contexts.values()
    origin_roi, origin = context["workspace"]["active_field"]
    before = (catalog.root / "labels.csv").read_bytes()
    def press(label):
        next(button for button in review.button if button.label == label).click().run()
        assert not review.exception
    press("Review cells in this field")
    queue = context["workspace"]["queue"]
    crop_source = queue["source"]
    assert all(cell.startswith(origin_roi.key + "|") for cell in queue["cells"])
    assert crop_source.contrast == (32768., 3000., 1000.)
    press("Next")
    press("Show in field")
    located_roi, located = context["workspace"]["located_field"]
    assert located_roi.key == origin_roi.key
    assert located.z_index == 0 and located.seed is None
    press("Return to cell review")
    assert context["workspace"]["queue"]["position"] == 1
    assert context["workspace"]["queue"]["source"].contrast == crop_source.contrast
    press("Return to previous view")
    returned_roi, returned = context["workspace"]["active_field"]
    assert returned_roi.key == origin_roi.key
    np.testing.assert_array_equal(returned.mask, origin.mask)
    np.testing.assert_array_equal(returned.origin_xy_um, origin.origin_xy_um)
    assert returned.z_index == origin.z_index == 1
    assert not context["history"]
    assert (catalog.root / "labels.csv").read_bytes() == before
    assert not list(catalog.root.rglob("changes-*.jsonl"))
