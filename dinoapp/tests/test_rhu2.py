"""RHU2 seeded sampling, ROI identity, and native pixel contracts."""
import base64
from io import BytesIO
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import zarr
from dino_markers import (
    DEFAULT_CONTRACT,
    SPATIAL_REVIEW,
    ContractMismatchError,
    MarkerConfigError,
    MarkerContract,
    config_to_json,
    parse_config,
    rhu2_probability_column,
)

from dinoapp.native import pick_cell, render_field
from dinoapp.rhu2 import FieldCropSource, RoiSource, SpatialSource, sample_location
from dinoapp.sources import AnnotationFiles

THREE = parse_config({
    'version': 1, 'reference': {'key': 'dapi', 'label': 'DAPI'},
    'markers': [{'key': 'edu', 'label': 'EdU', 'color': '#00ff00'},
                {'key': 'brdu', 'label': 'BrdU', 'color': '#ff00ff'},
                {'key': 'pcna', 'label': 'PCNA', 'color': '#0000ff'}],
    'presentation': {'crop_panels': ['dapi', 'pcna', 'edu', 'brdu'], 'negative_color': [.25, .25, .25],
                     'uncalled_color': '#999999'},
}, source='test')


def build_catalog(root, *, contract=DEFAULT_CONTRACT, fused_keys=('edu', 'pi', 'brdu'),
                  probabilities=None, version=1):
    """Write a two-ROI snapshot; `fused_keys` is the deliberately permuted channel order."""
    markers = contract.markers
    values = {'brdu': [.9, .2], 'edu': [.1, .8], 'pcna': [.7, .4]} if probabilities is None else probabilities
    rois, membership, branches = [], [], []
    for roi in ('1', '2'):
        key = f'section:{roi}'
        raw = np.zeros((3, 120, 120, len(fused_keys)), dtype=np.uint16)
        for index, channel in enumerate(fused_keys):
            raw[..., index] = {'pi': 2000, 'dapi': 2000, 'brdu': 3000, 'edu': 1000, 'pcna': 4000}[channel]
        labels = np.zeros((3, 120, 120), dtype=np.uint32)
        labels[:, 40:65, 40:65] = 1
        labels[:, 65:85, 65:85] = 2
        labels[:, 85:95, 85:95] = 3  # Segmented, outside the cortical cohort.
        fused_path, labels_path = root / f'fused-{roi}', root / f'labels-{roi}'
        fused = zarr.create_array(str(fused_path), data=raw)
        fused.attrs['key'] = list(fused_keys)
        zarr.create_array(str(labels_path), data=labels)
        rows = pd.DataFrame(dict(cell=[f'{key}|1', f'{key}|2'], label=[1, 2],
                                 x=[50., 75.], y=[50., 75.], z=[1., 1.],
                                 **{rhu2_probability_column(marker): values[marker] for marker in markers}))
        rows.to_parquet(root / f'{roi}.parquet', index=False)
        for cell, name in zip(rows.cell, ['cortex', 'cortex2'], strict=True):
            membership.append(dict(cell=cell, roi_key=key, ccf_adjusted=name))
            branches.append(dict(key=f'{key}:{name}', roi_key=key, dataset='section', roi=roi,
                                 ccf_adjusted=name, cells=1))
        rois.append(dict(key=key, dataset='section', roi=roi, cells=2, table=f'{roi}.parquet',
                         fused=str(fused_path), segmentation=str(labels_path), shape=[3, 120, 120],
                         channel_keys=list(fused_keys), pixel_size_um=.5))
    pd.DataFrame(columns=['cell', *markers]).to_csv(root / 'labels.csv', index=False)
    pd.DataFrame(membership).to_parquet(root / 'ccf_branches.parquet', index=False)
    branches.append(dict(key='outside:1:cortex', roi_key='outside:1', dataset='outside', roi='1',
                         ccf_adjusted='cortex', cells=0))
    fields = dict(rois=rois, branches=branches, branch_table='ccf_branches.parquet',
                  probability_columns=[rhu2_probability_column(marker) for marker in markers],
                  decision_thresholds=dict.fromkeys(markers, .5),
                  display_max=32768., uncertainty_log_vmax=4.)
    manifest = (SPATIAL_REVIEW.stamp(fields, contract) if version == 2
                else dict(artifact='rhu2-spatial-review-v1', **fields))
    (root / 'spatial.json').write_text(json.dumps(manifest))
    return manifest


@pytest.fixture
def catalog(tmp_path):
    build_catalog(tmp_path)
    return SpatialSource(tmp_path)


@pytest.fixture
def three_marker_catalog(tmp_path):
    (tmp_path / 'markers.json').write_text(json.dumps(config_to_json(THREE)))
    build_catalog(tmp_path, contract=THREE.contract, fused_keys=('pcna', 'dapi', 'edu', 'brdu'), version=2)
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
    assert list(a.images) == ['pi', 'brdu', 'edu']
    assert np.all(a.images['brdu'] == 3000)
    assert np.all(a.images['edu'] == 1000)
    assert np.all(a.images['pi'] == 2000)
    assert first.sampled_field(0, 7, 0, 40.).token != a.token
    annotations = AnnotationFiles(first.annotation_paths, first.default_annotation_path,
                                  contract=first.config.contract)
    calls = first.field_calls(a, annotations)
    assert dict(zip(a.ids, calls)) == {0: 4, 1: 2, 2: 1}
    for mode in ('Cell identities', 'Model uncertainty'):
        encoded = render_field(a, calls, config=first.config, mode=mode, selected=None, border_alpha=.35,
                               boundary_alpha=0, contrast=first.contrast,
                               uncertainty=first.field_uncertainty(a), uncertainty_log_vmax=4.)
        with Image.open(BytesIO(base64.b64decode(encoded.split(',', 1)[1]))) as image:
            image.load()
            assert image.size == (100, 100)


def test_annotation_override_and_undo_cannot_cross_rois(catalog):
    first, second = catalog.roi_source(0), catalog.roi_source(1)
    a, b = [source.sampled_field(0, 0, 0, 50.) for source in (first, second)]
    annotations = AnnotationFiles(first.annotation_paths, first.default_annotation_path,
                                  contract=first.config.contract)
    edit = annotations.assign(first.cell_id(1), (0, 1), expected=None)
    assert first.field_calls(a, annotations)[np.flatnonzero(a.ids == 1)[0]] == 1
    assert second.field_calls(b, annotations)[np.flatnonzero(b.ids == 1)[0]] == 2
    assert set(annotations.values) == {'section:1|1'}
    annotations.undo(edit)
    assert first.field_calls(a, annotations)[np.flatnonzero(a.ids == 1)[0]] == 2
    assert not AnnotationFiles(first.annotation_paths, first.default_annotation_path,
                                contract=first.config.contract).values


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
        RoiSource(catalog.root, meta, catalog.manifest, config=catalog.config)


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
    annotations = AnnotationFiles(source.annotation_paths, source.default_annotation_path,
                                  contract=source.config.contract)
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
    annotations = AnnotationFiles((labels,), labels, contract=catalog.config.contract)
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
    monkeypatch.setattr(rhu2_app, "get_source", lambda root, revision: catalog)
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


def test_spatial_source_resolves_its_config_and_shares_it_with_each_roi(catalog):
    assert catalog.config.contract == DEFAULT_CONTRACT
    roi = catalog.roi_source(0)
    assert roi.config is catalog.config
    crop_source = FieldCropSource(roi, (32768., 3000., 1000.), "opened-target", "snapshot")
    assert crop_source.config is catalog.config


def test_a_field_crop_source_built_after_a_config_change_carries_the_new_config(tmp_path, three_marker_catalog):
    other = tmp_path / 'default'
    other.mkdir()
    build_catalog(other)
    default_roi = SpatialSource(other).roi_source(0)
    three_roi = three_marker_catalog.roi_source(0)

    before = FieldCropSource(default_roi, (1., 1., 1.), "target", "revision")
    after = FieldCropSource(three_roi, (1., 1., 1., 1.), "target", "revision")

    assert before.config.contract == DEFAULT_CONTRACT
    assert after.config.contract == THREE.contract


def test_snapshot_v2_of_the_default_contract_loads(tmp_path):
    build_catalog(tmp_path, version=2)
    assert SpatialSource(tmp_path).config.contract == DEFAULT_CONTRACT


def test_v1_snapshots_must_not_carry_a_contract(tmp_path):
    manifest = build_catalog(tmp_path)
    (tmp_path / 'spatial.json').write_text(json.dumps({**manifest, 'marker_contract': DEFAULT_CONTRACT.to_json()}))
    with pytest.raises(MarkerConfigError, match='must not carry marker_contract'):
        SpatialSource(tmp_path)


def test_snapshot_contract_differing_from_the_directory_configuration_is_rejected(tmp_path):
    (tmp_path / 'markers.json').write_text(json.dumps(config_to_json(THREE)))
    build_catalog(tmp_path, contract=MarkerContract('pi', ('edu', 'brdu')), version=2)
    with pytest.raises(ContractMismatchError, match='marker order is part of the contract'):
        SpatialSource(tmp_path)


@pytest.mark.parametrize('columns', [['edu_prob', 'brdu_prob'], ['brdu_prob'], ['brdu_prob', 'edu_prob', 'pcna_prob']])
def test_probability_columns_must_follow_the_contract(tmp_path, columns):
    manifest = build_catalog(tmp_path)
    (tmp_path / 'spatial.json').write_text(json.dumps({**manifest, 'probability_columns': columns}))
    with pytest.raises(ValueError, match='probability_columns'):
        SpatialSource(tmp_path)


@pytest.mark.parametrize('thresholds', [{'brdu': .5}, {'brdu': .5, 'edu': .5, 'pcna': .5}, {'brdu': .5, 'pcna': .5}])
def test_decision_thresholds_must_name_the_contract_markers(tmp_path, thresholds):
    manifest = build_catalog(tmp_path)
    (tmp_path / 'spatial.json').write_text(json.dumps({**manifest, 'decision_thresholds': thresholds}))
    with pytest.raises(ValueError, match='decision_thresholds'):
        SpatialSource(tmp_path)


def test_roi_with_every_probability_at_its_threshold_loads_with_the_snapshot_scale(tmp_path):
    build_catalog(tmp_path, probabilities={'brdu': [.5, .5], 'edu': [.5, .5]})
    source = SpatialSource(tmp_path).roi_source(0)
    assert source.uncertainty_log_vmax == 4.
    assert np.isinf(source.uncertainty[[1, 2]]).all()
    np.testing.assert_array_equal(source.calls, [4, 3, 3])


def test_three_marker_roi_encodes_calls_in_contract_order(three_marker_catalog):
    source = three_marker_catalog.roi_source(0)
    # EdU, BrdU, PCNA probabilities: cell 1 (.1, .9, .7) -> 0b011, cell 2 (.8, .2, .4) -> 0b100.
    np.testing.assert_array_equal(source.calls, [8, 3, 4])
    assert source.contrast == dict.fromkeys(('dapi', 'edu', 'brdu', 'pcna'), (0.0, 32768.))
    assert source.probability_caption(1) == (
        'EdU probability 0.1000 · BrdU probability 0.9000 · PCNA probability 0.7000')


def test_three_marker_fields_and_crops_use_the_contract_channels(three_marker_catalog):
    source = three_marker_catalog.roi_source(0)

    field = source.sampled_field(0, 0, 0, 50.)

    assert list(field.images) == ['dapi', 'edu', 'brdu', 'pcna']
    assert np.all(field.images['pcna'] == 4000) and np.all(field.images['dapi'] == 2000)
    image, plane = source.cell_crop(1, (1000., 2000., 3000., 4000.))
    with Image.open(BytesIO(image)) as crop:
        crop.load()
        assert crop.width == crop.height * 4  # DAPI, PCNA, EdU, BrdU
    with pytest.raises(ValueError, match='4 finite positive values'):
        source.cell_crop(1, (1000., 2000., 3000.))
    encoded = render_field(field, source.field_calls(field, SimpleNamespace(values={})), config=source.config,
                           mode='Cell identities', selected=None, border_alpha=.35, boundary_alpha=0,
                           contrast=source.contrast)
    with Image.open(BytesIO(base64.b64decode(encoded.split(',', 1)[1]))) as image:
        assert image.size == (100, 100)


def test_roi_rejects_fused_channels_that_differ_from_the_contract(tmp_path):
    build_catalog(tmp_path, fused_keys=('edu', 'pi', 'brdu', 'pcna'))
    source = SpatialSource(tmp_path)
    with pytest.raises(ValueError, match='Native channel identities changed'):
        source.roi_source(0)


def test_prepare_writes_a_v2_snapshot_of_the_default_contract_that_loads(tmp_path):
    import anndata as ad
    from dinoapp.rhu2_prepare import prepare

    review, release = tmp_path / 'review', tmp_path / 'release'
    review.mkdir()
    (release / 'cohort').mkdir(parents=True)
    sources, cells, rois = {}, [], []
    for roi in ('1', '2'):
        key = f'section:{roi}'
        fused = zarr.create_array(str(tmp_path / f'fused-{roi}'), data=np.zeros((3, 40, 40, 3), dtype=np.uint16))
        fused.attrs['key'] = ['edu', 'pi', 'brdu']
        zarr.create_array(str(tmp_path / f'segmentation-{roi}'), data=np.zeros((3, 40, 40), dtype=np.uint32))
        sources[key] = dict(fused=str(tmp_path / f'fused-{roi}'), segmentation=str(tmp_path / f'segmentation-{roi}'))
        landmarks = release / 'datasets' / 'section' / 'rois' / roi / 'ccf/landmarks.json'
        landmarks.parent.mkdir(parents=True)
        landmarks.write_text(json.dumps({'sample_voxel_xy_um': .5}))
        rois.append(dict(dataset='section', roi=roi, branches=[dict(ccf_adjusted='cortex')]))
        for label, (brdu, edu) in enumerate([(.9, .1), (.2, .5)], start=1):
            cells.append({'cell': f'{key}|{label}', 'dataset': 'section', 'roi': roi, 'ccf_adjusted': 'cortex',
                          'x': 10. * label, 'y': 10. * label, 'z': 1., 'brdu_prob': brdu, 'edu_prob': edu})
    (review / 'complete.json').write_text(json.dumps({'sources': sources}))
    (release / 'manifest.json').write_text(json.dumps({'rois': rois}))
    obs = pd.DataFrame(cells).set_index('cell')
    ad.AnnData(X=np.zeros((len(obs), 1), dtype=np.float32), obs=obs).write_h5ad(release / 'cohort/cells.h5ad')

    prepare(review, release)

    manifest = json.loads((review / 'spatial.json').read_text())
    assert manifest['artifact'] == 'rhu2-spatial-review-v2'
    assert manifest['marker_contract'] == DEFAULT_CONTRACT.to_json()
    assert manifest['probability_columns'] == ['brdu_prob', 'edu_prob']
    assert manifest['decision_thresholds'] == {'brdu': .5, 'edu': .5}
    margins = np.minimum(np.abs(obs.brdu_prob.to_numpy() - .5), np.abs(obs.edu_prob.to_numpy() - .5))
    finite = np.log1p(1 / margins[margins > 0])
    assert manifest['uncertainty_log_vmax'] == pytest.approx(float(np.quantile(finite, .999)))
    source = SpatialSource(review)
    assert source.config.contract == DEFAULT_CONTRACT
    assert source.roi_source(0).calls.tolist() == [4, 2, 1]
