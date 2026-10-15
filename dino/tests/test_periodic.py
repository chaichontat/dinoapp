import json

import joblib
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from dino.periodic import main, balanced_sample, fit_field, correction_gain
from dino.intensity_context import IntensityContext
from dino_markers import DEFAULT_CONTRACT, ContractMismatchError, MarkerContract, MissingContractError

THREE = MarkerContract("dapi", ("edu", "brdu", "pcna"))


def classifier_bundle(contract=DEFAULT_CONTRACT, *, legacy=True, **fields):
    """A classifier bundle in the legacy form of the default contract, else the current form of `contract`."""
    schema = dict(schema="dino-knn-intensity-v1") if legacy else dict(schema="dino-knn-intensity-v2", marker_contract=contract.to_json())
    return {**schema, **fields}


def constant_field(path, targets, divisors, **extra):
    """A field whose gain divides each target's intensity by a constant: the periodic basis sums to one."""
    from sklearn.preprocessing import SplineTransformer

    bundle = dict(
        period_xy_um=np.ones(2),
        periodic_spline=SplineTransformer(
            knots=np.tile(np.linspace(0, 1, 5)[:, None], (1, 2)), extrapolation="periodic"
        ).fit([[0.0, 0.0], [1.0, 1.0]]),
        periodic_center=np.zeros(16),
        coefficient=np.column_stack([np.full(16, np.log(d)) for d in divisors]),
        targets=list(targets),
        **extra,
    )
    joblib.dump(bundle, path)


def test_shared_field_recovers_periodic_signal_across_depth():
    rng = np.random.default_rng(73)
    n = 3000
    x, y, z, u, v, r = rng.uniform(size=(6, n))
    field = 0.3 * np.sin(2 * np.pi * x) + 0.2 * np.cos(2 * np.pi * y)
    frame = pd.DataFrame(
        dict(
            label=np.arange(n) + 1,
            centroid_z=z * 2,
            x_um=x,
            y_um=y,
            u=u,
            v=v,
            r_vz=r,
            tile_index=np.arange(n) % 15,
            intensity_brdu_mean=np.exp(3 + u + field),
            intensity_edu_mean=np.exp(4 + v + field),
        )
    )
    sample = balanced_sample(frame, [0, 1, 2], 1000, 73)
    assert sample.groupby("z_slab").size().tolist() == [1000, 1000]
    assert sample.label.is_unique
    bundle, metrics, fold = fit_field(sample, [1, 1], 3, targets=("brdu", "edu"))
    xy = sample[["x_um", "y_um"]].to_numpy()
    gain = correction_gain(xy, bundle)
    np.testing.assert_allclose(gain, correction_gain(xy + [2, -3], bundle), atol=1e-12)
    truth = 0.3 * np.sin(2 * np.pi * xy[:, 0]) + 0.2 * np.cos(2 * np.pi * xy[:, 1])
    assert np.corrcoef(-np.log(gain[:, 0]), truth)[0, 1] > 0.98
    assert np.all(np.array(metrics["periodic_mse"]) < np.array(metrics["baseline_mse"]))
    assert bundle["targets"] == ["brdu", "edu"]
    for tile in sample.tile_index.unique():
        assert len(np.unique(fold[sample.tile_index == tile])) == 1


def test_apply_corrects_neighbors_and_binds_classifier_once(tmp_path):
    from sklearn.preprocessing import SplineTransformer

    spline = SplineTransformer(
        knots=np.tile(np.linspace(0, 1, 5)[:, None], (1, 2)), extrapolation="periodic"
    ).fit([[0.0, 0.0], [1.0, 1.0]])
    bundle = dict(
        period_xy_um=np.ones(2),
        periodic_spline=spline,
        periodic_center=np.zeros(16),
        coefficient=np.full((16, 2), np.log(2)),
        targets=["brdu", "edu"],
    )
    model = tmp_path / "gain.joblib"
    joblib.dump(bundle, model)
    props = tmp_path / "props.parquet"
    pd.DataFrame(
        dict(
            label=[1, 2, 3],
            centroid_z=[0, 1, 2],
            centroid_y=[0, 0, 0],
            centroid_x=[0, 0, 0],
            intensity_brdu_mean=[10, 20, 30],
            intensity_edu_mean=[20, 40, 60],
        )
    ).to_parquet(props)
    classifier = tmp_path / "classifier.joblib"
    joblib.dump(
        dict(
            schema="dino-knn-intensity-v1",
            intensity_props=str(props),
            intensity_scale_zyx_um=[1, 1, 1],
        ),
        classifier,
    )
    out = tmp_path / "corrected"
    args = [
        "apply",
        "--model",
        str(model),
        "--props",
        str(props),
        "--scale-zyx-um",
        "1",
        "1",
        "1",
        "--out-dir",
        str(out),
        "--classifier",
        str(classifier),
    ]
    inputs = {path: path.read_bytes() for path in (classifier, model, props)}
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.exception
    assert inputs == {path: path.read_bytes() for path in inputs}
    saved = joblib.load(out / "model.joblib")
    assert saved["schema"] == "dino-knn-intensity-v2"
    assert saved["marker_contract"] == DEFAULT_CONTRACT.to_json()
    correction = json.loads((out / "correction.json").read_text())
    assert "marker_contract" not in correction and "schema" not in correction
    assert saved["periodic_correction"] == correction
    context = IntensityContext.load(
        out / "props-corrected.parquet", scale_zyx_um=np.ones(3), markers=DEFAULT_CONTRACT.markers
    )
    np.testing.assert_allclose(context.means["brdu"], [5, 10, 15])
    np.testing.assert_allclose(
        context.features_for_labels(np.array([1]), neighbors=2)["brdu"],
        [[np.log(6), np.log(6 / 13.5)]],
    )
    assert saved["intensity_props"] == str(out / "props-corrected.parquet")
    result = CliRunner().invoke(
        main,
        [
            "apply",
            "--model",
            str(model),
            "--props",
            str(out / "props-corrected.parquet"),
            "--scale-zyx-um",
            "1",
            "1",
            "1",
            "--out-dir",
            str(tmp_path / "twice"),
        ],
    )
    assert result.exit_code != 0
    assert "already periodically corrected" in str(result.exception)


def fit_population(tmp_path, keys):
    rng = np.random.default_rng(21)
    n = 1200
    xyz = rng.uniform(size=(n, 3))
    frame = pd.DataFrame(
        dict(
            label=np.arange(n) + 1,
            centroid_x=xyz[:, 0],
            centroid_y=xyz[:, 1],
            centroid_z=xyz[:, 2] * 2,
            u=rng.uniform(size=n),
            v=rng.uniform(size=n),
            r_vz=rng.uniform(size=n),
            tile_index=np.arange(n) % 12,
            **{
                f"intensity_{key}_mean": np.exp(3 + index + 0.3 * np.sin(2 * np.pi * xyz[:, index % 2]))
                for index, key in enumerate(keys)
            },
        )
    )
    cells = tmp_path / "cells.parquet"
    frame.to_parquet(cells)
    return cells


def fit_arguments(cells, out, *extra):
    return [
        "fit",
        "--cells",
        str(cells),
        "--period-xy-um",
        "1",
        "1",
        "--scale-zyx-um",
        "1",
        "1",
        "1",
        "--z-edge",
        "0",
        "--z-edge",
        "1",
        "--z-edge",
        "2",
        "--cells-per-z",
        "400",
        "--folds",
        "3",
        "--out-dir",
        str(out),
        *extra,
    ]


def test_fit_cli_records_depth_samples_and_roundtrips(tmp_path):
    cells = fit_population(tmp_path, ("brdu", "edu"))
    out = tmp_path / "fit"
    result = CliRunner().invoke(main, fit_arguments(cells, out))
    assert result.exit_code == 0, result.exception
    assert pd.read_parquet(out / "fit-cells.parquet").groupby(
        "z_slab"
    ).size().tolist() == [400, 400]
    saved = joblib.load(out / "model.joblib")
    assert saved["targets"] == ["brdu", "edu"]
    assert correction_gain(np.array([[0.0, 0.0]]), saved).shape == (1, 2)


def test_fit_markers_define_the_targets_and_their_order(tmp_path):
    config = tmp_path / "markers.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "reference": {"key": "dapi", "label": "DAPI"},
                "markers": [
                    {"key": "edu", "label": "EdU", "color": "#00ff00"},
                    {"key": "brdu", "label": "BrdU", "color": "#ff00ff"},
                    {"key": "pcna", "label": "PCNA", "color": "#00ffff"},
                ],
                "presentation": {
                    "crop_panels": ["dapi", "edu", "brdu", "pcna"],
                    "negative_color": "#404040",
                    "uncalled_color": "#999999",
                },
            }
        )
    )
    cells = fit_population(tmp_path, THREE.markers)
    out = tmp_path / "fit"
    result = CliRunner().invoke(main, fit_arguments(cells, out, "--markers", str(config)))
    assert result.exit_code == 0, result.exception
    saved = joblib.load(out / "model.joblib")
    assert saved["targets"] == ["edu", "brdu", "pcna"]
    assert saved["coefficient"].shape[1] == 3
    assert correction_gain(np.array([[0.0, 0.0]]), saved).shape == (1, 3)
    assert "marker_contract" not in saved and "marker_contract" not in (out / "metrics.json").read_text()


def test_fit_without_the_markers_cells_column_fails_before_writing(tmp_path):
    cells = fit_population(tmp_path, ("brdu",))
    out = tmp_path / "fit"
    result = CliRunner().invoke(main, fit_arguments(cells, out))
    assert result.exit_code != 0
    assert not (out / "model.joblib").exists()


def measure_inputs(tmp_path, *, names=('brdu', 'edu', 'pi'), targets=('brdu', 'edu'), divisors=(2, 2), classifier_fields=None, contract=DEFAULT_CONTRACT):
    import zarr
    from dino.zarr_source import build_source_manifest, write_json_atomic

    labels_path = tmp_path / 'labels.zarr'
    labels = zarr.open_array(labels_path, mode='w', shape=(2, 4, 4), chunks=(1, 4, 4), dtype='u4')
    labels[:] = 1
    channels = {}
    for name in names:
        path = tmp_path / f'{name}.zarr'
        group = zarr.open_group(path, mode='w', zarr_format=2)
        array = group.create_array('0', shape=(2, 4, 4), chunks=(1, 4, 4), dtype='u2')
        array[:] = 1000
        group.attrs['multiscales'] = [dict(version='0.4', axes=[dict(name=n, type='space', unit='micrometer') for n in 'zyx'], datasets=[dict(path='0', coordinateTransformations=[dict(type='scale', scale=[1., 1., 1.]), dict(type='translation', translation=[0., 0., 0.])])])]
        channels[name] = path
    source = tmp_path / 'source'
    source.mkdir()
    manifest = build_source_manifest(labels_path=labels_path, channel_paths=channels, offset_zyx=(0, 0, 0), dataset='test', roi='test', required_channels=contract.channel_keys)
    write_json_atomic(manifest, source / 'source.json')
    model = tmp_path / 'gain.joblib'
    constant_field(model, targets, divisors, scale_zyx_um=np.ones(3))
    classifier = tmp_path / 'classifier.joblib'
    fields = dict(source_sha256=manifest['source_sha256'], intensity_scale_zyx_um=[1, 1, 1])
    joblib.dump(classifier_bundle(**fields) if classifier_fields is None else {**classifier_fields, **fields}, classifier)
    out = tmp_path / 'measured'
    args = ['measure', '--source-dir', str(source), '--model', str(model), '--classifier', str(classifier), '--scale-zyx-um', '1', '1', '1', '--workers', '1', '--out-dir', str(out)]
    return model, out, args


def measured_run(tmp_path, **options):
    model, out, args = measure_inputs(tmp_path, **options)
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.exception
    return model, out, args


@pytest.mark.parametrize("changed", ["props", "model", "run_fields"])
def test_measure_completed_reuse_preserves_artifacts_and_rejects_edits(tmp_path, changed):
    _, out, args = measured_run(tmp_path)
    before = {path.name: path.read_bytes() for path in out.iterdir() if path.is_file()}
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.exception
    assert before == {path.name: path.read_bytes() for path in out.iterdir() if path.is_file()}
    if changed == "props":
        props = out / 'props.parquet'
        frame = pd.read_parquet(props)
        frame['intensity_brdu_mean'] = 123.
        frame.to_parquet(props, index=False)
    elif changed == "model":
        classifier = joblib.load(out / 'model.joblib')
        classifier['intensity_props'] = 'changed.parquet'
        joblib.dump(classifier, out / 'model.joblib')
    else:
        import json

        complete = out / 'complete.json'
        record = json.loads(complete.read_text())
        record['classifier_sha256'] = 'changed'
        complete.write_text(json.dumps(record))
    completion_bytes = (out / 'complete.json').read_bytes()
    result = CliRunner().invoke(main, args)
    assert result.exit_code != 0
    assert (out / 'complete.json').read_bytes() == completion_bytes


def test_measured_props_cannot_be_corrected_again(tmp_path):
    import pyarrow.parquet as pq
    from dino.periodic import MARKER

    model, out, _ = measured_run(tmp_path)
    props = out / 'props.parquet'
    assert MARKER in (pq.ParquetFile(props).schema_arrow.metadata or {})
    np.testing.assert_allclose(pd.read_parquet(props).intensity_brdu_mean, [500.])
    for classifier in [None, out / 'model.joblib']:
        args = ['apply', '--model', str(model), '--props', str(props), '--scale-zyx-um', '1', '1', '1', '--out-dir', str(tmp_path / 'twice')]
        if classifier is not None:
            args += ['--classifier', str(classifier)]
        result = CliRunner().invoke(main, args)
        assert result.exit_code != 0
        assert 'already periodically corrected' in str(result.exception)


def test_apply_rejects_pixel_bound_classifier_with_unmarked_props(tmp_path):
    model, out, _ = measured_run(tmp_path)
    props = tmp_path / 'unmarked.parquet'
    pd.read_parquet(out / 'props.parquet').to_parquet(props, index=False)
    classifier = joblib.load(out / 'model.joblib')
    classifier['intensity_props'] = str(props)
    path = tmp_path / 'pixel-classifier.joblib'
    joblib.dump(classifier, path)
    result = CliRunner().invoke(main, ['apply', '--model', str(model), '--props', str(props), '--scale-zyx-um', '1', '1', '1', '--classifier', str(path), '--out-dir', str(tmp_path / 'twice')])
    assert result.exit_code != 0
    assert 'already has' in str(result.exception)


def test_measure_rejects_published_artifacts_without_completion(tmp_path):
    _, out, args = measured_run(tmp_path)
    (out / 'complete.json').rename(out / 'interrupted-complete.json')
    before = {path.name: path.read_bytes() for path in out.iterdir() if path.is_file()}
    result = CliRunner().invoke(main, args)
    assert result.exit_code != 0
    assert 'lack completion' in str(result.exception)
    assert before == {path.name: path.read_bytes() for path in out.iterdir() if path.is_file()}


def three_marker_props(path):
    pd.DataFrame(
        dict(
            label=[1, 2, 3],
            centroid_z=[0, 1, 2],
            centroid_y=[0, 0, 0],
            centroid_x=[0, 0, 0],
            intensity_edu_mean=[60, 120, 180],
            intensity_edu_max=[120, 240, 360],
            intensity_brdu_mean=[30, 60, 90],
            intensity_pcna_mean=[50, 100, 150],
        )
    ).to_parquet(path)


def apply_arguments(model, props, out, *classifier):
    args = ["apply", "--model", str(model), "--props", str(props), "--scale-zyx-um", "1", "1", "1", "--out-dir", str(out)]
    return args + (["--classifier", str(classifier[0])] if classifier else [])


def test_apply_uses_the_field_targets_for_a_three_marker_classifier(tmp_path):
    model, props, classifier, out = tmp_path / "gain.joblib", tmp_path / "props.parquet", tmp_path / "classifier.joblib", tmp_path / "corrected"
    constant_field(model, THREE.markers, (2, 3, 5))
    three_marker_props(props)
    joblib.dump(classifier_bundle(THREE, legacy=False, intensity_props=str(props), intensity_scale_zyx_um=[1, 1, 1]), classifier)
    before = classifier.read_bytes()
    result = CliRunner().invoke(main, apply_arguments(model, props, out, classifier))
    assert result.exit_code == 0, result.exception
    corrected = pd.read_parquet(out / "props-corrected.parquet")
    np.testing.assert_allclose(corrected.intensity_edu_mean, [30, 60, 90])
    np.testing.assert_allclose(corrected.intensity_edu_max, [60, 120, 180])
    np.testing.assert_allclose(corrected.intensity_brdu_mean, [10, 20, 30])
    np.testing.assert_allclose(corrected.intensity_pcna_mean, [10, 20, 30])
    saved = joblib.load(out / "model.joblib")
    assert (saved["schema"], saved["marker_contract"]) == ("dino-knn-intensity-v2", THREE.to_json())
    assert classifier.read_bytes() == before


def test_apply_corrects_each_target_only_by_its_own_columns(tmp_path):
    model, props, out = tmp_path / "gain.joblib", tmp_path / "props.parquet", tmp_path / "corrected"
    constant_field(model, ("edu", "edu_2"), (2, 3))
    pd.DataFrame(
        dict(label=[1], centroid_z=[0], centroid_y=[0], centroid_x=[0], intensity_edu_mean=[60.0], intensity_edu_2_mean=[60.0])
    ).to_parquet(props)
    result = CliRunner().invoke(main, apply_arguments(model, props, out))
    assert result.exit_code == 0, result.exception
    corrected = pd.read_parquet(out / "props-corrected.parquet")
    assert corrected.intensity_edu_mean.tolist() == pytest.approx([30.0])
    assert corrected.intensity_edu_2_mean.tolist() == pytest.approx([20.0])


@pytest.mark.parametrize(
    ("targets", "divisors", "message"),
    [
        (("edu", "edu"), (2, 3), "nonempty list of unique keys"),
        (("edu", "brdu"), (2,), "coefficient has 1 columns for 2 targets"),
    ],
)
def test_apply_rejects_an_invalid_field(tmp_path, targets, divisors, message):
    model, props, out = tmp_path / "gain.joblib", tmp_path / "props.parquet", tmp_path / "corrected"
    constant_field(model, targets, divisors)
    three_marker_props(props)
    result = CliRunner().invoke(main, apply_arguments(model, props, out))
    assert isinstance(result.exception, ValueError)
    assert message in str(result.exception)
    assert not out.exists()


@pytest.mark.parametrize(
    ("classifier_contract", "targets"),
    [
        (MarkerContract("pi", ("edu", "brdu")), ("brdu", "edu")),
        (DEFAULT_CONTRACT, ("brdu", "edu", "pcna")),
    ],
)
def test_apply_rejects_a_classifier_whose_marker_order_differs_from_the_field(tmp_path, classifier_contract, targets):
    model, props, classifier, out = tmp_path / "gain.joblib", tmp_path / "props.parquet", tmp_path / "classifier.joblib", tmp_path / "corrected"
    constant_field(model, targets, (2,) * len(targets))
    three_marker_props(props)
    joblib.dump(classifier_bundle(classifier_contract, legacy=False, intensity_props=str(props), intensity_scale_zyx_um=[1, 1, 1]), classifier)
    result = CliRunner().invoke(main, apply_arguments(model, props, out, classifier))
    assert isinstance(result.exception, ContractMismatchError)
    assert "marker order is part of the contract" in str(result.exception)
    assert not out.exists()


def test_apply_requires_the_classifier_to_carry_its_contract(tmp_path):
    model, props, classifier, out = tmp_path / "gain.joblib", tmp_path / "props.parquet", tmp_path / "classifier.joblib", tmp_path / "corrected"
    constant_field(model, ("brdu", "edu"), (2, 2))
    three_marker_props(props)
    joblib.dump(dict(schema="dino-knn-intensity-v2", intensity_props=str(props), intensity_scale_zyx_um=[1, 1, 1]), classifier)
    result = CliRunner().invoke(main, apply_arguments(model, props, out, classifier))
    assert isinstance(result.exception, MissingContractError)


def test_measure_binds_the_field_targets_to_a_three_marker_classifier(tmp_path):
    names = ("edu", "brdu", "pcna", "dapi")
    classifier = classifier_bundle(THREE, legacy=False)
    model, out, _ = measured_run(tmp_path, names=names, targets=THREE.markers, divisors=(2, 4, 5), classifier_fields=classifier, contract=THREE)
    props = pd.read_parquet(out / "props.parquet").iloc[0]
    assert [props[f"intensity_{key}_mean"] for key in THREE.markers] == pytest.approx([500.0, 250.0, 200.0])
    assert "intensity_dapi_mean" not in props
    saved = joblib.load(out / "model.joblib")
    assert (saved["schema"], saved["marker_contract"]) == ("dino-knn-intensity-v2", THREE.to_json())
    assert saved["pixel_correction"]["model"] == str(model.resolve())
    assert not {"schema", "marker_contract"} & set(json.loads((out / "run.json").read_text()))
    assert not {"schema", "marker_contract"} & set(json.loads((out / "complete.json").read_text()))


def test_measure_writes_a_current_default_classifier_and_leaves_its_inputs(tmp_path):
    model, out, args = measured_run(tmp_path)
    classifier = tmp_path / "classifier.joblib"
    assert joblib.load(classifier)["schema"] == "dino-knn-intensity-v1"
    saved = joblib.load(out / "model.joblib")
    assert (saved["schema"], saved["marker_contract"]) == ("dino-knn-intensity-v2", DEFAULT_CONTRACT.to_json())
    assert "marker_contract" not in joblib.load(model) and "schema" not in joblib.load(model)
    run, complete = (json.loads((out / name).read_text()) for name in ("run.json", "complete.json"))
    assert not {"schema", "marker_contract"} & (set(run) | set(complete))
    assert set(run) == {"source_sha256", "classifier", "classifier_sha256", "pixel_correction"}


def test_measure_rejects_a_classifier_whose_marker_order_differs_from_the_field(tmp_path):
    _, out, args = measure_inputs(
        tmp_path, targets=("brdu", "edu"), classifier_fields=classifier_bundle(MarkerContract("pi", ("edu", "brdu")), legacy=False)
    )
    result = CliRunner().invoke(main, args)
    assert isinstance(result.exception, ContractMismatchError)
    assert "marker order is part of the contract" in str(result.exception)
    assert not out.exists()
