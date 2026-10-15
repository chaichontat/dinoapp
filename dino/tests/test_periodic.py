import joblib
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from dino.periodic import main, balanced_sample, fit_field, correction_gain
from dino.intensity_context import IntensityContext


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
    bundle, metrics, fold = fit_field(sample, [1, 1], 3)
    xy = sample[["x_um", "y_um"]].to_numpy()
    gain = correction_gain(xy, bundle)
    np.testing.assert_allclose(gain, correction_gain(xy + [2, -3], bundle), atol=1e-12)
    truth = 0.3 * np.sin(2 * np.pi * xy[:, 0]) + 0.2 * np.cos(2 * np.pi * xy[:, 1])
    assert np.corrcoef(-np.log(gain[:, 0]), truth)[0, 1] > 0.98
    assert np.all(np.array(metrics["periodic_mse"]) < np.array(metrics["baseline_mse"]))
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
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.exception
    saved = joblib.load(out / "model.joblib")
    context = IntensityContext.load(
        out / "props-corrected.parquet", scale_zyx_um=np.ones(3)
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


def test_fit_cli_records_depth_samples_and_roundtrips(tmp_path):
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
            intensity_brdu_mean=np.exp(3 + 0.3 * np.sin(2 * np.pi * xyz[:, 0])),
            intensity_edu_mean=np.exp(4 + 0.2 * np.cos(2 * np.pi * xyz[:, 1])),
        )
    )
    cells = tmp_path / "cells.parquet"
    frame.to_parquet(cells)
    out = tmp_path / "fit"
    result = CliRunner().invoke(
        main,
        [
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
        ],
    )
    assert result.exit_code == 0, result.exception
    assert pd.read_parquet(out / "fit-cells.parquet").groupby(
        "z_slab"
    ).size().tolist() == [400, 400]
    assert correction_gain(
        np.array([[0.0, 0.0]]), joblib.load(out / "model.joblib")
    ).shape == (1, 2)


def measured_run(tmp_path):
    import zarr
    from dino.zarr_source import build_source_manifest, write_json_atomic
    from sklearn.preprocessing import SplineTransformer

    labels_path = tmp_path / 'labels.zarr'
    labels = zarr.open_array(labels_path, mode='w', shape=(2, 4, 4), chunks=(1, 4, 4), dtype='u4')
    labels[:] = 1
    channels = {}
    for name in ['brdu', 'edu', 'pi']:
        path = tmp_path / f'{name}.zarr'
        group = zarr.open_group(path, mode='w', zarr_format=2)
        array = group.create_array('0', shape=(2, 4, 4), chunks=(1, 4, 4), dtype='u2')
        array[:] = 1000
        group.attrs['multiscales'] = [dict(version='0.4', axes=[dict(name=n, type='space', unit='micrometer') for n in 'zyx'], datasets=[dict(path='0', coordinateTransformations=[dict(type='scale', scale=[1., 1., 1.]), dict(type='translation', translation=[0., 0., 0.])])])]
        channels[name] = path
    source = tmp_path / 'source'
    source.mkdir()
    manifest = build_source_manifest(labels_path=labels_path, channel_paths=channels, offset_zyx=(0, 0, 0), dataset='test', roi='test')
    write_json_atomic(manifest, source / 'source.json')
    spline = SplineTransformer(knots=np.tile(np.linspace(0, 1, 5)[:, None], (1, 2)), extrapolation='periodic').fit([[0., 0.], [1., 1.]])
    model = tmp_path / 'gain.joblib'
    joblib.dump(dict(period_xy_um=np.ones(2), periodic_spline=spline, periodic_center=np.zeros(16), coefficient=np.full((16, 2), np.log(2)), targets=['brdu', 'edu'], scale_zyx_um=np.ones(3)), model)
    classifier = tmp_path / 'classifier.joblib'
    joblib.dump(dict(schema='dino-knn-intensity-v1', source_sha256=manifest['source_sha256'], intensity_scale_zyx_um=[1, 1, 1]), classifier)
    out = tmp_path / 'measured'
    args = ['measure', '--source-dir', str(source), '--model', str(model), '--classifier', str(classifier), '--scale-zyx-um', '1', '1', '1', '--workers', '1', '--out-dir', str(out)]
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
