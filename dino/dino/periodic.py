"""Shared XY intensity correction fitted with equal sampling across Z slabs."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import click
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import SplineTransformer

from dino.artifacts import file_sha256

TARGETS = ("brdu", "edu")
MARKER = b"dino_periodic_correction"


def periodic_basis(xy, period, spline):
    phase = np.remainder(np.asarray(xy) / period, 1.0)
    basis = spline.transform(phase)
    half = basis.shape[1] // 2
    return np.einsum("ij,ik->ijk", basis[:, :half], basis[:, half:]).reshape(len(xy), -1)


def correction_gain(xy, bundle):
    field = (
        periodic_basis(xy, bundle["period_xy_um"], bundle["periodic_spline"]) - bundle["periodic_center"]
    ) @ bundle["coefficient"]
    gain = np.exp(-field)
    if not np.isfinite(gain).all() or np.any(gain <= 0):
        raise ValueError("Periodic correction produced invalid gains")
    return gain


def balanced_sample(frame, edges, count, seed):
    """Sample without replacement within explicit half-open Z slabs."""
    edges = np.asarray(edges, dtype=float)
    if len(edges) < 3 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0) or count <= 0:
        raise ValueError("Require increasing finite Z edges and a positive sample count")
    if frame.label.duplicated().any():
        raise ValueError("Fit cell labels must be unique")
    z = frame.centroid_z.to_numpy()
    if not np.isfinite(z).all():
        raise ValueError("Z coordinates must be finite")
    selected = frame.loc[(z >= edges[0]) & (z < edges[-1])].copy()
    selected["z_slab"] = np.searchsorted(edges, selected.centroid_z, side="right") - 1
    counts = selected.groupby("z_slab").size().reindex(range(len(edges) - 1), fill_value=0)
    if counts.min() < count:
        raise ValueError(f"Insufficient cells for balanced Z sampling: {counts.to_dict()}")
    return (
        selected.groupby("z_slab", group_keys=False).sample(n=count, random_state=seed).reset_index(drop=True)
    )


def fit_field(frame, periods, folds):
    """Select a shared periodic field by original-tile-held-out intensity error."""
    periods = np.asarray(periods, dtype=float)
    if periods.shape != (2,) or not np.isfinite(periods).all() or np.any(periods <= 0):
        raise ValueError("XY periods must be finite and positive")
    columns = ["u", "v", "r_vz", "centroid_z"]
    numeric = frame[columns + ["x_um", "y_um", *[f"intensity_{t}_mean" for t in TARGETS]]]
    if not np.isfinite(numeric.to_numpy()).all() or frame.tile_index.isna().any():
        raise ValueError("Fit coordinates, intensities and tile groups must be present and finite")
    intensity = frame[[f"intensity_{t}_mean" for t in TARGETS]].to_numpy()
    if np.any(intensity <= 0):
        raise ValueError("Fit intensities must be positive")
    response = np.log(intensity)
    if np.any(np.var(response, axis=0) == 0):
        raise ValueError("Both targets must have varying intensities")
    spline = SplineTransformer(n_knots=6, extrapolation="linear").fit(frame[columns])
    values = spline.transform(frame[columns])
    width = values.shape[1] // 4
    anatomy = np.column_stack(
        [
            np.einsum("ij,ik->ijk", values[:, :width], values[:, width : 2 * width]).reshape(len(frame), -1),
            values[:, 2 * width :],
        ]
    )
    splits = list(StratifiedGroupKFold(n_splits=folds).split(frame, frame.z_slab, frame.tile_index))
    fold_ids = np.empty(len(frame), dtype=int)
    baseline = np.empty_like(response)
    for fold, (train, test) in enumerate(splits):
        fold_ids[test] = fold
        baseline[test] = Ridge(alpha=10).fit(anatomy[train], response[train]).predict(anatomy[test])
    if not pd.crosstab(frame.z_slab, fold_ids).gt(0).all().all():
        raise ValueError("Every Z slab must have cells in each held-out tile fold")
    axis = np.linspace(0, 1, 64, endpoint=False)
    grid = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    candidates = []
    for knots in (5, 9):
        periodic_spline = SplineTransformer(
            knots=np.tile(np.linspace(0, 1, knots)[:, None], (1, 2)),
            extrapolation="periodic",
        ).fit([[0.0, 0.0], [1.0, 1.0]])
        center = periodic_basis(grid * periods, periods, periodic_spline).mean(axis=0)
        periodic = periodic_basis(frame[["x_um", "y_um"]], periods, periodic_spline) - center
        design = np.column_stack([anatomy, periodic])
        prediction = np.empty_like(response)
        for train, test in splits:
            prediction[test] = Ridge(alpha=10).fit(design[train], response[train]).predict(design[test])
        mse = np.mean((response - prediction) ** 2, axis=0)
        candidates.append(
            (
                float(np.sum(mse / np.var(response, axis=0))),
                knots,
                periodic_spline,
                center,
                design,
                mse,
            )
        )
    _, knots, periodic_spline, center, design, mse = min(candidates, key=lambda c: c[0])
    fit = Ridge(alpha=10).fit(design, response)
    bundle = dict(
        period_xy_um=periods,
        periodic_spline=periodic_spline,
        periodic_center=center,
        coefficient=fit.coef_[:, anatomy.shape[1] :].T,
        targets=list(TARGETS),
    )
    metrics = dict(
        selected_knots=knots,
        ridge_alpha=10,
        baseline_mse=np.mean((response - baseline) ** 2, axis=0).tolist(),
        periodic_mse=mse.tolist(),
        candidate_mse={str(c[1]): c[5].tolist() for c in candidates},
    )
    return bundle, metrics, fold_ids


@click.group()
def main():
    """Fit and apply a shared periodic XY gain to intensity features."""


@main.command("fit")
@click.option(
    "--cells",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Population Parquet: label, centroid_z/y/x (pixels), u, v, r_vz (um), tile_index, intensity_brdu/edu_mean.",
)
@click.option("--period-xy-um", type=(float, float), required=True)
@click.option("--scale-zyx-um", type=(float, float, float), required=True)
@click.option(
    "--z-edge",
    type=float,
    multiple=True,
    required=True,
    help="Repeat in increasing order; pixel coordinates, last edge excluded.",
)
@click.option("--cells-per-z", type=click.IntRange(min=1), default=20000, show_default=True)
@click.option("--folds", type=click.IntRange(min=2), default=5)
@click.option("--seed", type=int, default=73)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
def fit_command(cells, period_xy_um, scale_zyx_um, z_edge, cells_per_z, folds, seed, out_dir):
    scale = np.asarray(scale_zyx_um)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Voxel spacing must be finite and positive")
    if out_dir.exists():
        raise FileExistsError(out_dir)
    parquet = pq.ParquetFile(cells)
    if MARKER in (parquet.schema_arrow.metadata or {}):
        raise ValueError("Fit requires uncorrected intensities")
    frame = balanced_sample(pd.read_parquet(cells), z_edge, cells_per_z, seed)
    frame["x_um"] = frame.centroid_x * scale[2]
    frame["y_um"] = frame.centroid_y * scale[1]
    bundle, metrics, frame["fold"] = fit_field(frame, period_xy_um, folds)
    bundle["scale_zyx_um"] = scale
    metrics.update(
        cells=str(cells.resolve()),
        cells_sha256=file_sha256(cells),
        cells_per_z=cells_per_z,
        z_edges_px=list(z_edge),
        scale_zyx_um=list(scale_zyx_um),
        period_xy_um=list(period_xy_um),
        seed=seed,
        folds=folds,
        fit_cells=len(frame),
        anatomical_covariates=["u", "v", "r_vz", "centroid_z"],
        r_vz_units="um",
    )
    out_dir.mkdir(parents=True)
    joblib.dump(bundle, out_dir / "model.joblib")
    frame.to_parquet(out_dir / "fit-cells.parquet", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    click.echo(str(out_dir / "model.joblib"))


@main.command("apply")
@click.option("--model", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--props", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--scale-zyx-um", type=(float, float, float), required=True)
@click.option(
    "--classifier",
    type=click.Path(exists=True, path_type=Path),
    help="Optional intensity-context classifier to bind to the corrected table.",
)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
def apply_command(model, props, scale_zyx_um, classifier, out_dir):
    """Correct all cells before kNN feature extraction; preserve classifier weights."""
    if out_dir.exists():
        raise FileExistsError(out_dir)
    scale = np.asarray(scale_zyx_um)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Voxel spacing must be finite and positive")
    bundle = joblib.load(model)
    if list(bundle["targets"]) != list(TARGETS):
        raise ValueError("Correction target order must be brdu, edu")
    if "scale_zyx_um" in bundle and not np.array_equal(bundle["scale_zyx_um"], scale):
        raise ValueError("Correction and props voxel spacings differ")
    parquet = pq.ParquetFile(props)
    metadata = dict(parquet.schema_arrow.metadata or {})
    if MARKER in metadata:
        raise ValueError("Intensity table is already periodically corrected")
    classifier_bundle = None
    if classifier is not None:
        from dino.intensity_context import MODEL_SCHEMA

        classifier_bundle = joblib.load(classifier)
        if (
            classifier_bundle["schema"] != MODEL_SCHEMA
            or Path(classifier_bundle["intensity_props"]).resolve() != props.resolve()
        ):
            raise ValueError("Classifier must use this uncorrected intensity table")
        if not np.array_equal(classifier_bundle["intensity_scale_zyx_um"], scale):
            raise ValueError("Classifier and props voxel spacings differ")
        if "periodic_correction" in classifier_bundle or "pixel_correction" in classifier_bundle:
            raise ValueError("Classifier already has correction")
    record = dict(
        model=str(model.resolve()),
        model_sha256=file_sha256(model),
        intensity_props=str(props.resolve()),
        intensity_sha256=file_sha256(props),
        scale_zyx_um=list(scale_zyx_um),
        scope="intensity features; persisted DINO embeddings unchanged",
    )
    metadata[MARKER] = json.dumps(record).encode()
    schema = parquet.schema_arrow.with_metadata(metadata)
    # Use float64 even when raw intensity statistics were integer-valued.
    columns = [
        f"intensity_{t}_{stat}"
        for t in TARGETS
        for stat in ("min", "mean", "max")
        if f"intensity_{t}_{stat}" in schema.names
    ]
    for target in TARGETS:
        if f"intensity_{target}_mean" not in columns:
            raise ValueError(f"Missing mean intensity for {target}")
    for column in columns:
        schema = schema.set(schema.get_field_index(column), pa.field(column, pa.float64()))
    out_dir.mkdir(parents=True)
    output = out_dir / "props-corrected.parquet"
    with pq.ParquetWriter(output, schema) as writer:
        for batch in parquet.iter_batches(batch_size=100000):
            table = pa.Table.from_batches([batch])
            xy = np.column_stack(
                [
                    table["centroid_x"].to_numpy() * scale[2],
                    table["centroid_y"].to_numpy() * scale[1],
                ]
            )
            gain = correction_gain(xy, bundle)
            for j, target in enumerate(TARGETS):
                for column in columns:
                    if column.startswith(f"intensity_{target}_"):
                        values = table[column].to_numpy() * gain[:, j]
                        if not np.isfinite(values).all() or np.any(values < 0):
                            raise ValueError(f"Invalid corrected intensity: {column}")
                        table = table.set_column(
                            table.schema.get_field_index(column),
                            column,
                            pa.array(values),
                        )
            writer.write_table(table.cast(schema))
    if classifier_bundle is not None:
        classifier_bundle["intensity_props"] = str(output.resolve())
        classifier_bundle["periodic_correction"] = record
        joblib.dump(classifier_bundle, out_dir / "model.joblib")
    (out_dir / "correction.json").write_text(json.dumps(record, indent=2) + "\n")
    click.echo(str(output))


@main.command("measure")
@click.option("--source-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--model",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Shared periodic gain model.",
)
@click.option(
    "--classifier",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Classifier bundle to bind to corrected pixels and measured properties.",
)
@click.option("--scale-zyx-um", type=(float, float, float), required=True)
@click.option("--workers", type=click.IntRange(min=1), default=8)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
def measure_command(source_dir, model, classifier, scale_zyx_um, workers, out_dir):
    """Measure lazy corrected pixels and bind the same correction for DINO."""
    from squisher_segment.segment.regionprops import measure_zarr

    from dino.intensity_context import MODEL_SCHEMA
    from dino.periodic_pixels import PeriodicPixels
    from dino.zarr_source import load_source, write_json_atomic

    source = load_source(source_dir)
    bundle = joblib.load(classifier)
    if bundle["schema"] != MODEL_SCHEMA or bundle["source_sha256"] != source.manifest["source_sha256"]:
        raise ValueError("Classifier must be an intensity-context model for this source")
    if "pixel_correction" in bundle or "periodic_correction" in bundle:
        raise ValueError("Classifier already has correction; supply the original classifier")
    if source.offset_zyx != (0, 0, 0):
        raise ValueError("Pixel-corrected regionprops requires aligned image and label grids")
    if not np.allclose(bundle["intensity_scale_zyx_um"], scale_zyx_um, rtol=1e-12, atol=0):
        raise ValueError("Classifier and correction voxel spacings differ")
    for target in TARGETS:
        transforms = source.manifest["channels"][target]["ome"]["coordinate_transformations"]
        scales = [entry["scale"] for entry in transforms if entry["type"] == "scale"]
        if len(scales) != 1 or not np.allclose(scales[0], scale_zyx_um, rtol=1e-12, atol=0):
            raise ValueError(f"Image and correction voxel spacings differ: {target}")
    correction = PeriodicPixels(str(model.resolve()), file_sha256(model), tuple(scale_zyx_um))
    record = dict(
        source_sha256=source.manifest["source_sha256"],
        classifier=str(classifier.resolve()),
        classifier_sha256=file_sha256(classifier),
        pixel_correction=correction.identity,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "run.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != record:
        raise ValueError("Output directory belongs to a different pixel correction run")
    if not manifest_path.exists() and any(out_dir.iterdir()):
        raise ValueError("Nonempty pixel correction output lacks a run manifest")
    props = out_dir / "props.parquet"
    output_model = out_dir / "model.joblib"
    completion_path = out_dir / "complete.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text())
        if any(completion.get(key) != value for key, value in record.items()):
            raise ValueError("Completion belongs to a different pixel correction run")
        if (
            not props.is_file()
            or not output_model.is_file()
            or completion.get("props_sha256") != file_sha256(props)
            or completion.get("model_sha256") != file_sha256(output_model)
        ):
            raise ValueError("Completed pixel correction artifacts have changed")
        click.echo(str(output_model))
        return
    if props.exists() or output_model.exists():
        raise ValueError("Published pixel correction artifacts lack completion; use a fresh output directory")
    write_json_atomic(record, manifest_path)
    measure_zarr(
        Path(source.manifest["labels"]["root"]),
        props,
        intensity_paths={target: Path(source.manifest["channels"][target]["root"]) for target in TARGETS},
        intensity_transform=correction,
        workers=workers,
    )
    parquet = pq.ParquetFile(props)
    metadata = dict(parquet.schema_arrow.metadata or {})
    metadata[MARKER] = json.dumps(record).encode()
    schema = parquet.schema_arrow.with_metadata(metadata)
    with tempfile.NamedTemporaryFile(dir=out_dir, suffix=".parquet", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with pq.ParquetWriter(temporary_path, schema) as writer:
            for batch in parquet.iter_batches(batch_size=100000):
                writer.write_batch(batch.replace_schema_metadata(metadata))
        os.replace(temporary_path, props)
    finally:
        temporary_path.unlink(missing_ok=True)
    bundle["intensity_props"] = str(props.resolve())
    bundle["pixel_correction"] = correction.identity
    joblib.dump(bundle, output_model)
    write_json_atomic(
        dict(
            **record,
            props_sha256=file_sha256(props),
            model_sha256=file_sha256(output_model),
        ),
        completion_path,
    )
    click.echo(str(output_model))
