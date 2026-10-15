"""Nested spatial-block validation and application of smooth XY subtraction."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from dino.spatial_subtraction import ACCURACY_CONTRACT, SpatialSubtraction, fit_spatial_subtraction
from dino.spatial_evaluation import (
    GRID,
    SEED,
    geometry as physical_geometry,
    inner_splits,
    metrics,
    outside,
    select_penalty,
    validate_location,
)


@dataclass(frozen=True)
class WorkflowConfig:
    output_dir: Path
    prefix: str
    solver_absolute_tolerance: float = 5e-4
    solver_relative_tolerance: float = 1e-6
    solver_max_iter: int = 60000
    solver_time_limit: float = 240

    def __post_init__(self) -> None:
        validate_location(self.output_dir, self.prefix)
        tolerances = [self.solver_absolute_tolerance, self.solver_relative_tolerance]
        if not np.isfinite(tolerances).all() or min(tolerances) <= 0:
            raise ValueError("Solver tolerances must be finite and positive")
        if not isinstance(self.solver_max_iter, int) or self.solver_max_iter < 1:
            raise ValueError("Solver iteration budget must be a positive integer")
        if not np.isfinite(self.solver_time_limit) or self.solver_time_limit <= 0:
            raise ValueError("Solver time limit must be finite and positive")


def geometry(config: WorkflowConfig, frame: pd.DataFrame) -> pd.DataFrame:
    return physical_geometry(frame, samples_path=config.output_dir / f"{config.prefix}_cortex_samples.parquet")


def validate_checkpoint_precision(model: SpatialSubtraction, config: WorkflowConfig) -> None:
    """Reject cached fits made at looser precision than explicitly requested."""
    if (
        model.diagnostics["tolerance"] > config.solver_absolute_tolerance
        or model.diagnostics["relative_tolerance"] > config.solver_relative_tolerance
    ):
        raise ValueError(
            "Saved checkpoint solver precision is looser than requested; use its recorded settings or a fresh output prefix"
        )


def fit_checkpoint(
    config: WorkflowConfig, frame: pd.DataFrame, penalty: float, stem: str
) -> SpatialSubtraction:
    if len(frame) > 50000:
        raise ValueError("Saved training roster exceeds the 50,000-observation cap")
    path = config.output_dir / f"{stem}.joblib"
    if path.exists():
        model = SpatialSubtraction.load(path)
        assert model.penalty == penalty and model.diagnostics["samples"] == len(frame)
        assert np.array_equal(
            model.domain_xy_um,
            np.column_stack([frame[["x_um", "y_um"]].min(), frame[["x_um", "y_um"]].max()]),
        )
        if model.diagnostics.get("accuracy_contract") == ACCURACY_CONTRACT:
            validate_checkpoint_precision(model, config)
            return model
    model = fit_spatial_subtraction(
        frame[["x_um", "y_um"]].to_numpy(),
        frame.dapi.to_numpy(),
        frame.brdu.to_numpy(),
        penalty=penalty,
        tolerance=config.solver_absolute_tolerance,
        relative_tolerance=config.solver_relative_tolerance,
        max_iter=config.solver_max_iter,
        time_limit=config.solver_time_limit,
    )
    model.save(path)
    (config.output_dir / f"{stem}_solver.json").write_text(
        json.dumps(
            {
                **model.diagnostics,
                "domain_xy_um": model.domain_xy_um.tolist(),
                "penalty": penalty,
                "quantile": 0.05,
                "knots_xy": [3, 4],
                "intensity_scale": 1000,
                "offset_coefficient_units": "raw target / 1000",
                "alpha_coefficient_units": "target/reference",
                "extrapolation": "constant on each training-domain edge",
                "reference_centering": "none",
            },
            indent=2,
        )
        + "\n"
    )
    return model


def candidate(job: tuple) -> dict:
    config, train, test, penalty, inner, label = job
    stem = f"{config.prefix}_smooth_{label}_i{inner}_l{str(penalty).replace('.', '_')}"
    checkpoint = config.output_dir / f"{stem}_metrics.json"
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved.get("accuracy_contract") == ACCURACY_CONTRACT:
            fit_checkpoint(config, train, penalty, stem)
            return saved
    model = fit_checkpoint(config, train, penalty, stem)
    xy = test[["x_um", "y_um"]].to_numpy()
    predicted = model.predict(xy, test.dapi.to_numpy())
    report = {
        "penalty": penalty,
        "inner_fold": inner,
        **metrics(test.brdu.to_numpy(), predicted),
        "fit_samples": len(train),
        "outside_training_domain": int(outside(model, xy).sum()),
        "accuracy_contract": ACCURACY_CONTRACT,
    }
    test[["z_index", "y_index", "x_index"]].assign(predicted=predicted).to_parquet(
        config.output_dir / f"{stem}_predictions.parquet", index=False
    )
    checkpoint.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"{label}: lambda={penalty},inner={inner},loss={report['pinball_loss']:.4f},iterations={model.diagnostics['iterations']}",
        flush=True,
    )
    return report


def tune(config: WorkflowConfig, frame: pd.DataFrame, label: str) -> float:
    selection = config.output_dir / f"{config.prefix}_smooth_{label}_selection.json"
    if selection.exists():
        for penalty in GRID:
            for inner in range(3):
                stem = f"{config.prefix}_smooth_{label}_i{inner}_l{str(penalty).replace('.', '_')}"
                validate_checkpoint_precision(
                    SpatialSubtraction.load(config.output_dir / f"{stem}.joblib"), config
                )
        return float(json.loads(selection.read_text())["selected_penalty"])
    splits = inner_splits(frame)
    inner_assignment = frame[["z_index", "y_index", "x_index", "x_block", "y_block", "z_band"]].copy()
    inner_assignment["inner_fold"] = -1
    for inner, (_train, test) in enumerate(splits):
        inner_assignment.iloc[test, inner_assignment.columns.get_loc("inner_fold")] = inner
    assert inner_assignment.groupby(["x_block", "y_block"]).inner_fold.nunique().max() == 1
    inner_assignment.to_parquet(
        config.output_dir / f"{config.prefix}_smooth_{label}_inner_folds.parquet", index=False
    )
    jobs = [
        (config, frame.iloc[train_idx], frame.iloc[test_idx], penalty, inner, label)
        for penalty in GRID
        for inner, (train_idx, test_idx) in enumerate(splits)
    ]
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(candidate, jobs))
    table = pd.DataFrame(rows)
    table.to_csv(config.output_dir / f"{config.prefix}_smooth_{label}_tuning.tsv", sep="\t", index=False)
    chosen, scores = select_penalty(table)
    selection.write_text(
        json.dumps(
            {
                "selected_penalty": chosen,
                "weighted_inner_losses": scores,
                "selection": "minimum 3-fold spatially held-out pinball loss; exact ties favor larger penalty",
                "preprocessing": "outer-training trimming fixed; no additional inner trimming; spline domain fitted on inner training only",
                "seed": SEED,
                "knots_xy": [3, 4],
                "samples": len(frame),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{label}: selected lambda={chosen},scores={scores}", flush=True)
    return chosen


def benchmark(config: WorkflowConfig) -> None:
    data = geometry(
        config, pd.read_parquet(config.output_dir / f"{config.prefix}_cortex_global_training_samples.parquet")
    ).sample(n=3000, random_state=932)
    data.to_parquet(config.output_dir / f"{config.prefix}_smooth_benchmark_training.parquet", index=False)
    model = fit_checkpoint(config, data, 0.01, f"{config.prefix}_smooth_benchmark")
    (config.output_dir / f"{config.prefix}_smooth_benchmark_record.json").write_text(
        json.dumps(
            {
                **model.diagnostics,
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "model": "q=.05 cubic3x4XYknots; nonnegativealpha; quadraticsecond differences onalphaandbeta/1000",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(model.diagnostics), flush=True)


def run(config: WorkflowConfig) -> None:
    started = time.perf_counter()
    final_train = geometry(
        config, pd.read_parquet(config.output_dir / f"{config.prefix}_cortex_global_training_samples.parquet")
    )
    penalty = tune(config, final_train, "final")
    final = fit_checkpoint(config, final_train, penalty, f"{config.prefix}_smooth_final_model")
    report = {
        "quantile": 0.05,
        "knots_xy": [3, 4],
        "basis_per_field": 30,
        "penalty_grid": GRID,
        "final_penalty": penalty,
        "final_domain_xy_um": final.domain_xy_um.tolist(),
        "final_solver": final.diagnostics,
        "training": str(config.output_dir / f"{config.prefix}_cortex_global_training_samples.parquet"),
        "outer_training_pattern": f"{config.prefix}_cortex_cv_fold<N>.training.parquet",
        "outer_test_pattern": f"{config.prefix}_cortex_cv_fold<N>.predictions.parquet",
        "selection": "nested 3-fold spatial block CV inside each outer training set; final penalty separately selected inside global training",
        "penalty_units": "mean pinball in target/1000 + lambda/2*(alpha second differences squared + (beta/1000) second differences squared)",
        "solver_absolute_tolerance": config.solver_absolute_tolerance,
        "solver_relative_tolerance": config.solver_relative_tolerance,
        "solver_max_iter": config.solver_max_iter,
        "solver_time_limit": config.solver_time_limit,
        "accuracy_contract": ACCURACY_CONTRACT,
        "extrapolation": "constant at training XY edges",
        "wall_seconds": time.perf_counter() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (config.output_dir / f"{config.prefix}_smooth_record.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("final model ready", flush=True)
    outer_predictions, outer_metrics, regional_metrics, depth_metrics = [], [], [], []
    for fold in range(4):
        label = f"outer{fold}"
        train = geometry(
            config,
            pd.read_parquet(config.output_dir / f"{config.prefix}_cortex_cv_fold{fold}.training.parquet"),
        )
        test = geometry(
            config,
            pd.read_parquet(config.output_dir / f"{config.prefix}_cortex_cv_fold{fold}.predictions.parquet"),
        )
        penalty = tune(config, train, label)
        model = fit_checkpoint(config, train, penalty, f"{config.prefix}_smooth_{label}_model")
        xy = test[["x_um", "y_um"]].to_numpy()
        test["smooth_predicted"] = model.predict(xy, test.dapi.to_numpy())
        test["outside_smooth_training_domain"] = outside(model, xy)
        test.to_parquet(
            config.output_dir / f"{config.prefix}_smooth_{label}_predictions.parquet", index=False
        )
        reports = []
        for name, column in [("global", "global_predicted"), ("smooth", "smooth_predicted")]:
            reports.append(
                {
                    "fold": fold,
                    "model": name,
                    "penalty": penalty,
                    "outside_training_domain": int(test.outside_smooth_training_domain.sum()),
                    **metrics(test.brdu.to_numpy(), test[column].to_numpy()),
                }
            )
            for keys, summaries in [(["x_block", "y_block"], regional_metrics), (["z_band"], depth_metrics)]:
                for key, group in test.groupby(keys):
                    summaries.append(
                        {
                            "fold": fold,
                            "model": name,
                            **dict(zip(keys, key)),
                            **metrics(group.brdu.to_numpy(), group[column].to_numpy()),
                        }
                    )
        pd.DataFrame(reports).to_csv(
            config.output_dir / f"{config.prefix}_smooth_{label}_metrics.tsv", sep="\t", index=False
        )
        outer_predictions.append(test)
        outer_metrics.extend(reports)
        print(
            f"outer{fold}: complete lambda={penalty}; global={reports[0]['pinball_loss']:.4f},smooth={reports[1]['pinball_loss']:.4f}",
            flush=True,
        )
    pd.concat(outer_predictions).to_parquet(
        config.output_dir / f"{config.prefix}_smooth_heldout_predictions.parquet", index=False
    )
    pd.DataFrame(outer_metrics).to_csv(
        config.output_dir / f"{config.prefix}_smooth_heldout_metrics.tsv", sep="\t", index=False
    )
    pd.DataFrame(regional_metrics).to_csv(
        config.output_dir / f"{config.prefix}_smooth_heldout_regions.tsv", sep="\t", index=False
    )
    pd.DataFrame(depth_metrics).to_csv(
        config.output_dir / f"{config.prefix}_smooth_heldout_depth.tsv", sep="\t", index=False
    )
    report["final_phase_seconds"] = report.pop("wall_seconds")
    report["wall_seconds"] = time.perf_counter() - started
    report["main_process_max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["largest_child_process_max_rss_kib"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    (config.output_dir / f"{config.prefix}_smooth_record.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("all nested validation and final model complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["benchmark", "run"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--solver-absolute-tolerance", type=float, default=5e-4)
    parser.add_argument("--solver-relative-tolerance", type=float, default=1e-6)
    parser.add_argument("--solver-max-iterations", type=int, default=60000)
    parser.add_argument("--solver-time-limit", type=float, default=240)
    args = parser.parse_args()
    config = WorkflowConfig(
        output_dir=args.output_dir.resolve(),
        prefix=args.prefix,
        solver_absolute_tolerance=args.solver_absolute_tolerance,
        solver_relative_tolerance=args.solver_relative_tolerance,
        solver_max_iter=args.solver_max_iterations,
        solver_time_limit=args.solver_time_limit,
    )
    if args.mode == "benchmark":
        benchmark(config)
    else:
        run(config)


if __name__ == "__main__":
    main()
