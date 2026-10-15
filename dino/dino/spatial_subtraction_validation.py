"""Validate saved nested smooth subtraction artifacts without refitting."""

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

from dino.spatial_subtraction import ACCURACY_CONTRACT, INTENSITY_SCALE, SpatialSubtraction, _basis, _objective, _roughness
from dino.spatial_evaluation import (
    validate_location,
    geometry,
    inner_splits,
    metrics,
    outside,
    select_penalty,
)

KEY = ["z_index", "y_index", "x_index"]


def validate_fit(model: SpatialSubtraction, train: pd.DataFrame, *, label: str, penalty: float) -> dict:
    """Check a saved fit against its exact training roster and acceptance gate."""
    np.testing.assert_array_equal(
        model.domain_xy_um,
        np.column_stack([train[["x_um", "y_um"]].min(), train[["x_um", "y_um"]].max()]),
    )
    assert model.penalty == penalty and model.quantile == 0.05
    assert np.isfinite(model.alpha_coefficients).all() and (model.alpha_coefficients >= 0).all()
    assert np.isfinite(model.beta_scaled_coefficients).all()
    assert model.diagnostics["samples"] == len(train) <= 50000
    assert model.diagnostics["status"] == "solved" and model.diagnostics["accuracy_contract"] == ACCURACY_CONTRACT
    assert np.isfinite(
        [
            model.diagnostics["raw_primal_feasibility"],
            model.diagnostics["relative_projected_objective_gap"],
            model.diagnostics["projected_objective_sum_scaled"],
            model.diagnostics["dual_objective_sum_scaled"],
        ]
    ).all()
    assert (
        0 <= model.diagnostics["raw_primal_feasibility"] <= 1
        and 0 <= model.diagnostics["relative_projected_objective_gap"] <= 1e-5
    )
    basis = _basis(train[["x_um", "y_um"]].to_numpy(), model.splines)
    prediction = (basis @ model.alpha_coefficients) * train.dapi.to_numpy() / INTENSITY_SCALE + basis @ model.beta_scaled_coefficients
    objective = _objective(
        train.brdu.to_numpy() / INTENSITY_SCALE - prediction,
        model.alpha_coefficients,
        model.beta_scaled_coefficients,
        _roughness(model.splines),
        penalty=model.penalty,
        quantile=model.quantile,
    )
    np.testing.assert_allclose(objective, model.diagnostics["projected_objective_sum_scaled"], rtol=1e-10, atol=1e-10)
    dual_objective = model.diagnostics["dual_objective_sum_scaled"]
    relative_gap = abs(objective - dual_objective) / max(1, abs(objective), abs(dual_objective))
    assert relative_gap <= 1e-5
    return {
        "label": label,
        "raw_primal": model.diagnostics["raw_primal_feasibility"],
        "relative_gap": relative_gap,
        "raw_mean_gap": (objective - dual_objective)
        * 1000
        / model.diagnostics["samples"],
    }


def validate(output_dir: Path, prefix: str, *, destination_dir: Path | None = None) -> dict:
    validate_location(output_dir, prefix)
    destination = output_dir if destination_dir is None else destination_dir
    if not destination.is_dir():
        raise ValueError("Validation destination must be an existing directory")
    samples = pd.read_parquet(output_dir / f"{prefix}_cortex_samples.parquet")
    labels = ["outer0", "outer1", "outer2", "outer3", "final"]
    numerical = []
    all_training, selections, outer_predictions = {}, {}, []
    for label in labels:
        path = (
            f"{prefix}_cortex_global_training_samples.parquet"
            if label == "final"
            else f"{prefix}_cortex_cv_fold{label[-1]}.training.parquet"
        )
        train = geometry(pd.read_parquet(output_dir / path), samples_path=output_dir / f"{prefix}_cortex_samples.parquet")
        assert np.isfinite(train[["brdu", "dapi"]].to_numpy()).all()
        assert (train[["brdu", "dapi"]].to_numpy() > 0).all()
        all_training[label] = train
        assignments = pd.read_parquet(output_dir / f"{prefix}_smooth_{label}_inner_folds.parquet")
        assert np.array_equal(train[KEY], assignments[KEY])
        assert assignments.groupby(["x_block", "y_block"]).inner_fold.nunique().max() == 1
        np.testing.assert_array_equal(
            train[["x_block", "y_block", "z_band"]], assignments[["x_block", "y_block", "z_band"]]
        )
        expected_folds = np.full(len(train), -1)
        for inner, (_fit, test) in enumerate(inner_splits(train)):
            expected_folds[test] = inner
        np.testing.assert_array_equal(expected_folds, assignments.inner_fold)
        tuning = pd.read_csv(output_dir / f"{prefix}_smooth_{label}_tuning.tsv", sep="\t")
        assert len(tuning) == 9
        assert set(zip(tuning.penalty, tuning.inner_fold)) == {
            (p, i) for p in [0.001, 0.01, 0.1] for i in range(3)
        }
        chosen, scores = select_penalty(tuning)
        selection = json.loads((output_dir / f"{prefix}_smooth_{label}_selection.json").read_text())
        assert selection["selected_penalty"] == chosen
        np.testing.assert_allclose(
            list(scores.values()),
            [selection["weighted_inner_losses"][key] for key in scores],
            rtol=0,
            atol=1e-10,
        )
        selections[label] = chosen
        for penalty in [0.001, 0.01, 0.1]:
            for inner in range(3):
                subset = train[assignments.inner_fold != inner]
                test = train[assignments.inner_fold == inner]
                stem = f"{prefix}_smooth_{label}_i{inner}_l{str(penalty).replace('.', '_')}"
                model = SpatialSubtraction.load(output_dir / f"{stem}.joblib")
                numerical.append(validate_fit(model, subset, label=stem, penalty=penalty))
                prediction = pd.read_parquet(output_dir / f"{stem}_predictions.parquet")
                assert np.array_equal(test[KEY], prediction[KEY])
                assert np.isfinite(prediction.predicted).all()
                np.testing.assert_allclose(
                    model.predict(test[["x_um", "y_um"]].to_numpy(), test.dapi.to_numpy()),
                    prediction.predicted,
                    atol=1e-9,
                    rtol=0,
                )
                row = tuning[(tuning.penalty == penalty) & (tuning.inner_fold == inner)].iloc[0]
                for key, value in metrics(test.brdu.to_numpy(), prediction.predicted.to_numpy()).items():
                    np.testing.assert_allclose(row[key], value, rtol=0, atol=1e-9)
        if label != "final":
            old = pd.read_parquet(output_dir / f"{prefix}_cortex_cv_fold{label[-1]}.predictions.parquet")
            new = pd.read_parquet(output_dir / f"{prefix}_smooth_{label}_predictions.parquet")
            assert np.array_equal(old[KEY], new[KEY])
            np.testing.assert_array_equal(
                old[["brdu", "dapi", "x_um", "y_um", "z_um"]], new[["brdu", "dapi", "x_um", "y_um", "z_um"]]
            )
            assert np.isfinite(new[["brdu", "dapi", "global_predicted", "smooth_predicted"]].to_numpy()).all()
            np.testing.assert_array_equal(old.global_predicted, new.global_predicted)
            np.testing.assert_allclose(
                SpatialSubtraction.load(output_dir / f"{prefix}_smooth_{label}_model.joblib").predict(
                    new[["x_um", "y_um"]].to_numpy(), new.dapi.to_numpy()
                ),
                new.smooth_predicted,
                atol=1e-9,
                rtol=0,
            )
            train_blocks = set(zip(np.floor(train.x_um / 250), np.floor(train.y_um / 250)))
            test_blocks = set(zip(np.floor(new.x_um / 250), np.floor(new.y_um / 250)))
            assert train_blocks.isdisjoint(test_blocks)
            outer_predictions.append(new)
    model = SpatialSubtraction.load(output_dir / f"{prefix}_smooth_final_model.joblib")
    for label in labels:
        numerical.append(
            validate_fit(
                SpatialSubtraction.load(output_dir / f"{prefix}_smooth_{label}_model.joblib"),
                all_training[label],
                label=f"{prefix}_smooth_{label}_model",
                penalty=selections[label],
            )
        )
    assert len(numerical) == 50
    xy = samples[["x_um", "y_um"]].to_numpy()
    alpha, beta = model.fields(xy)
    correction = model.predict(xy, samples.dapi.to_numpy())
    assert np.isfinite(alpha).all() and np.isfinite(beta).all() and np.isfinite(correction).all()
    report = {
        "validated_labels": labels,
        "sample_count": len(samples),
        "alpha_range": [float(alpha.min()), float(alpha.max())],
        "beta_range_raw": [float(beta.min()), float(beta.max())],
        "negative_predicted_correction": int((correction < 0).sum()),
        "outside_final_domain": int(outside(model, xy).sum()),
        "final_domain_xy_um": model.domain_xy_um.tolist(),
        "max_raw_primal": max(d["raw_primal"] for d in numerical),
        "max_relative_gap": max(d["relative_gap"] for d in numerical),
        "max_absolute_raw_mean_gap": max(abs(d["raw_mean_gap"]) for d in numerical),
    }
    heldout = pd.read_parquet(output_dir / f"{prefix}_smooth_heldout_predictions.parquet")
    np.testing.assert_array_equal(
        heldout[KEY + ["brdu", "dapi", "global_predicted", "smooth_predicted"]],
        pd.concat(outer_predictions, ignore_index=True)[
            KEY + ["brdu", "dapi", "global_predicted", "smooth_predicted"]
        ],
    )
    pooled = []
    regions = []
    depth = []
    for name, column in [("global", "global_predicted"), ("smooth", "smooth_predicted")]:
        pooled.append({"model": name, **metrics(heldout.brdu.to_numpy(), heldout[column].to_numpy())})
        for keys, regional_table in [(["x_block", "y_block"], regions), (["z_band"], depth)]:
            for key, group in heldout.groupby(keys):
                regional_table.append(
                    {
                        "model": name,
                        **dict(zip(keys, key)),
                        **metrics(group.brdu.to_numpy(), group[column].to_numpy()),
                    }
                )
    report["pooled_heldout"] = pooled
    pd.DataFrame(regions).to_csv(destination / f"{prefix}_smooth_pooled_regions.tsv", sep="\t", index=False)
    pd.DataFrame(depth).to_csv(destination / f"{prefix}_smooth_pooled_depth.tsv", sep="\t", index=False)
    pd.DataFrame(numerical).to_csv(
        destination / f"{prefix}_smooth_numerical_validation.tsv", sep="\t", index=False
    )
    (destination / f"{prefix}_smooth_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--destination-dir", type=Path)
    args = parser.parse_args()
    validate(args.output_dir.resolve(), args.prefix, destination_dir=args.destination_dir)


if __name__ == "__main__":
    main()
