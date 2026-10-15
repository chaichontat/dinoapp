from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dino import spatial_subtraction_workflow as workflow


def training_frame() -> pd.DataFrame:
    rows = []
    for block in range(12):
        for depth in range(6):
            rows.append(
                dict(
                    z_index=depth,
                    y_index=block,
                    x_index=len(rows),
                    x_um=block * 250 + 20.0,
                    y_um=(block % 3) * 250 + 30.0,
                    z_um=depth * 100 + 10.0,
                    brdu=1000.0 + block * 10 + depth,
                    dapi=5000.0 + depth,
                )
            )
    return pd.DataFrame(rows)


def test_geometry_and_inner_folds_preserve_whole_spatial_groups(tmp_path):
    config = workflow.WorkflowConfig(tmp_path, "prod2")
    raw = training_frame()
    frame = workflow.geometry(config, raw)
    splits = workflow.inner_splits(frame)
    repeated = workflow.inner_splits(frame)
    heldout = []
    for (train, test), (again_train, again_test) in zip(splits, repeated):
        np.testing.assert_array_equal(train, again_train)
        np.testing.assert_array_equal(test, again_test)
        train_blocks = set(map(tuple, frame.iloc[train][["x_block", "y_block"]].to_numpy()))
        test_blocks = set(map(tuple, frame.iloc[test][["x_block", "y_block"]].to_numpy()))
        assert train_blocks.isdisjoint(test_blocks)
        assert frame.iloc[test].z_band.nunique() == 6
        heldout.extend(test)
    assert sorted(heldout) == list(range(len(frame)))
    np.testing.assert_array_equal(frame[raw.columns], raw)


def test_old_training_tables_join_geometry_without_changing_rows(tmp_path):
    config = workflow.WorkflowConfig(tmp_path, "prod3")
    raw = training_frame()
    raw.to_parquet(tmp_path / "prod3_cortex_samples.parquet", index=False)
    subset = raw.iloc[[8, 0, 17]].drop(columns=["x_um", "y_um", "z_um"])
    joined = workflow.geometry(config, subset)
    np.testing.assert_array_equal(
        joined[["z_index", "y_index", "x_index"]], subset[["z_index", "y_index", "x_index"]]
    )
    np.testing.assert_array_equal(
        joined[["x_um", "y_um", "z_um"]], raw.iloc[[8, 0, 17]][["x_um", "y_um", "z_um"]]
    )


def test_selection_weights_observations_and_preserves_tie_rule():
    rows = []
    for penalty, losses in [(0.001, [1.0, 9.0, 9.0]), (0.01, [6.0, 6.0, 6.0]), (0.1, [7.0, 7.0, 7.0])]:
        for fold, (loss, count) in enumerate(zip(losses, [1, 99, 99])):
            rows.append(dict(penalty=penalty, inner_fold=fold, pinball_loss=loss, samples=count))
    table = pd.DataFrame(rows)
    chosen, scores = workflow.select_penalty(table)
    assert chosen == 0.01
    assert scores["0.001"] > scores["0.01"]
    table["pinball_loss"] = 5.0
    assert workflow.select_penalty(table)[0] == 0.1


def test_tuning_fits_only_inner_training_and_does_not_read_outer_test(tmp_path, monkeypatch):
    config = workflow.WorkflowConfig(tmp_path, "prod2")
    frame = workflow.geometry(config, training_frame())
    calls = []

    class InlinePool:
        def __init__(self, max_workers):
            assert max_workers == 3

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def map(self, function, jobs):
            return map(function, jobs)

    def fit(config, train, penalty, stem):
        calls.append((train.copy(), penalty, stem))
        domain = np.column_stack([train[["x_um", "y_um"]].min(), train[["x_um", "y_um"]].max()])
        return SimpleNamespace(
            domain_xy_um=domain,
            diagnostics={"iterations": 1},
            predict=lambda xy, _reference: np.full(len(xy), train.brdu.mean()),
        )

    def forbidden_read(*_args, **_kwargs):
        raise AssertionError("Inner tuning must not read an outer test roster")

    monkeypatch.setattr(workflow, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(workflow, "fit_checkpoint", fit)
    monkeypatch.setattr(pd, "read_parquet", forbidden_read)
    chosen = workflow.tune(config, frame, "outer0")
    assert chosen in workflow.GRID
    assert len(calls) == 9
    for position, (train, penalty, stem) in enumerate(calls):
        expected_train, _test = workflow.inner_splits(frame)[position % 3]
        np.testing.assert_array_equal(train, frame.iloc[expected_train])
        assert penalty == workflow.GRID[position // 3]
        assert stem.startswith("prod2_smooth_outer0_")
    assert (tmp_path / "prod2_smooth_outer0_selection.json").exists()
    assert (tmp_path / "prod2_smooth_outer0_inner_folds.parquet").exists()
    assert not list(tmp_path.glob("prod3_*"))


def test_invalid_prefix_and_training_cap_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="prefix"):
        workflow.WorkflowConfig(tmp_path, "../prod2")
    with pytest.raises(ValueError, match="50,000"):
        workflow.fit_checkpoint(
            workflow.WorkflowConfig(tmp_path, "prod2"), pd.DataFrame(index=range(50001)), 0.001, "unused"
        )


def test_requested_precision_preserves_defaults_and_rejects_looser_checkpoints(tmp_path):
    default = workflow.WorkflowConfig(tmp_path, "prod3")
    assert (default.solver_absolute_tolerance, default.solver_relative_tolerance) == (5e-4, 1e-6)
    precise = workflow.WorkflowConfig(tmp_path, "prod2", 5e-5, 1e-7)
    loose_model = SimpleNamespace(diagnostics={"tolerance": 5e-4, "relative_tolerance": 1e-6})
    workflow.validate_checkpoint_precision(loose_model, default)
    with pytest.raises(ValueError, match="looser"):
        workflow.validate_checkpoint_precision(loose_model, precise)
    strict_model = SimpleNamespace(diagnostics={"tolerance": 1e-5, "relative_tolerance": 1e-8})
    workflow.validate_checkpoint_precision(strict_model, precise)
    with pytest.raises(ValueError, match="tolerances"):
        workflow.WorkflowConfig(tmp_path, "prod2", np.nan, 1e-7)


def test_larger_budget_reuses_accepted_checkpoint_without_refitting(tmp_path, monkeypatch):
    config = workflow.WorkflowConfig(
        tmp_path,
        "prod2",
        solver_absolute_tolerance=5e-5,
        solver_relative_tolerance=1e-7,
        solver_max_iter=180000,
        solver_time_limit=600,
    )
    frame = training_frame()
    model = SimpleNamespace(
        penalty=0.1,
        domain_xy_um=np.column_stack([frame[["x_um", "y_um"]].min(), frame[["x_um", "y_um"]].max()]),
        diagnostics={
            "accuracy_contract": workflow.ACCURACY_CONTRACT,
            "samples": len(frame),
            "tolerance": 5e-5,
            "relative_tolerance": 1e-7,
            "max_iter": 60000,
            "time_limit": 240,
        },
    )
    path = tmp_path / "accepted.joblib"
    path.write_bytes(b"accepted checkpoint")
    monkeypatch.setattr(workflow.SpatialSubtraction, "load", lambda _path: model)

    def forbidden_fit(*_args, **_kwargs):
        raise AssertionError("A larger budget must not refit an accepted checkpoint")

    monkeypatch.setattr(workflow, "fit_spatial_subtraction", forbidden_fit)
    assert workflow.fit_checkpoint(config, frame, 0.1, "accepted") is model
    assert path.read_bytes() == b"accepted checkpoint"
    default = workflow.WorkflowConfig(tmp_path, "prod3")
    assert (default.solver_max_iter, default.solver_time_limit) == (60000, 240)
    with pytest.raises(ValueError, match="iteration budget"):
        workflow.WorkflowConfig(tmp_path, "prod2", solver_max_iter=0)
    with pytest.raises(ValueError, match="time limit"):
        workflow.WorkflowConfig(tmp_path, "prod2", solver_time_limit=np.nan)


def test_cli_passes_explicit_numerical_budget_to_run(tmp_path, monkeypatch):
    import sys

    captured = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "workflow",
            "run",
            "--output-dir",
            str(tmp_path),
            "--prefix",
            "prod2",
            "--solver-absolute-tolerance",
            "5e-5",
            "--solver-relative-tolerance",
            "1e-7",
            "--solver-max-iterations",
            "180000",
            "--solver-time-limit",
            "600",
        ],
    )
    monkeypatch.setattr(workflow, "run", captured.append)
    workflow.main()
    config = captured[0]
    assert (
        config.solver_absolute_tolerance,
        config.solver_relative_tolerance,
        config.solver_max_iter,
        config.solver_time_limit,
    ) == (5e-5, 1e-7, 180000, 600)
