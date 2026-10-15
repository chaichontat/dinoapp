from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from dino import barrage, workspace_inputs


def _write_h5ad(path: Path, obs: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    X = np.empty((len(obs), 0), dtype=np.float32)
    adata = ad.AnnData(X=X, obs=obs)
    adata.obs_names = pd.Index(obs.index.astype(str))
    adata.write_h5ad(path)


def test_barrage_can_load_per_roi_h5ads_dir(tmp_path: Path) -> None:
    h5ads_dir = tmp_path / "h5ads"
    required = {
        "roi": str,
        "x": float,
        "y": float,
        "z": float,
        "brdu_mean": float,
        "brdu_min": float,
        "brdu_max": float,
        "brdu_median": float,
        "brdu_std": float,
        "edu_mean": float,
        "edu_min": float,
        "edu_max": float,
        "edu_median": float,
        "edu_std": float,
    }

    obs_a = pd.DataFrame({k: [t("1")] if t is str else [t(1)] for k, t in required.items()}, index=["roiA|1"])
    obs_a["roi"] = "roiA"
    obs_b = pd.DataFrame({k: [t("2")] if t is str else [t(2)] for k, t in required.items()}, index=["roiB|2"])
    obs_b["roi"] = "roiB"

    _write_h5ad(h5ads_dir / "roiA.h5ad", obs_a)
    _write_h5ad(h5ads_dir / "roiB.h5ad", obs_b)

    adata = barrage.load_barrage_adata(h5ad_path=h5ads_dir, dataset="DS1")
    assert set(adata.obs_names.astype(str).tolist()) == {"roiA|1", "roiB|2"}
    assert set(adata.obs["dataset"].astype(str).unique().tolist()) == {"DS1"}


def test_select_barrage_adata_supports_all_datasets_and_single_dataset() -> None:
    obs = pd.DataFrame(
        {
            "dataset": ["DS1", "DS2", "DS1"],
            "roi": ["1", "2", "3"],
        },
        index=["a", "b", "c"],
    )
    adata = ad.AnnData(X=np.empty((3, 0), dtype=np.float32), obs=obs)
    adata.obs_names = pd.Index(obs.index.astype(str))

    got_all = barrage._select_barrage_adata(adata=adata, dataset=barrage.ALL_DATASETS)
    assert set(got_all.obs_names.astype(str).tolist()) == {"a", "b", "c"}

    got_ds1 = barrage._select_barrage_adata(adata=adata, dataset="DS1")
    assert set(got_ds1.obs_names.astype(str).tolist()) == {"a", "c"}


def test_workspace_for_dataset_checks_nvme_root(monkeypatch, tmp_path: Path) -> None:
    working_root = tmp_path / "working"
    nvme_root = tmp_path / "nvme"
    dataset = "DS1"
    dataset_root = nvme_root / dataset
    dataset_root.mkdir(parents=True)
    (dataset_root / "ready.DONE").write_text("", encoding="utf-8")
    monkeypatch.setattr(workspace_inputs, "WORKSPACE_ROOT_BASES", (working_root, nvme_root))

    cache = {}
    ws = barrage._workspace_for_dataset(dataset=dataset, cache=cache)
    assert ws.path == dataset_root.resolve()
    monkeypatch.setattr(
        barrage, "resolve_workspace_root",
        lambda _: pytest.fail("cached workspace must bypass discovery"),
    )
    assert barrage._workspace_for_dataset(dataset=dataset, cache=cache) is ws


def test_select_additional_barrage_cells_preserves_existing_and_adds_new_ones() -> None:
    obs = pd.DataFrame(
        {
            "brdu_mean": [1.0, 2.0, 3.0, 4.0],
            "brdu_std": [1.0, 1.0, 1.0, 1.0],
            "edu_mean": [1.0, 2.0, 3.0, 4.0],
            "edu_std": [1.0, 1.0, 1.0, 1.0],
        },
        index=["existing", "labeled_new", "fresh_a", "fresh_b"],
    )

    preserved, new_cells = barrage._select_additional_barrage_cells(
        obs=obs,
        feature_cols=["brdu_mean", "brdu_std", "edu_mean", "edu_std"],
        existing_cells=["existing"],
        preselected=["labeled_new"],
        n_cells=2,
        n_bins=2,
        seed=0,
    )

    assert preserved == ["existing"]
    assert new_cells[0] == "labeled_new"
    assert len(new_cells) == 2
    assert "existing" not in new_cells


def _rebuild_fixture(monkeypatch, tmp_path, *, dataset="DS1", finite=True, n_cells=1000):
    from types import SimpleNamespace

    out = tmp_path / "barrage"
    (out / "images").mkdir(parents=True)
    (out / "images" / "old.png").write_bytes(b"previous image")
    pd.DataFrame({"cell": ["r|1"], "dataset": ["DS1"], "roi": ["r"]}).to_csv(out / "index.csv", index=False)
    (out / "labels.csv").write_text("cell,brdu,edu\nr|1,1,0\n")
    source = tmp_path / "source.h5ad"
    source.touch()
    cfg = barrage.BarrageConfig(
        dataset, source, out, True, n_cells, 0, 2, "edu", "pi", "seg", 10, 3, "magma", "cyan", True
    )
    obs = {
        key: [1.0]
        for key in ["x", "y", "z"]
        + [
            f"{marker}_{stat}"
            for marker in ["brdu", "edu"]
            for stat in ["mean", "min", "max", "median", "std"]
        ]
    }
    obs.update(dataset=[dataset], roi=["r"])
    if not finite:
        obs["brdu_mean"] = [np.nan]
    adata = ad.AnnData(X=np.empty((1, 0)), obs=pd.DataFrame(obs, index=["r|1"]))
    monkeypatch.setattr(barrage, "_parse_args", lambda: cfg)
    monkeypatch.setattr(barrage, "load_barrage_adata", lambda **kwargs: adata)
    monkeypatch.setattr(barrage, "_workspace_for_dataset", lambda **kwargs: None)
    fig = SimpleNamespace(
        tight_layout=lambda: None, savefig=lambda path, **kwargs: path.write_bytes(b"new image")
    )
    monkeypatch.setattr(
        barrage, "plot_cell_thumbnail_with_outline", lambda **kwargs: (fig, None, np.zeros((1, 1)))
    )
    monkeypatch.setattr(barrage.plt, "close", lambda fig: None)
    return out


def test_overwrite_rejects_changed_label_namespace_without_modification(monkeypatch, tmp_path):
    import pytest

    out = _rebuild_fixture(monkeypatch, tmp_path, dataset="DS2", n_cells=1)
    before = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="identity|namespace"):
        barrage.main()
    assert {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()} == before


def test_overwrite_fully_labeled_population_rebuilds(monkeypatch, tmp_path):
    out = _rebuild_fixture(monkeypatch, tmp_path)
    labels = (out / "labels.csv").read_bytes()
    barrage.main()
    assert pd.read_csv(out / "index.csv")["cell"].tolist() == ["r|1"]
    assert (out / "labels.csv").read_bytes() == labels
    assert len(list((out / "images").glob("*.png"))) == 1


def test_overwrite_rejects_unprovable_namespace(monkeypatch, tmp_path):
    import pytest

    out = _rebuild_fixture(monkeypatch, tmp_path, n_cells=1)
    (out / "index.csv").write_text("cell,roi\nr|1,r\n")
    before = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="identity|namespace"):
        barrage.main()
    assert {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()} == before


def test_overwrite_invalid_features_preserves_artifacts(monkeypatch, tmp_path):
    import pytest

    out = _rebuild_fixture(monkeypatch, tmp_path, finite=False)
    (out / "labels.csv").write_text("cell,brdu,edu\n")
    before = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="No finite rows"):
        barrage.main()
    assert {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()} == before


def test_overwrite_without_retained_labels_rebuilds_old_index(monkeypatch, tmp_path):
    out = _rebuild_fixture(monkeypatch, tmp_path)
    (out / "labels.csv").write_text("cell,brdu,edu\n")
    (out / "index.csv").write_text("obsolete index")
    barrage.main()
    assert pd.read_csv(out / "index.csv")["cell"].tolist() == ["r|1"]
    assert not (out / "images" / "old.png").exists()


def test_thumbnail_keeps_target_and_neighbor_outline_masks(monkeypatch) -> None:
    from types import SimpleNamespace

    import zarr

    cell = 'DS:r|2'
    adata = ad.AnnData(
        X=np.empty((1, 0)),
        obs=pd.DataFrame(dict(roi=['r'], x=[1], y=[1], z=[0]), index=[cell]),
    )
    labels = np.array([[0, 2, 0], [2, 2, 3], [0, 3, 0]], dtype=np.uint16)
    fused = zarr.array(np.zeros((1, 3, 3, 1), dtype=np.uint16))
    fused.attrs['key'] = ['brdu']
    monkeypatch.setattr(barrage, 'cell_thumbnail_from_fused', lambda *args, **kwargs: np.zeros((3, 3, 1)))
    outlines = []
    monkeypatch.setattr(barrage.plt.Axes, 'contour', lambda self, mask, **kwargs: outlines.append(mask))
    fig, _, _ = barrage.plot_cell_thumbnail_with_outline(
        adata=adata, dataset='DS', ws=SimpleNamespace(), cell=cell,
        codebook='edu', seg_codebook='pi', segmentation_name='seg', size=3,
        fused_cache={('DS', 'r', 'edu'): fused},
        seg_cache={('DS', 'r', 'pi', 'seg'): zarr.array(labels[None])},
        render_all_outlines=True,
    )
    barrage.plt.close(fig)
    assert len(outlines) == 2
    np.testing.assert_array_equal(outlines[0], labels == 3)
    np.testing.assert_array_equal(outlines[1], labels == 2)
