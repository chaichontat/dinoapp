from pathlib import Path
import sys
from types import ModuleType

import dinov2
import pytest
import torch

from dino import cell_dino, infer, train, zarr_model


def test_transform_uses_installed_checkout_without_changing_import_path(monkeypatch, tmp_path):
    repo = tmp_path / "checkout"
    monkeypatch.setattr(dinov2, "__file__", str(repo / "dinov2" / "__init__.py"))
    transforms = ModuleType("dinov2.data.cell_dino.transforms")
    calls = []
    transform = torch.nn.Identity()

    def make_transform(*, resize_size, crop_size):
        calls.append((resize_size, crop_size))
        return transform

    monkeypatch.setattr(transforms, "make_classification_eval_cell_transform", make_transform, raising=False)
    monkeypatch.setitem(sys.modules, transforms.__name__, transforms)
    before = list(sys.path)

    assert cell_dino.build_cell_transform(repo=repo, resize_size=8, crop_size=4) is transform
    assert calls == [(8, 4)]
    assert sys.path == before


def test_checkout_identity_accepts_symlink(tmp_path):
    installed_repo = Path(dinov2.__file__).resolve().parents[1]
    alias = tmp_path / "alias"
    alias.symlink_to(installed_repo, target_is_directory=True)

    cell_dino.require_dinov2_checkout(alias)


def test_transform_rejects_another_checkout_before_importing_transforms(tmp_path):
    before = list(sys.path)
    with pytest.raises(RuntimeError, match="Installed dinov2.*configured checkout"):
        cell_dino.build_cell_transform(repo=tmp_path / "another", resize_size=8, crop_size=4)
    assert sys.path == before


@pytest.mark.parametrize("loader", [infer._load_cell_dino_model, train._load_cell_dino_model, zarr_model.load_cell_dino_model])
def test_local_hub_rejects_another_installed_checkout(loader, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: calls.append((args, kwargs)))
    kwargs = {"precision": "float32"} if loader is zarr_model.load_cell_dino_model else {}

    with pytest.raises(RuntimeError, match="Installed dinov2.*configured checkout"):
        loader(repo=tmp_path / "another", hub_model="cell_dino_hpa_vitl16", weights=tmp_path / "weights.pth", device=torch.device("cpu"), **kwargs)
    assert calls == []


@pytest.mark.parametrize("loader", [infer._load_cell_dino_model, train._load_cell_dino_model, zarr_model.load_cell_dino_model])
def test_local_hub_keeps_checkout_entrypoint_and_checkpoint(loader, monkeypatch, tmp_path):
    repo = Path(dinov2.__file__).resolve().parents[1]
    weights = tmp_path / "weights.pth"
    model = torch.nn.Linear(2, 2)
    calls = []
    attention = ModuleType("dinov2.layers.attention")
    monkeypatch.setattr(attention, "XFORMERS_AVAILABLE", False, raising=False)
    monkeypatch.setitem(sys.modules, attention.__name__, attention)

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return model

    monkeypatch.setattr(torch.hub, "load", load)
    kwargs = {"precision": "float32"} if loader is zarr_model.load_cell_dino_model else {}
    before = list(sys.path)

    result = loader(repo=repo, hub_model="cell_dino_hpa_vitl16", weights=weights, device=torch.device("cpu"), **kwargs)

    assert result is model
    assert not result.training
    assert calls == [((str(repo), "cell_dino_hpa_vitl16"), {"source": "local", "pretrained_path": str(weights)})]
    assert sys.path == before
