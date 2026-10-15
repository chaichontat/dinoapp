import importlib.metadata
import json
import sys
from types import ModuleType

import pytest
import torch

from dino import cell_dino, zarr_model


def test_transform_uses_installed_package_without_changing_import_path(monkeypatch):
    transforms = ModuleType("dinov2.data.cell_dino.transforms")
    calls = []
    transform = torch.nn.Identity()

    def make_transform(*, resize_size, crop_size):
        calls.append((resize_size, crop_size))
        return transform

    monkeypatch.setattr(transforms, "make_classification_eval_cell_transform", make_transform, raising=False)
    monkeypatch.setitem(sys.modules, transforms.__name__, transforms)
    before = list(sys.path)

    assert cell_dino.build_cell_transform(resize_size=8, crop_size=4) is transform
    assert calls == [(8, 4)]
    assert sys.path == before


class _Distribution:
    def __init__(self, direct_url: dict | None) -> None:
        self.direct_url = direct_url

    def read_text(self, name: str) -> str | None:
        assert name == "direct_url.json"
        return None if self.direct_url is None else json.dumps(self.direct_url)


def _install(monkeypatch, direct_url: dict | None) -> None:
    def distribution(name: str) -> _Distribution:
        assert name == "dinov2"
        return _Distribution(direct_url)

    monkeypatch.setattr(importlib.metadata, "distribution", distribution)


def test_installed_dinov2_source_names_the_git_commit(monkeypatch):
    _install(monkeypatch, {
        "url": "https://github.com/chaichontat/dinov2",
        "vcs_info": {"commit_id": "76d1757", "requested_revision": "76d1757", "vcs": "git"},
    })

    assert cell_dino.installed_dinov2_source() == "git+https://github.com/chaichontat/dinov2@76d1757"


@pytest.mark.parametrize("direct_url", [
    {"url": "file:///home/user/dinov2", "dir_info": {"editable": True}},
    {"url": "file:///home/user/dinov2", "dir_info": {}},
    None,
], ids=["editable", "directory", "index"])
def test_installed_dinov2_source_rejects_non_vcs_installs(monkeypatch, direct_url):
    _install(monkeypatch, direct_url)

    fix = 'pip install --no-deps "dinov2 @ git+https://github.com/chaichontat/dinov2@<commit>"'
    with pytest.raises(RuntimeError) as error:
        cell_dino.installed_dinov2_source()
    assert fix in str(error.value)


def test_installed_dinov2_source_rejects_a_missing_package(monkeypatch):
    def distribution(name: str):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", distribution)

    with pytest.raises(RuntimeError, match="dinov2 is not installed.*pip install --no-deps"):
        cell_dino.installed_dinov2_source()


def test_cell_dino_backbone_resolves_installed_entrypoint():
    import dinov2.hub.cell_dino.backbones as backbones

    assert zarr_model.cell_dino_backbone("cell_dino_hpa_vitl16") is backbones.cell_dino_hpa_vitl16


@pytest.mark.parametrize("name", ["cell_dino_unknown", "torch", "Weights", "_make_cell_dino_model"])
def test_cell_dino_backbone_rejects_names_that_are_not_backbones(name):
    with pytest.raises(ValueError, match=r"dinov2\.hub\.cell_dino\.backbones"):
        zarr_model.cell_dino_backbone(name)


def test_loader_calls_installed_backbone_with_checkpoint(monkeypatch, tmp_path):
    weights = tmp_path / "weights.pth"
    model = torch.nn.Linear(2, 2)
    calls = []
    attention = ModuleType("dinov2.layers.attention")
    monkeypatch.setattr(attention, "XFORMERS_AVAILABLE", False, raising=False)
    monkeypatch.setitem(sys.modules, attention.__name__, attention)
    backbones = ModuleType("dinov2.hub.cell_dino.backbones")

    def cell_dino_hpa_vitl16(**kwargs):
        calls.append(kwargs)
        return model

    monkeypatch.setattr(backbones, "cell_dino_hpa_vitl16", cell_dino_hpa_vitl16, raising=False)
    monkeypatch.setitem(sys.modules, backbones.__name__, backbones)
    before = list(sys.path)

    result = zarr_model.load_cell_dino_model(
        hub_model="cell_dino_hpa_vitl16", weights=weights, device=torch.device("cpu"), precision="float32"
    )

    assert result is model
    assert not result.training
    assert calls == [{"pretrained_path": str(weights)}]
    assert sys.path == before
