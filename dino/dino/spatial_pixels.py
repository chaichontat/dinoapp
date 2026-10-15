"""Lazy raw BrdU/DAPI subtraction before existing DINO pixel preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dino_markers import MarkerContract, bundle_contract, derive_bundle

from dino.artifacts import file_sha256
from dino.spatial_subtraction import SpatialSubtraction


@lru_cache(maxsize=4)
def _load_model(path: str, sha256: str) -> SpatialSubtraction:
    if file_sha256(Path(path)) != sha256:
        raise ValueError(f"Spatial subtraction model hash mismatch: {path}")
    return SpatialSubtraction.load(Path(path))


@dataclass(frozen=True)
class SpatialPixels:
    """Use full fused-grid coordinates and the uncorrected reference crop."""

    path: str
    sha256: str
    scale_zyx_um: tuple[float, float, float]
    origin_zyx_um: tuple[float, float, float]

    def __post_init__(self) -> None:
        scale, origin = np.asarray(self.scale_zyx_um), np.asarray(self.origin_zyx_um)
        if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Spatial pixels require finite positive ZYX voxel spacing")
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("Spatial pixels require a finite physical ZYX origin")

    @property
    def target(self) -> str:
        return "brdu"

    @property
    def reference(self) -> str:
        return "pi"

    @property
    def identity(self) -> dict[str, Any]:
        return dict(model=self.path, model_sha256=self.sha256,
                    scale_zyx_um=list(self.scale_zyx_um), origin_zyx_um=list(self.origin_zyx_um),
                    target=self.target, reference=self.reference, reference_identity="true 405 nm DAPI",
                    operation="uint16(clip(rint(raw BrdU - alpha(physical XY) * raw pi - beta(physical XY)), 0, 65535)); zero paired-unsupported pixels",
                    order="before periodic gains and DINO preprocessing")

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> SpatialPixels:
        transform = cls(spec["model"], spec["model_sha256"], tuple(spec["scale_zyx_um"]), tuple(spec["origin_zyx_um"]))
        if dict(spec) != transform.identity:
            raise ValueError("Unsupported spatial pixel subtraction contract")
        if file_sha256(Path(transform.path)) != transform.sha256:
            raise ValueError(f"Spatial subtraction model hash mismatch: {transform.path}")
        _load_model(transform.path, transform.sha256)
        return transform

    def __call__(self, brdu: np.ndarray, dapi: np.ndarray, origin_zyx: tuple[int, int, int]) -> np.ndarray:
        if brdu.ndim != 2 or dapi.shape != brdu.shape:
            raise ValueError("Spatial pixels require paired matching 2D BrdU/DAPI crops")
        y = (np.arange(brdu.shape[0]) + origin_zyx[1]) * self.scale_zyx_um[1] + self.origin_zyx_um[1]
        x = (np.arange(brdu.shape[1]) + origin_zyx[2]) * self.scale_zyx_um[2] + self.origin_zyx_um[2]
        yy, xx = np.meshgrid(y, x, indexing="ij")
        xy = np.column_stack([xx.ravel(), yy.ravel()])
        model = _load_model(self.path, self.sha256)
        signed = model.signed_residual(xy, dapi.ravel(), brdu.ravel()).reshape(brdu.shape)
        return np.clip(np.rint(signed), 0, 65535).astype(np.uint16)


def require_spatial_subtraction_contract(contract: MarkerContract, *, artifact: str) -> None:
    if contract.reference != "pi" or "brdu" not in contract.markers:
        raise ValueError(
            f"{artifact}: spatial subtraction supports only BrdU against the pi channel verified as 405 nm DAPI; "
            f"contract is {contract}"
        )


def bind_spatial_subtraction(
    bundle: Mapping[str, Any], *, model_path: Path, source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a bound model bundle, preserving classifiers and periodic gains.

    The source manifest owns the common fused physical grid. Its ``pi`` channel
    must have been verified as the true 405 nm DAPI source by the run producer.
    This API binds metadata only and writes neither source pixels nor artifacts.
    """
    require_spatial_subtraction_contract(bundle_contract(bundle, where="model bundle"), artifact="model bundle")
    if "spatial_subtraction" in bundle:
        raise ValueError("Model bundle already has spatial subtraction")
    if bundle["source_sha256"] != source_manifest["source_sha256"]:
        raise ValueError("Spatial subtraction requires the classifier's source")
    channels = source_manifest["channels"]
    if "pi" not in channels:
        raise ValueError("Spatial subtraction requires raw pi (405 nm DAPI)")
    geometry = channels["brdu"]["ome"]
    if geometry != channels["pi"]["ome"]:
        raise ValueError("BrdU and DAPI physical grids must match")
    if geometry["axes"] != [dict(name=n, type="space", unit="micrometer") for n in "zyx"]:
        raise ValueError("Spatial subtraction requires ZYX axes in micrometers")
    transforms = geometry["coordinate_transformations"]
    if [item["type"] for item in transforms] != ["scale", "translation"]:
        raise ValueError("Spatial subtraction requires recorded scale then translation")
    scale, origin = tuple(transforms[0]["scale"]), tuple(transforms[1]["translation"])
    if "pixel_correction" in bundle and not np.allclose(bundle["pixel_correction"]["scale_zyx_um"], scale, rtol=1e-12, atol=1e-12):
        raise ValueError("Periodic and spatial subtraction voxel scales differ")
    path = model_path.resolve()
    transform = SpatialPixels(str(path), file_sha256(path), scale, origin)
    _load_model(transform.path, transform.sha256)
    return derive_bundle(bundle, {"spatial_subtraction": transform.identity}, where="model bundle")
