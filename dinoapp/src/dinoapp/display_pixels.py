"""Read corrected display pixels independently of saved prediction overlays."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
from dino_markers import bundle_contract, contract_of, require_same_contract

from dino.artifacts import file_sha256
from dino.periodic_pixels import PeriodicPixels, require_periodic_targets
from dino.spatial_pixels import SpatialPixels, require_spatial_subtraction_contract
from dino.zarr_source import CellCrop, open_review_source
from dino.pixel_corrections import correct_channels

class DisplayPixels:
    """Use the exact bound spatial subtraction and periodic transform, in order."""

    def __init__(self, model_path: Path, source_dir: Path):
        self.model_path = model_path.resolve()
        self.source = open_review_source(source_dir)
        bundle = joblib.load(self.model_path)
        if bundle['source_sha256'] != self.source.manifest['source_sha256']:
            raise ValueError('Display model and source identities differ')
        self.contract = bundle_contract(bundle, where=str(self.model_path))
        require_same_contract(
            contract_of(source_dir), self.contract,
            expected_from=f'display source {source_dir}', actual_from=str(self.model_path),
        )
        require_spatial_subtraction_contract(self.contract, artifact=str(self.model_path))
        self.spatial = SpatialPixels.from_spec(bundle['spatial_subtraction'])
        manifest = self.source.manifest
        geometry = manifest['channels']['brdu']['ome']
        if geometry != manifest['channels']['pi']['ome']:
            raise ValueError('Display BrdU and DAPI physical grids differ')
        transforms = geometry['coordinate_transformations']
        if (transforms != [{'type': 'scale', 'scale': list(self.spatial.scale_zyx_um)},
                           {'type': 'translation', 'translation': list(self.spatial.origin_zyx_um)}]):
            raise ValueError('Display subtraction and source physical grids differ')
        spec = bundle['pixel_correction']
        self.periodic = PeriodicPixels(spec['model'], spec['model_sha256'], tuple(spec['scale_zyx_um']))
        if self.periodic.identity != spec:
            raise ValueError('Unsupported display periodic correction contract')
        if file_sha256(Path(self.periodic.path)) != self.periodic.sha256:
            raise ValueError('Display periodic correction artifact identity differs')
        require_periodic_targets(self.periodic, self.contract, artifact=str(self.model_path))
        self.identity = file_sha256(self.model_path)
        self.description = 'BrdU/DAPI subtracted; periodic intensity corrected'

    def validate_native_grid(self, *, labels, shape, scale, origin):
        if (Path(self.source.manifest['labels']['root']).resolve() != Path(labels).resolve()
                or tuple(self.source.labels.shape) != tuple(shape)
                or self.source.offset_zyx != (0, 0, 0)
                or not np.allclose(self.spatial.scale_zyx_um, scale, rtol=1e-12, atol=1e-12)
                or not np.allclose(self.spatial.origin_zyx_um, origin, rtol=1e-12, atol=1e-12)):
            raise ValueError('Corrected display source differs from native labels or physical grid')

    def channels(self, channels, origin_zyx, *, reference):
        return correct_channels(
            channels, origin_zyx, reference=reference,
            spatial_subtraction=self.spatial, pixel_correction=self.periodic,
        )

    def crop(self, crop: CellCrop, origin_zyx, *, reference: str) -> CellCrop:
        return replace(crop, channels=self.channels(crop.channels, origin_zyx, reference=reference))

    def field_images(self, images, selection):
        z, ys, xs = selection
        raw = {key: np.asarray(self.source.channels[key][selection]) for key in self.contract.channel_keys}
        corrected = self.channels(raw, (z, ys.start, xs.start), reference=self.spatial.reference)
        return {**images, **{marker: corrected[marker] for marker in self.contract.markers}}
