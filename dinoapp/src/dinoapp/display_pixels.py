"""Read corrected display pixels independently of saved prediction overlays."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np

from dino.artifacts import file_sha256
from dino.periodic_pixels import PeriodicPixels
from dino.spatial_pixels import SpatialPixels
from dino.zarr_source import CellCrop, open_review_source
from dino.pixel_corrections import correct_channels

OUTPUT_ROOT = Path('/home/chaichontat/fishtools2/thesis-internal/thesis/output')


class DisplayPixels:
    """Use the exact bound spatial subtraction and periodic transform, in order."""

    def __init__(self, model_path: Path, source_dir: Path):
        self.model_path = model_path.resolve()
        self.source = open_review_source(source_dir)
        bundle = joblib.load(self.model_path)
        if bundle['source_sha256'] != self.source.manifest['source_sha256']:
            raise ValueError('Display model and source identities differ')
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
        self.identity = file_sha256(self.model_path)
        self.description = 'BrdU/DAPI subtracted; periodic intensity corrected'

    @classmethod
    def for_dataset(cls, dataset: str) -> DisplayPixels:
        root = OUTPUT_ROOT / f'{dataset}_brdu_dapi_subtraction' / 'reembedding'
        return cls(root / 'model/model.joblib', root / 'source')

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
        raw = {marker: np.asarray(self.source.channels[marker][selection]) for marker in ('brdu', 'edu', 'pi')}
        corrected = self.channels(raw, (z, ys.start, xs.start), reference='pi')
        return {**images, 'BrdU': corrected['brdu'], 'EdU': corrected['edu']}
