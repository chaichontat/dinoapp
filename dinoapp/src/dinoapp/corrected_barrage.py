"""Render corrected native pixels for an existing indexed barrage."""
from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np
import pyarrow.parquet as pq

from dino.crop_rendering import read_crop, render_crop
from dino_markers import default_config
from dino.zarr_source import CellCropReader
from dinoapp.display_pixels import DisplayPixels
from dinoapp.native import render_brdu_crop, render_edu_crop
from dinoapp.sources import BarrageSource, Crop


class CorrectedBarrageSource(BarrageSource):
    """Keep the existing index and labels, replacing only the displayed crops."""

    def __init__(self, root: Path):
        super().__init__(root)
        spec = json.loads((self.root / 'display.json').read_text())
        if spec['schema'] != 'dino-corrected-display-v1':
            raise ValueError('Unsupported corrected barrage display contract')
        self.display_pixels = DisplayPixels(Path(spec['model']), Path(spec['source']))
        self.contract = json.loads((self.root / 'run.json').read_text())
        self.reader = CellCropReader(labels=self.display_pixels.source.labels,
                                     channels=self.display_pixels.source.channels,
                                     offset_zyx=self.display_pixels.source.offset_zyx,
                                     cache_bytes=4 * int(np.prod(self.display_pixels.source.labels.chunks))
                                     * self.display_pixels.source.labels.dtype.itemsize + 64 * 1024**2)
        self.props = pq.read_table(Path(spec['source']) / 'cells.parquet',
                                  columns=['label', 'plane_z', 'centroid_y', 'centroid_x'],
                                  filters=[('label', 'in', self.rows.label.tolist())]).to_pandas().set_index('label')
        if self.props.index.duplicated().any() or set(self.props.index) != set(self.rows.label):
            raise ValueError('Corrected display crop identities differ from barrage index')
        for row in self.rows.itertuples(index=False):
            geometry = self.props.loc[row.label]
            if (int(row.plane_z) != int(geometry.plane_z)
                    or not np.allclose([row.y, row.x], [geometry.centroid_y, geometry.centroid_x], rtol=0, atol=1e-8)):
                raise ValueError('Corrected display crop geometry differs from barrage index')
        self.lock = Lock()
        self.source_id += f':display:{self.display_pixels.identity}'
        self.display_name += f' · {self.display_pixels.description}'

    @lru_cache(maxsize=64)
    def crop(self, cell: str, marker: str) -> Crop:
        if marker not in {'brdu', 'edu', 'combined'}:
            raise ValueError(f'Unexpected crop marker: {marker}')
        if cell not in self._by_cell.index:
            raise KeyError(f'Cell is absent from barrage index: {cell}')
        index = self._by_cell.loc[cell]
        row = self.props.loc[index.label].copy()
        row['label'] = index.label
        size = self.contract['size']
        origin = (int(row.plane_z) + self.reader.offset_zyx[0],
                  int(np.rint(row.centroid_y)) + self.reader.offset_zyx[1] - size // 2,
                  int(np.rint(row.centroid_x)) + self.reader.offset_zyx[2] - size // 2)
        with self.lock:
            crop = read_crop(self.reader, row, size=size)
            crop = self.display_pixels.crop(crop, origin, reference='pi')
        if not crop.mask.any():
            raise ValueError(f'Cell {cell} has no mask on its saved crop plane')
        if marker == 'combined':
            output = io.BytesIO()
            render_crop(crop, cell=cell, display_max=self.contract['display_max'],
                        panels=default_config().panels).save(output, format='PNG')
            image = output.getvalue()
        else:
            render = {'brdu': render_brdu_crop, 'edu': render_edu_crop}[marker]
            image = render(crop, self.contract['display_max'][marker])
        return Crop(image, f'Cell {index.label}', self.display_pixels.description)
