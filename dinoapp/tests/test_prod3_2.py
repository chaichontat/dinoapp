import base64
import io

import numpy as np
from PIL import Image

from dinoapp.app import get_native_modules
from dinoapp.native import Field, render_field


def test_prod3_2_is_a_native_dataset():
    native, shared_app = get_native_modules('Prod3-2')
    assert native.__name__ == 'dinoapp.prod3_2'
    assert shared_app.__name__ == 'dinoapp.native_app'


def test_single_ventricular_surface_and_true_dapi_render():
    mask = np.ones((40, 40), dtype=np.uint32)
    field = Field(
        {'DAPI': np.full(mask.shape, 64, dtype=np.uint16),
         'BrdU': np.zeros(mask.shape), 'EdU': np.zeros(mask.shape)},
        mask, np.array([1], dtype=np.uint32), np.zeros(mask.shape, dtype=int),
        np.zeros(mask.shape, dtype=bool), np.zeros(2), np.ones(2),
        np.full(2, 20.), 12., 20, 0, 0,
        [np.array([[[10., 5.], [10., 35.]]])], {},
    )
    field.anatomy_channel = 'DAPI'
    result = render_field(field, np.array([0]), mode='DAPI', selected=None,
                          border_alpha=0., boundary_alpha=1.,
                          contrast=np.tile([0., 255.], (3, 1)))
    image = Image.open(io.BytesIO(base64.b64decode(result.split(',', 1)[1])))
    assert image.size == (40, 40)
    assert image.getpixel((1, 1))[:3] == (64, 64, 64)
