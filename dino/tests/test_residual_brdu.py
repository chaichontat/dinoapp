import numpy as np
from scipy.ndimage import map_coordinates
import SimpleITK as sitk

from dino.residual_brdu import chromatic_pull, raw_mask_coordinates


def test_chromatic_pull_matches_owning_composite_transform():
    coefficients = np.array([.99, .001, -.002, 1.01, .3, -.7])
    displacement = np.array([12., -8.])
    matrix, offset = chromatic_pull(coefficients, displacement, (2048, 2048))
    translation = sitk.TranslationTransform(3)
    translation.SetParameters((-12., 8., 0.))
    affine = sitk.AffineTransform(3)
    full = np.eye(3)
    full[:2, :2] = matrix
    affine.SetMatrix(full.ravel())
    affine.SetTranslation((.3, -.7, 0.))
    affine.SetCenter((1023.5 - 12, 1023.5 + 8, 0.))
    composite = sitk.CompositeTransform(3)
    composite.AddTransform(translation)
    composite.AddTransform(affine)
    for point in [(0., 0.), (100., 200.), (2047., 2047.)]:
        expected = composite.TransformPoint((*point, 0.))[:2]
        np.testing.assert_allclose(matrix @ point + offset, expected, atol=1e-10)


def test_inverse_mask_mapping_preserves_integer_labels_and_pixel_centers():
    # A raw2x2 group shares one native mask label; labels never get interpolated.
    coordinates, _ = raw_mask_coordinates(np.eye(2), np.zeros(2), np.array([0., 0.]), (46, 46))
    native = np.array([[0, 1000, 0], [0, 2000, 0], [0, 0, 0]], dtype=np.uint32)
    mapped = map_coordinates(native, coordinates, order=0, mode='grid-constant', cval=0)
    assert mapped.dtype == np.uint32
    assert set(np.unique(mapped)) <= {0, 1000, 2000}
    assert np.array_equal(mapped[40:42, 42:44], np.full((2, 2), 1000))
    assert np.array_equal(mapped[42:44, 42:44], np.full((2, 2), 2000))
