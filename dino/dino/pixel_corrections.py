"""Apply bound raw-pixel corrections in their declared numerical order."""
from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np


def correct_channels(
    channels: Mapping[str, np.ndarray],
    origin_zyx: tuple[int, int, int],
    *,
    reference: str,
    spatial_subtraction: Callable[[np.ndarray, np.ndarray, tuple[int, int, int]], np.ndarray] | None = None,
    pixel_correction: Callable[[str, np.ndarray, tuple[int, int, int]], np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    corrected = dict(channels)
    if spatial_subtraction is not None:
        corrected['brdu'] = spatial_subtraction(channels['brdu'], channels[reference], origin_zyx)
    if pixel_correction is not None:
        corrected = {
            name: pixel_correction(name, pixels, origin_zyx) if name in ('brdu', 'edu') else pixels
            for name, pixels in corrected.items()
        }
    return corrected
