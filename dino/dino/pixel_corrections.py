"""Apply bound raw-pixel corrections in their declared numerical order."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np


class ChannelSubtraction(Protocol):
    @property
    def target(self) -> str: ...

    @property
    def reference(self) -> str: ...

    def __call__(self, target: np.ndarray, reference: np.ndarray, origin_zyx: tuple[int, int, int]) -> np.ndarray: ...


class PixelCorrection(Protocol):
    @property
    def targets(self) -> tuple[str, ...]: ...

    def __call__(self, name: str, pixels: np.ndarray, origin_zyx: tuple[int, int, int]) -> np.ndarray: ...


def correct_channels(
    channels: Mapping[str, np.ndarray],
    origin_zyx: tuple[int, int, int],
    *,
    reference: str | None = None,
    spatial_subtraction: ChannelSubtraction | None = None,
    pixel_correction: PixelCorrection | None = None,
) -> dict[str, np.ndarray]:
    """`reference` names the crop channel holding the subtraction's reference pixels."""
    corrected = dict(channels)
    if spatial_subtraction is not None:
        if reference is None:
            raise ValueError('Spatial subtraction requires the crop channel holding its reference pixels')
        corrected[spatial_subtraction.target] = spatial_subtraction(
            channels[spatial_subtraction.target], channels[reference], origin_zyx
        )
    if pixel_correction is not None:
        targets = pixel_correction.targets
        corrected = {
            name: pixel_correction(name, pixels, origin_zyx) if name in targets else pixels
            for name, pixels in corrected.items()
        }
    return corrected
