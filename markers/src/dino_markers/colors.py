"""Color parsing, identity colors and badge colors."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from .calls import validate_label
from .contract import RGB, Label, MarkerConfigError

if TYPE_CHECKING:
    from .config import MarkerConfig

_HEX = re.compile(r"#[0-9a-fA-F]{6}")


def rgb_or_none(value: object) -> RGB | None:
    """The color as three floats in [0, 1], or None when it is not a valid color."""
    if isinstance(value, str):
        if _HEX.fullmatch(value) is None:
            return None
        return (int(value[1:3], 16) / 255, int(value[3:5], 16) / 255, int(value[5:7], 16) / 255)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        if all(
            isinstance(entry, (int, float)) and not isinstance(entry, bool) and math.isfinite(entry)
            and 0 <= entry <= 1
            for entry in value
        ):
            return (float(value[0]), float(value[1]), float(value[2]))
    return None


def parse_color(value: object, *, where: str) -> RGB:
    color = rgb_or_none(value)
    if color is None:
        raise MarkerConfigError(f"invalid color {value!r} for {where}")
    return color


def hex_color(rgb: RGB) -> str:
    return "#%02x%02x%02x" % tuple(round(channel * 255) for channel in rgb)


def identity_color(config: MarkerConfig, value: Label | None) -> RGB:
    if value is None:
        return config.presentation.uncalled_color
    label = validate_label(config.contract, value)
    if not any(label):
        return config.presentation.negative_color
    total = [0.0, 0.0, 0.0]
    for marker, positive in zip(config.markers, label):
        if positive:
            for channel in range(3):
                total[channel] += marker.color[channel]
    return (min(1.0, total[0]), min(1.0, total[1]), min(1.0, total[2]))


def _linear(channel: float) -> float:
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def _luminance(rgb: RGB) -> float:
    red, green, blue = (_linear(channel) for channel in rgb)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def badge_colors(config: MarkerConfig, value: Label | None) -> tuple[str, str]:
    label = None if value is None else validate_label(config.contract, value)
    for badge in config.presentation.badge_colors:
        if badge.state == label:
            return hex_color(badge.background), hex_color(badge.foreground)
    background = hex_color(identity_color(config, label))
    shown = parse_color(background, where="badge background")
    return background, "#202020" if _luminance(shown) > 0.5 else "#ffffff"
