"""Identity colors and badge colors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .calls import validate_label
from .contract import RGB, Label

if TYPE_CHECKING:
    from .config import MarkerConfig

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
    rgb = identity_color(config, label)
    background = hex_color(rgb)
    shown = tuple(round(channel * 255) / 255 for channel in rgb)
    return background, "#202020" if _luminance(shown) > 0.5 else "#ffffff"
