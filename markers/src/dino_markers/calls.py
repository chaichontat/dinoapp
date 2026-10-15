"""Labels, call codes, saved-state filters and label text."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import TYPE_CHECKING

from .contract import Label, MarkerContract, SavedFilter

if TYPE_CHECKING:
    from .config import MarkerConfig

MINUS = "\u2212"


def validate_label(contract: MarkerContract, value: object) -> Label:
    count = len(contract.markers)
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == count
        and all(isinstance(entry, Integral) and entry in (0, 1) for entry in value)
    ):
        return tuple(int(entry) for entry in value)
    raise ValueError(f"Marker values must be a {count}-tuple of binary labels, got {value!r}")


def call_weights(contract: MarkerContract) -> tuple[int, ...]:
    count = len(contract.markers)
    return tuple(2 ** (count - 1 - index) for index in range(count))


def uncalled_code(contract: MarkerContract) -> int:
    return 2 ** len(contract.markers)


def encode_call(contract: MarkerContract, value: Label) -> int:
    label = validate_label(contract, value)
    return sum(weight * entry for weight, entry in zip(call_weights(contract), label))


def decode_call(contract: MarkerContract, code: int) -> Label | None:
    sentinel = uncalled_code(contract)
    if not isinstance(code, Integral) or isinstance(code, bool) or not 0 <= code <= sentinel:
        raise ValueError(f"Call code must be an integer in 0..{sentinel}, got {code!r}")
    if code == sentinel:
        return None
    return tuple((int(code) // weight) % 2 for weight in call_weights(contract))


def toggle_marker(contract: MarkerContract, value: Label, marker: str) -> Label:
    label = validate_label(contract, value)
    if marker not in contract.markers:
        raise ValueError(f"Unknown marker {marker!r}; the contract has {list(contract.markers)!r}")
    index = contract.markers.index(marker)
    return label[:index] + (1 - label[index],) + label[index + 1 :]


def matches_saved(value: Label, saved: SavedFilter) -> bool:
    if len(value) != len(saved):
        raise ValueError(f"Saved filter {saved!r} must have one entry per marker of {value!r}")
    return all(entry is None or entry == actual for actual, entry in zip(value, saved))


def state_label(config: MarkerConfig, value: Label) -> str:
    label = validate_label(config.contract, value)
    return " / ".join(
        f"{marker.label}{'+' if positive else MINUS}" for marker, positive in zip(config.markers, label)
    )


def identity_name(config: MarkerConfig, value: Label | None) -> str:
    if value is None:
        return "No saved call"
    label = validate_label(config.contract, value)
    count = len(label)
    if not any(label):
        return "Neither" if count == 2 else "All negative"
    if all(label) and count >= 2:
        return "Dual positive" if count == 2 else "All positive"
    positives = " + ".join(marker.label for marker, positive in zip(config.markers, label) if positive)
    return f"{positives} only"
