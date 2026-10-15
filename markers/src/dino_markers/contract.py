"""The semantic marker contract: ordered keys, their rules, and the errors that report violations."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from .names import derived_names

RGB = tuple[float, float, float]
Label = tuple[int, ...]
SavedFilter = tuple[int | None, ...]

MAX_MARKERS: int = 7
CONTRACT_FIELD: str = "marker_contract"
RESERVED_KEYS: frozenset[str] = frozenset(
    {
        "cell",
        "combined",
        "rank",
        "dataset",
        "roi",
        "label",
        "x",
        "y",
        "z",
        "centroid_z",
        "centroid_y",
        "centroid_x",
        "plane_z",
        "area",
        "plane_area",
        "thumb_shape",
        "image_path",
        "source_sha256",
        "tranche",
        "split",
        "labels",
        "cell_sha256",
        "embedding_model",
        "joint_model",
    }
)

_KEY = re.compile(r"[a-z][a-z0-9_]*")


class MarkerConfigError(ValueError):
    """Invalid config or contract content."""


class ContractMismatchError(ValueError):
    """Two contracts differ."""


class MissingContractError(ValueError):
    """A current-version record has no marker_contract."""


class UnsupportedArtifactError(ValueError):
    """A version string is not recognized for its kind."""


def keys_problem(
    value: Mapping[str, object], required: set[str], optional: frozenset[str] = frozenset()
) -> str | None:
    """Describe unexpected and missing keys of a JSON object, or None when its keys are exact."""
    unexpected, missing = sorted(set(value) - required - optional), sorted(required - set(value))
    parts = []
    if unexpected:
        parts.append(f"unexpected keys {unexpected}")
    if missing:
        parts.append(f"missing keys {missing}")
    return "; ".join(parts) or None


def check_version(value: object) -> str | None:
    if type(value) is int and value == 1:
        return None
    return f"unsupported config version {value!r}; expected 1"


def key_problem(reference: object, markers: object) -> str | None:
    """The first rule of the key, uniqueness, count and derived-name checks that fails, or None."""
    if not isinstance(markers, tuple):
        return f"markers must be a tuple of keys, got {markers!r}"
    if not 1 <= len(markers) <= MAX_MARKERS:
        return f"between 1 and {MAX_MARKERS} markers are required, got {len(markers)}"
    keys = (reference, *markers)
    for key in keys:
        if not isinstance(key, str) or _KEY.fullmatch(key) is None:
            return f"key {key!r} must match [a-z][a-z0-9_]*"
        if key in RESERVED_KEYS:
            return f"key {key!r} is reserved"
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            return f"duplicate key {key!r}"
        seen.add(key)
    owners: dict[str, str] = {}
    for key in keys:
        for name in derived_names(key):
            if name in RESERVED_KEYS:
                return f"key {key!r} collides with {name!r} through {name!r}"
            owner = owners.setdefault(name, key)
            if owner != key:
                return f"key {owner!r} collides with {key!r} through {name!r}"
    return None


@dataclass(frozen=True)
class MarkerContract:
    reference: str
    markers: tuple[str, ...]

    def __post_init__(self) -> None:
        problem = key_problem(self.reference, self.markers)
        if problem is not None:
            raise MarkerConfigError(f"marker contract: {problem}")

    @property
    def channel_keys(self) -> tuple[str, ...]:
        return (self.reference, *self.markers)

    def to_json(self) -> dict[str, object]:
        return {"version": 1, "reference": self.reference, "markers": list(self.markers)}

    @classmethod
    def from_json(cls, value: object, *, where: str) -> MarkerContract:
        def invalid(problem: str) -> MarkerConfigError:
            return MarkerConfigError(f"{where}: invalid marker_contract: {problem}")

        if not isinstance(value, Mapping):
            raise invalid(f"expected a JSON object, got {value!r}")
        problem = keys_problem(value, {"version", "reference", "markers"})
        if problem is not None:
            raise invalid(problem)
        problem = check_version(value["version"])
        if problem is not None:
            raise invalid(problem)
        if not isinstance(value["markers"], list):
            raise invalid(f"markers must be a list of keys, got {value['markers']!r}")
        markers = tuple(value["markers"])
        problem = key_problem(value["reference"], markers)
        if problem is not None:
            raise invalid(problem)
        return cls(value["reference"], markers)

    def __str__(self) -> str:
        return f"reference {self.reference!r}, markers {list(self.markers)!r}"


DEFAULT_CONTRACT = MarkerContract("pi", ("brdu", "edu"))


def require_same_contract(
    expected: MarkerContract, actual: MarkerContract, *, expected_from: str, actual_from: str
) -> None:
    if expected != actual:
        raise ContractMismatchError(
            f"Marker contract mismatch: {actual_from} records {actual}, "
            f"but {expected_from} requires {expected}; marker order is part of the contract"
        )
