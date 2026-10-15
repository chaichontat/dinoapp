"""The marker configuration file: schema, validation, packaged default."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from pathlib import Path

from .contract import (
    RGB,
    Label,
    MarkerConfigError,
    MarkerContract,
    check_version,
    key_problem,
    keys_problem,
)

_HEX = re.compile(r"#[0-9a-fA-F]{6}")


def rgb_or_none(value: object) -> RGB | None:
    """The color as three floats in [0, 1], or None when it is not a valid color."""
    if isinstance(value, str):
        if _HEX.fullmatch(value) is None:
            return None
        return (int(value[1:3], 16) / 255, int(value[3:5], 16) / 255, int(value[5:7], 16) / 255)
    if (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(e, (int, float)) and not isinstance(e, bool) and 0 <= e <= 1 for e in value)
    ):
        return (float(value[0]), float(value[1]), float(value[2]))
    return None


@dataclass(frozen=True)
class Channel:
    key: str
    label: str


@dataclass(frozen=True)
class Marker:
    key: str
    label: str
    color: RGB


@dataclass(frozen=True)
class BadgeColor:
    state: Label | None
    background: RGB
    foreground: RGB


@dataclass(frozen=True)
class Presentation:
    crop_panels: tuple[str, ...]
    negative_color: RGB
    uncalled_color: RGB
    badge_colors: tuple[BadgeColor, ...] = ()


def _panels_problem(keys: tuple[str, ...]) -> str:
    return f"crop_panels must list each of {list(keys)!r} exactly once"


def _color_problem(value: object, location: str) -> str | None:
    if not isinstance(value, tuple) or rgb_or_none(value) is None:
        return f"invalid color {value!r} for {location}"
    return None


def _badge_state_ok(state: object, count: int) -> bool:
    return state is None or (
        isinstance(state, tuple)
        and len(state) == count
        and all(isinstance(entry, int) and not isinstance(entry, bool) and entry in (0, 1) for entry in state)
    )


def config_problem(reference: Channel, markers: tuple[Marker, ...], presentation: Presentation) -> str | None:
    """The first rule of 1.4 that the typed configuration violates, or None."""
    if not isinstance(markers, tuple):
        return f"markers must be a tuple, got {markers!r}"
    keys = (reference.key, *(marker.key for marker in markers))
    problem = key_problem(reference.key, keys[1:])
    if problem is not None:
        return problem
    labels = [(reference.key, reference.label), *((marker.key, marker.label) for marker in markers)]
    for key, label in labels:
        if not isinstance(label, str) or label == "":
            return f"empty label for {key!r}"
    seen: set[str] = set()
    for _, label in labels:
        if label in seen:
            return f"duplicate label {label!r}"
        seen.add(label)
    panels = presentation.crop_panels
    if not (
        isinstance(panels, tuple)
        and len(panels) == len(keys)
        and all(isinstance(panel, str) for panel in panels)
        and set(panels) == set(keys)
    ):
        return _panels_problem(keys)
    for index, marker in enumerate(markers):
        problem = _color_problem(marker.color, f"markers[{index}].color")
        if problem is not None:
            return problem
    for name in ("negative_color", "uncalled_color"):
        problem = _color_problem(getattr(presentation, name), f"presentation.{name}")
        if problem is not None:
            return problem
    states: list[Label | None] = []
    for index, badge in enumerate(presentation.badge_colors):
        if not _badge_state_ok(badge.state, len(markers)):
            return f"invalid badge state {badge.state!r}"
        if badge.state in states:
            return f"duplicate badge state {badge.state!r}"
        states.append(badge.state)
        for name in ("background", "foreground"):
            problem = _color_problem(getattr(badge, name), f"presentation.badge_colors[{index}].{name}")
            if problem is not None:
                return problem
    return None


@dataclass(frozen=True)
class MarkerConfig:
    reference: Channel
    markers: tuple[Marker, ...]
    presentation: Presentation

    def __post_init__(self) -> None:
        problem = config_problem(self.reference, self.markers, self.presentation)
        if problem is not None:
            raise MarkerConfigError(f"marker config: {problem}")

    @property
    def contract(self) -> MarkerContract:
        return MarkerContract(self.reference.key, tuple(marker.key for marker in self.markers))

    @property
    def panels(self) -> tuple[tuple[str, str], ...]:
        return tuple((key, self.label(key)) for key in self.presentation.crop_panels)

    @property
    def display_markers(self) -> tuple[str, ...]:
        return tuple(key for key in self.presentation.crop_panels if key != self.reference.key)

    def label(self, key: str) -> str:
        if key == self.reference.key:
            return self.reference.label
        return self.marker(key).label

    def marker(self, key: str) -> Marker:
        for marker in self.markers:
            if marker.key == key:
                return marker
        raise KeyError(key)


def parse_config(value: object, *, source: str) -> MarkerConfig:
    def fail(problem: str) -> MarkerConfigError:
        return MarkerConfigError(f"{source}: {problem}")

    def obj(node: object, location: str, required: set[str], optional: frozenset[str] = frozenset()) -> dict:
        prefix = f"{location}: " if location else ""
        if not isinstance(node, dict):
            raise fail(f"{prefix}expected a JSON object, got {node!r}")
        problem = keys_problem(node, required, optional)
        if problem is not None:
            raise fail(f"{prefix}{problem}")
        return node

    def color(node: object, location: str) -> RGB:
        rgb = rgb_or_none(node)
        if rgb is None:
            raise fail(f"invalid color {node!r} for {location}")
        return rgb

    top = obj(value, "", {"version", "reference", "markers", "presentation"}, frozenset({"review_targets"}))
    problem = check_version(top["version"])
    if problem is not None:
        raise fail(problem)
    reference_node = obj(top["reference"], "reference", {"key", "label"})
    reference = Channel(reference_node["key"], reference_node["label"])
    if not isinstance(top["markers"], list):
        raise fail(f"markers must be a list, got {top['markers']!r}")
    markers = []
    for index, node in enumerate(top["markers"]):
        item = obj(node, f"markers[{index}]", {"key", "label", "color"})
        markers.append(Marker(item["key"], item["label"], color(item["color"], f"markers[{index}].color")))
    markers = tuple(markers)
    marker_keys = tuple(marker.key for marker in markers)
    problem = key_problem(reference.key, marker_keys)
    if problem is not None:
        raise fail(problem)
    presentation_node = obj(
        top["presentation"],
        "presentation",
        {"crop_panels", "negative_color", "uncalled_color"},
        frozenset({"badge_colors"}),
    )
    panel_node = presentation_node["crop_panels"]
    if not isinstance(panel_node, list):
        raise fail(_panels_problem((reference.key, *marker_keys)))
    badges = []
    badge_nodes = presentation_node.get("badge_colors", [])
    if not isinstance(badge_nodes, list):
        raise fail(f"badge_colors must be a list, got {badge_nodes!r}")
    for index, node in enumerate(badge_nodes):
        entry_location = f"presentation.badge_colors[{index}]"
        entry = obj(node, entry_location, {"state", "background", "foreground"})
        state = entry["state"]
        if state is not None:
            if not (
                isinstance(state, dict)
                and set(state) == set(marker_keys)
                and all(type(state[key]) is int and state[key] in (0, 1) for key in marker_keys)
            ):
                raise fail(f"invalid badge state {state!r}")
            state = tuple(state[key] for key in marker_keys)
        badges.append(
            BadgeColor(
                state,
                color(entry["background"], f"{entry_location}.background"),
                color(entry["foreground"], f"{entry_location}.foreground"),
            )
        )
    presentation = Presentation(
        tuple(panel_node),
        color(presentation_node["negative_color"], "presentation.negative_color"),
        color(presentation_node["uncalled_color"], "presentation.uncalled_color"),
        tuple(badges),
    )
    problem = config_problem(reference, markers, presentation)
    if problem is not None:
        raise fail(problem)
    return MarkerConfig(reference, markers, presentation)


def load_config(path: Path) -> MarkerConfig:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MarkerConfigError(f"{path}: invalid JSON: {error}") from error
    return parse_config(value, source=str(path))


@dataclass(frozen=True)
class ReviewTarget:
    """A dataset the reviewer opens with a named adapter, its paths resolved against its config file."""

    name: str
    adapter: str
    root: Path
    review_root: Path
    cell_prefix: str
    raw_channels: Mapping[str, str] = field(hash=False)
    embedding_run: Path | None = None


def load_review_targets(path: Path) -> tuple[ReviewTarget, ...]:
    """The review targets of a marker configuration file, validated against its contract."""
    path = Path(path)
    channels = load_config(path).contract.channel_keys
    nodes = json.loads(path.read_text(encoding="utf-8")).get("review_targets", [])
    base = path.absolute().parent

    def fail(problem: str) -> MarkerConfigError:
        return MarkerConfigError(f"{path}: {problem}")

    if not isinstance(nodes, list):
        raise fail(f"review_targets must be a list, got {nodes!r}")
    targets: list[ReviewTarget] = []
    for index, node in enumerate(nodes):
        location = f"review_targets[{index}]"
        if not isinstance(node, dict):
            raise fail(f"{location}: expected a JSON object, got {node!r}")
        problem = keys_problem(
            node, {"name", "adapter", "root", "review_root", "cell_prefix", "raw_channels"},
            frozenset({"embedding_run"}),
        )
        if problem is not None:
            raise fail(f"{location}: {problem}")
        for key in ("name", "adapter", "root", "review_root", "embedding_run"):
            if key in node and not (isinstance(node[key], str) and node[key]):
                raise fail(f"{location}.{key} must be a non-empty string, got {node[key]!r}")
        if not isinstance(node["cell_prefix"], str):
            raise fail(f"{location}.cell_prefix must be a string, got {node['cell_prefix']!r}")
        if any(target.name == node["name"] for target in targets):
            raise fail(f"duplicate review target name {node['name']!r}")
        raw = node["raw_channels"]
        if not (
            isinstance(raw, dict)
            and all(isinstance(value, str) for value in raw.values())
            and sorted(raw.values()) == sorted(channels)
        ):
            raise fail(f"{location}.raw_channels must map one raw array to each of {list(channels)!r}, got {raw!r}")
        targets.append(
            ReviewTarget(
                node["name"],
                node["adapter"],
                base / node["root"],
                base / node["review_root"],
                node["cell_prefix"],
                dict(raw),
                base / node["embedding_run"] if "embedding_run" in node else None,
            )
        )
    return tuple(targets)


@cache
def default_config() -> MarkerConfig:
    text = resources.files("dino_markers").joinpath("markers.json").read_text(encoding="utf-8")
    return parse_config(json.loads(text), source="packaged default markers.json")


def config_to_json(config: MarkerConfig) -> dict[str, object]:
    presentation: dict[str, object] = {
        "crop_panels": list(config.presentation.crop_panels),
        "negative_color": list(config.presentation.negative_color),
        "uncalled_color": list(config.presentation.uncalled_color),
    }
    if config.presentation.badge_colors:
        keys = config.contract.markers
        presentation["badge_colors"] = [
            {
                "state": None if badge.state is None else dict(zip(keys, badge.state)),
                "background": list(badge.background),
                "foreground": list(badge.foreground),
            }
            for badge in config.presentation.badge_colors
        ]
    return {
        "version": 1,
        "reference": {"key": config.reference.key, "label": config.reference.label},
        "markers": [
            {"key": marker.key, "label": marker.label, "color": list(marker.color)}
            for marker in config.markers
        ],
        "presentation": presentation,
    }
