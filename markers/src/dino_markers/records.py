"""Versioned artifact records, their marker contracts, and directory-level contract resolution."""

from __future__ import annotations

import copy
import json
import os
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .config import MarkerConfig, config_to_json, default_config, load_config
from .contract import (
    CONTRACT_FIELD,
    DEFAULT_CONTRACT,
    MarkerConfigError,
    MarkerContract,
    MissingContractError,
    UnsupportedArtifactError,
    require_same_contract,
)


@dataclass(frozen=True)
class ArtifactKind:
    name: str
    version_key: str
    legacy: str
    current: str
    legacy_fields: Mapping[str, object] = field(default_factory=dict)

    def recognizes(self, record: Mapping[str, object]) -> bool:
        return record.get(self.version_key) in (self.legacy, self.current)

    def contract(self, record: Mapping[str, object], *, where: str) -> MarkerContract:
        version = record.get(self.version_key)
        if version == self.legacy:
            for key in (CONTRACT_FIELD, *self.legacy_fields):
                if key in record:
                    raise MarkerConfigError(f"{where}: {self.legacy} records must not carry {key}")
            return DEFAULT_CONTRACT
        if version == self.current:
            if CONTRACT_FIELD not in record:
                raise MissingContractError(
                    f"{where}: {self.name} {self.version_key}={self.current!r} has no {CONTRACT_FIELD}"
                )
            return MarkerContract.from_json(record[CONTRACT_FIELD], where=where)
        raise UnsupportedArtifactError(
            f"{where}: unsupported {self.name} {self.version_key}={version!r}; "
            f"expected {self.legacy!r} or {self.current!r}"
        )

    def stamp(self, fields: Mapping[str, object], contract: MarkerContract) -> dict[str, object]:
        for key in (self.version_key, CONTRACT_FIELD):
            if key in fields:
                raise ValueError(f"{self.name} fields must not contain {key!r}; stamp sets it")
        return {self.version_key: self.current, **fields, CONTRACT_FIELD: contract.to_json()}

    def normalize(self, record: Mapping[str, object], *, where: str) -> dict[str, object]:
        self.contract(record, where=where)
        if record[self.version_key] == self.current:
            return dict(record)
        return {
            **record,
            **copy.deepcopy(self.legacy_fields),
            self.version_key: self.current,
            CONTRACT_FIELD: DEFAULT_CONTRACT.to_json(),
        }


PREPARED_SOURCE = ArtifactKind(
    "prepared source completion", "artifact", "dino-zarr-source-complete-v1", "dino-zarr-source-complete-v2"
)
BARRAGE = ArtifactKind(
    "barrage record",
    "artifact",
    "dino-zarr-barrage-v3",
    "dino-zarr-barrage-v4",
    legacy_fields={"crop_panels": [["pi", "PI"], ["edu", "EdU"], ["brdu", "BrdU"]]},
)
FEATURE_CACHE = ArtifactKind("feature cache", "schema", "dino-zarr-features-v2", "dino-zarr-features-v3")
ZARR_MODEL = ArtifactKind("model bundle", "schema", "dino-zarr-model-v6", "dino-zarr-model-v7")
INTENSITY_MODEL = ArtifactKind(
    "intensity-context model bundle", "schema", "dino-knn-intensity-v1", "dino-knn-intensity-v2"
)
JOINT_MODEL = ArtifactKind("joint model bundle", "schema", "dino-joint-dataset-v1", "dino-joint-dataset-v2")
INFERENCE = ArtifactKind("inference record", "artifact", "dino-zarr-infer-v2", "dino-zarr-infer-v3")
CORTEX_RESCORE = ArtifactKind(
    "cortex rescore record", "artifact", "dino-cortex-rescore-v1", "dino-cortex-rescore-v2"
)
SPATIAL_REVIEW = ArtifactKind(
    "RHU2 spatial review manifest", "artifact", "rhu2-spatial-review-v1", "rhu2-spatial-review-v2"
)

MODEL_KINDS = (ZARR_MODEL, INTENSITY_MODEL, JOINT_MODEL)


def _model_kind(bundle: Mapping[str, object], where: str) -> ArtifactKind:
    for kind in MODEL_KINDS:
        if kind.recognizes(bundle):
            return kind
    expected = [version for kind in MODEL_KINDS for version in (kind.legacy, kind.current)]
    raise UnsupportedArtifactError(
        f"{where}: unsupported model bundle schema={bundle.get('schema')!r}; expected one of {expected!r}"
    )


def bundle_contract(bundle: Mapping[str, object], *, where: str) -> MarkerContract:
    return _model_kind(bundle, where).contract(bundle, where=where)


def derive_bundle(
    bundle: Mapping[str, object], updates: Mapping[str, object], *, where: str
) -> dict[str, object]:
    for key in ("schema", CONTRACT_FIELD):
        if key in updates:
            raise ValueError(f"derive_bundle updates must not contain {key!r}")
    return {**_model_kind(bundle, where).normalize(bundle, where=where), **updates}


CONTRACT_FILES: tuple[str, ...] = ("complete.json", "run.json", "spatial.json", "markers.json")

_FAMILIES = (("dino-zarr-source-complete-v", PREPARED_SOURCE), ("dino-zarr-barrage-v", BARRAGE))


def declared_contract(directory: Path) -> MarkerContract | None:
    found: list[tuple[Path, MarkerContract]] = []
    for name in ("complete.json", "run.json", "spatial.json"):
        path = directory / name
        if not path.is_file():
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            record = None
        if not isinstance(record, dict):
            if name == "complete.json":
                continue
            raise MarkerConfigError(f"{path}: {name} must be a JSON object")
        if name == "spatial.json":
            kind = SPATIAL_REVIEW
        else:
            artifact = record.get("artifact")
            kind = next(
                (
                    family
                    for prefix, family in _FAMILIES
                    if isinstance(artifact, str) and artifact.startswith(prefix)
                ),
                None,
            )
            if kind is None:
                if CONTRACT_FIELD in record:
                    raise UnsupportedArtifactError(
                        f"{path}: unrecognized record {artifact!r} carries {CONTRACT_FIELD}"
                    )
                continue
        found.append((path, kind.contract(record, where=str(path))))
    path = directory / "markers.json"
    if path.is_file():
        found.append((path, load_config(path).contract))
    if not found:
        return None
    first_path, first = found[0]
    for path, contract in found[1:]:
        require_same_contract(first, contract, expected_from=str(first_path), actual_from=str(path))
    return first


def contract_of(directory: Path) -> MarkerContract:
    contract = declared_contract(directory)
    return DEFAULT_CONTRACT if contract is None else contract


def resolve_directory(
    directory: Path, *, supplied: MarkerConfig | None = None, supplied_from: str = "--markers"
) -> MarkerConfig:
    if supplied is not None:
        config, config_from = supplied, supplied_from
    elif (directory / "markers.json").is_file():
        config, config_from = load_config(directory / "markers.json"), str(directory / "markers.json")
    else:
        config, config_from = default_config(), "packaged default markers.json"
    require_same_contract(
        contract_of(directory), config.contract, expected_from=str(directory), actual_from=config_from
    )
    return config


def save_config(config: MarkerConfig, directory: Path) -> Path:
    path = directory / "markers.json"
    text = json.dumps(config_to_json(config), indent=2) + "\n"
    temporary = directory / f".markers-{uuid.uuid4().hex}.tmp"
    handle = open(temporary, "x", encoding="utf-8")
    try:
        with handle:
            handle.write(text)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if load_config(path) != config:
                raise MarkerConfigError(
                    f"{path} differs from this run's marker configuration; "
                    "supply that file with --markers or use a new output directory"
                ) from None
    finally:
        os.unlink(temporary)
    return path


def marker_revision(directories: Iterable[Path]) -> str:
    def mtime(path: Path) -> int | None:
        try:
            return os.stat(path).st_mtime_ns
        except FileNotFoundError:
            return None

    return json.dumps(
        [[str(directory), [mtime(directory / name) for name in CONTRACT_FILES]] for directory in directories]
    )
