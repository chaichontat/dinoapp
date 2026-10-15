# Marker contract specification

This spec binds the WF2 `dino_markers` implementer and slices A–I of `MARKER_CONFIG_PLAN.md` section 5. Slices code against it in parallel worktrees and may not change it. Where it departs from the plan's wording, the "Decisions" section gives the reason.

Line references are to commit `9042e7b` on `generalize`; source files there are identical to `c1e45f4`. The plan's references `zarr_train.py:475`, `zarr_barrage.py:487` and `zarr_infer.py:518` point to record literals. The equality checks are at `zarr_train.py:496`, `zarr_barrage.py:508` and `:515`, and `zarr_infer.py:518` and `:520`.

## Decisions

These are the points where the two WF1 drafts disagreed, where the plan's text needed refinement, or where the author's decisions D8–D12 (plan section 1) and the two Astra reviews changed the design.

1. Every new artifact records the contract (D8). It carries its family's current version string and a `marker_contract` field under every contract, including `DEFAULT_CONTRACT` (reference `pi`, markers `brdu, edu`). Legacy records are never rewritten. The spec applies D8 through two distinctions:
   - A resumed artifact is one whose run already holds a record: an interrupted inference, rescore or barrage run, an extended feature cache, or an existing journal file. Once its stored record passes the normalized comparison of 2.2, new code continues the run with the record it loaded, so no run mixes versions (2.3). A derived artifact is a new file built from an existing one, such as the classifier written by `periodic apply`/`measure` or the bundle returned by `bind_spatial_subtraction`. It is new, so it is written in the current version with its input's contract (`derive_bundle`, 1.5), and the input is left untouched.
   - Records of one run are compared with each other exactly: partition provenance against its run record, `complete.json` against `contract.json` or `run.json`. These comparisons are raw, so the versions within a run must agree. A stored record is compared with a record this code built after normalizing it (`ArtifactKind.normalize`, 1.5), because that comparison crosses the legacy and current representations of the same semantics (2.2).

   Fresh default outputs therefore differ from today's only by the version string and the added field, and 5.0 lists every WF0 assertion this changes. Records that hold no marker identity are not stamped: `source.json` (its digest identifies the source), chunk partials, `display.json` caches and embedding partitions (covered by their run's records). The periodic field bundle, `correction.json` and the measure records are an author-approved exception to D8 (D12). They keep their versions, and the field bundle's ordered `targets` list is their recorded contract, checked against the active contract wherever it is bound (4.4). `correction.json` and the measure records identify the field bundle by hash and carry no contract of their own. The exception does not cover the classifier bundles that `periodic apply`/`measure` write, which are derived artifacts.
2. The field is named `marker_contract` on every record, including journal records. The plan names the journal field `markers`, but the contract JSON has its own `markers` key, which would make `record["markers"]["markers"]`.
3. `correct_channels` keeps an explicit `reference` argument naming the crop channel that holds the subtraction reference. Prod3 crops carry `pi` = 561 nm and `dapi` = the source's 405 nm channel and subtract against `dapi` (`dinoapp/src/dinoapp/prod3.py:93-96`, `:211`). The compat draft read the reference from the subtraction identity (`pi`), which would subtract 561 nm from BrdU in Prod3.
4. Writers distinguish creating an artifact from resolving an existing one. `declared_contract(directory)` returns `None` when the directory declares nothing, so a fresh barrage destination inherits its source's contract; `contract_of(directory)` resolves an existing artifact and gives a directory that declares nothing the default contract (legacy directories, hand-made barrages). Only the dino record families declare a contract, because `dinoapp/docs/data-format.md` says the app does not read a hand-made `complete.json`, and the production RHU2 crop directory `/working/chaichontat/rhu2-barrage-review` holds `run.json` and `complete.json` with artifact `solarmap-native-barrage-v1`. A foreign record that carries `marker_contract`, and a `run.json` that is not a JSON object, fail instead of falling back to the default (1.6).
5. Records without a version are not given the default contract in general. The one exception is the native adapters' prediction manifest: a manifest with no `artifact` key and no `marker_contract` reads as the default contract, and one that carries `marker_contract` is rejected. Today's adapters do not check `artifact`; the WF0 native builder writes such a manifest (`test_characterization_review.py::build_native_root`), and the Prod3-2/Prod2-2 adapter uses the unversioned reembedding `complete.json` as its manifest when no pointer is set (`prod3_2.py:56-57`). Production pointer manifests are `dino-cortex-rescore-v1`.
6. The packaged default writes `negative_color` as `[0.25, 0.25, 0.25]`. The plan's `#404040` is 64/255 = 0.25098, but plan 2.5 requires today's palette exactly (`dinoapp/src/dinoapp/native.py:39-41`, pinned by `test_identity_palette_is_neither_edu_brdu_dual_uncalled`). Colors therefore accept `#rrggbb` or a float triple.
7. `rhu2_prepare` writes `rhu2-spatial-review-v2` with `marker_contract` equal to `DEFAULT_CONTRACT`. It cannot write another contract, because it reads the frozen RHU2 release's `brdu_prob`/`edu_prob` (`rhu2_prepare.py:48`, D4). A v1 snapshot normalizes to the default contract, and `review_activation` updates a snapshot in place in the version it finds (`review_activation.py:49-57`).
8. A barrage records its crop panels, because panel order and titles change the rendered PNGs. Every v4 record carries `crop_panels` (`[key, label]` pairs), and a v3 record normalizes to PI | EdU | BrdU. Colors are not recorded, because barrage PNGs use magma (`crop_rendering.py:19`).
9. The barrage display cache requires `vmax` keys equal to the source's channel set. `_measure_display_max` measures every source channel (`zarr_barrage.py:115`), while today's check requires exactly `{pi, edu, brdu}` (`:156-161`).
10. `call_provenance.load_calls` reads the contract from the run's records. `cortex_score.score_rows` takes `markers` explicitly. The WF0 test bundle for `score_rows` has no `schema`, and its invocation helper can pass the markers.
11. `SpatialPixels` gains read-only `target`/`reference` properties returning its identity's constants. Callers read them from an instance and only when a subtraction exists: the correction branch of `read_model_images` also runs for periodic-only correction (`zarr_model.py:183`), and `model_crop_reader` discards the instance it builds today (`:150`). `from_spec` already rejects any other identity by dict equality (`spatial_pixels.py:49`), so the constructor is unchanged.
12. A contract change for an active review target discards and recreates its session context, which gives a new session id, journal and backup. `BarrageSource.source_id` is unchanged because a WF0 test pins it. Every target's review revision includes `marker_revision` of its annotation directories, and the barrage, native and RHU2 reader caches and the native field cache key on the same value (3.2), so any config change rebuilds the readers and sampled fields and drops captured queues and `FieldCropSource` objects.
13. Two text helpers exist because today shows two forms: `state_label` gives "BrdU− / EdU+" (`sources.py:102-108`) and `identity_name` gives "EdU only" (`native.py:42`).
14. Signatures stay compatible only where a WF0 test body or stub calls the API directly: `load_labels(path)`, `BarrageSource(root)`, `correction_rows(..., identity=...)`, `read_crop(reader, row, *, size)`, `DisplayPixels(model_path, source_dir)` and `crop(..., *, reference)`, `read_embedding_artifact(path)`, plus the `IntensityContext` constructor used by frozen RHU2 code. Every other changed API is reached through a WF0 invocation helper, which the owning slice adapts (section 5.0).
15. `training_split` gains `split_dataset_rows` with integer strata for `joint_train`, which reproduce `joint_train.py:236`. `split_training_rows` keeps string strata, which reproduce today's default split.
16. Under the default contract, a training target given no `--input-channel` keeps today's per-target default (`zarr_train.py:405-408`). Under any other contract every marker must be given.
17. `dino/dino/review_activation.py`, which calls `combined_uncertainty` (`:36`), is assigned to slice I. Plan section 5 assigns it to no slice.
18. Keys exclude every name the code already uses as a role or record key in a namespace that marker keys or their derived names share: the crop-cache role `labels` (`zarr_source.py:513`), the embedding digest `cell_sha256` (`zarr_infer.py:618`), the training-frame columns `_source_order`, `_feature_order` and `_missing_targets` (`zarr_train.py:257-258`, `:450`), and the deployed-bundle keys `embedding_model` and `joint_model` (`joint_train.py:318`, `:323`), which `model_key` would produce for markers named `embedding` and `joint`. Keys therefore start with a letter, and `RESERVED_KEYS` lists the other names (1.3).
19. Badge colors follow D9. `presentation.badge_colors` optionally overrides the badge of named states, and every other state uses its identity color with a foreground chosen by luminance, so no config enumerates the 2^N states. The packaged default overrides EdU-only and "no saved call"; with the fallback it reproduces all five badges of `native_app.py:39-46` (1.9).
20. Out-of-repo callers move to the new APIs without aliases (D7, D10, D11). Slice C owns `dino_neighbor_eval.py` and `dino_train_405.sh`; slice D owns `dino_render_calls.py`, `test_dino_call_provenance.py`, `thesis/gam/gam/uv_soft_joint.py` and `thesis/scripts/prepare_uv_gam.py` (section 6).
21. New journal files carry an explicit version. Field presence alone cannot tell a legacy journal record from a damaged new record that lost `marker_contract`, so every new record carries `schema: "dino-review-journal-v1"` with the field. A journal file whose first record has no `schema` is legacy and continues only under `DEFAULT_CONTRACT` (4.12).

## 1. The `markers/` workspace member

### 1.1 Layout and packaging

```text
markers/
├── pyproject.toml
├── src/dino_markers/
│   ├── __init__.py      re-exports every public name in this section
│   ├── config.py        Channel, Marker, BadgeColor, Presentation, MarkerConfig, parse/load/default, errors
│   ├── contract.py      MarkerContract, DEFAULT_CONTRACT, require_same_contract
│   ├── records.py       ArtifactKind, kind constants, bundle_contract, derive_bundle, declared_contract,
│   │                    contract_of, resolve_directory, save_config, marker_revision
│   ├── names.py         derived column and key names
│   ├── calls.py         labels, call codes, filters, text
│   ├── colors.py        color parsing, identity and badge colors
│   └── markers.json     packaged default
└── tests/
```

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "dino-markers"
version = "0.1.0"
description = "Marker configuration and artifact contracts shared by thesis-dino and dinoapp"
requires-python = ">=3.10"
dependencies = []

[tool.hatch.build.targets.wheel]
packages = ["src/dino_markers"]

[tool.ruff]
line-length = 110
target-version = "py310"
```

Root `pyproject.toml`: `members = ["dinoapp", "dino", "markers"]`, `[tool.uv.sources]` gains `dino-markers = { workspace = true }`, and `testpaths` gains `"markers/tests"`. `dino/pyproject.toml` and `dinoapp/pyproject.toml` add `"dino-markers>=0.1.0"` to `[project].dependencies`. For dinoapp this is a core dependency, not the `native` extra, because basic crop review needs it without `thesis-dino`. The package uses only the standard library and Python 3.10 syntax (no `typing.Self`, `StrEnum`, `tomllib`, PEP 695).

### 1.2 Packaged default `markers.json`

```json
{
  "version": 1,
  "reference": {"key": "pi", "label": "PI"},
  "markers": [
    {"key": "brdu", "label": "BrdU", "color": "#ff00ff"},
    {"key": "edu", "label": "EdU", "color": "#00ff00"}
  ],
  "presentation": {
    "crop_panels": ["pi", "edu", "brdu"],
    "negative_color": [0.25, 0.25, 0.25],
    "uncalled_color": "#999999",
    "badge_colors": [
      {"state": {"brdu": 0, "edu": 1}, "background": "#008f3a", "foreground": "#ffffff"},
      {"state": null, "background": "#777777", "foreground": "#ffffff"}
    ]
  }
}
```

`#999999` is 153/255 = 0.6 exactly, `#ff00ff` and `#00ff00` are exact, and the float triple gives 0.25 (decision 6). The two badge overrides are today's EdU-only and no-call badges (`native_app.py:39-46`); the other three follow from the fallback rule of 1.9 (decision 19).

### 1.3 Types

```python
RGB = tuple[float, float, float]
Label = tuple[int, ...]              # one 0/1 per contract marker, contract order
SavedFilter = tuple[int | None, ...] # one entry per contract marker; None matches 0 or 1

MAX_MARKERS: int = 7
CONTRACT_FIELD: str = "marker_contract"
RESERVED_KEYS: frozenset[str] = frozenset({
    "cell", "combined", "rank", "dataset", "roi", "label", "x", "y", "z",
    "centroid_z", "centroid_y", "centroid_x", "plane_z", "area", "plane_area",
    "thumb_shape", "image_path", "source_sha256", "tranche", "split",
    "labels", "cell_sha256", "embedding_model", "joint_model",
})   # decision 18; derived names are checked against this set too (1.4)

@dataclass(frozen=True)
class Channel:          # the reference
    key: str
    label: str

@dataclass(frozen=True)
class Marker:
    key: str
    label: str
    color: RGB

@dataclass(frozen=True)
class BadgeColor:
    state: Label | None          # None is "no saved call"
    background: RGB
    foreground: RGB

@dataclass(frozen=True)
class Presentation:
    crop_panels: tuple[str, ...]
    negative_color: RGB
    uncalled_color: RGB
    badge_colors: tuple[BadgeColor, ...] = ()   # absent in JSON means no overrides

@dataclass(frozen=True)
class MarkerContract:
    reference: str
    markers: tuple[str, ...]
    # __post_init__ applies the key, uniqueness, count and reference-not-marker rules of 1.4
    @property
    def channel_keys(self) -> tuple[str, ...]: ...   # (reference, *markers)
    def to_json(self) -> dict[str, object]: ...      # {"version": 1, "reference": ..., "markers": [...]}
    @classmethod
    def from_json(cls, value: object, *, where: str) -> MarkerContract: ...
    def __str__(self) -> str: ...                    # "reference 'pi', markers ['brdu', 'edu']"

@dataclass(frozen=True)
class MarkerConfig:
    reference: Channel
    markers: tuple[Marker, ...]
    presentation: Presentation
    # __post_init__ applies every rule in 1.4; direct construction cannot skip validation
    @property
    def contract(self) -> MarkerContract: ...
    @property
    def panels(self) -> tuple[tuple[str, str], ...]: ...   # (key, label) in crop_panels order
    @property
    def display_markers(self) -> tuple[str, ...]: ...      # crop_panels order without the reference
    def label(self, key: str) -> str: ...                  # reference or marker; KeyError otherwise
    def marker(self, key: str) -> Marker: ...

DEFAULT_CONTRACT: MarkerContract   # MarkerContract("pi", ("brdu", "edu"))

def parse_config(value: object, *, source: str) -> MarkerConfig
def load_config(path: Path) -> MarkerConfig          # source=str(path); JSON errors become MarkerConfigError
def default_config() -> MarkerConfig                  # packaged file via importlib.resources, cached
def config_to_json(config: MarkerConfig) -> dict[str, object]

@dataclass(frozen=True)
class ReviewTarget:
    name: str
    adapter: str
    root: Path
    review_root: Path
    cell_prefix: str
    raw_channels: Mapping[str, str]       # excluded from the hash
    embedding_run: Path | None = None

def load_review_targets(path: Path) -> tuple[ReviewTarget, ...]
def require_same_contract(expected: MarkerContract, actual: MarkerContract, *,
                          expected_from: str, actual_from: str) -> None
```

`config_to_json` writes every color as a list of its three floats and writes `badge_colors` only when it is nonempty (a state as an object keyed by marker in contract order), so `parse_config(config_to_json(c), source=...) == c`. Hexadecimal remains an accepted input and the UI form (`hex_color`, 1.9), and the packaged default (1.2) keeps its hexadecimal colors. `MarkerConfig` equality is dataclass equality: two configs that differ only in a color are unequal configs with equal contracts. `default_config().contract == DEFAULT_CONTRACT` and `default_config().display_markers == ("edu", "brdu")`.

A marker configuration file may also carry an optional top-level `review_targets` list, the datasets the reviewer offers (3.2). Review targets are not part of `MarkerConfig` or the marker contract: `parse_config` accepts the key and ignores it, so `config_to_json` and `save_config` never write it and a directory's stored `markers.json` never carries targets. `load_review_targets(path)` loads the file's config and validates each entry: an object with the keys `name` (a non-empty string, unique in the file), `adapter` (a non-empty string; this package does not know adapter names), `root` and `review_root` (non-empty path strings), `cell_prefix` (a string), `raw_channels` (an object mapping raw array names to channel keys, one raw array for each of the file's `contract.channel_keys`), and optionally `embedding_run` (a non-empty path string). Relative paths resolve against the directory that contains the file; absolute paths are kept. Any other or missing key raises `MarkerConfigError` with the file path, the entry location and `unexpected keys [...]` / `missing keys [...]`. A file without `review_targets` gives `()`. The packaged default has none.

### 1.4 Validation and errors

```python
class MarkerConfigError(ValueError): ...       # invalid config or contract content
class ContractMismatchError(ValueError): ...   # two contracts differ
class MissingContractError(ValueError): ...    # current-version record without marker_contract
class UnsupportedArtifactError(ValueError): ...# version string not recognized for its kind
```

All four subclass `ValueError`, because existing callers catch `ValueError` to show or reject an artifact (`dinoapp/src/dinoapp/app.py:755`, `:769`). Every message starts with `f"{source}: "` (or `where`).

| Rule | Message content |
| --- | --- |
| Top level has exactly `version`, `reference`, `markers`, `presentation`, and optionally `review_targets` (validated only by `load_review_targets`, 1.3); nested objects have exactly the keys in 1.2, where `presentation.badge_colors` is optional | `unexpected keys [...]` / `missing keys [...]` |
| `version == 1` | `unsupported config version {v!r}; expected 1` |
| Each key is a string matching `^[a-z][a-z0-9_]*$` (this excludes the `\|` cell delimiter and the leading-underscore frame columns of decision 18) | `key {k!r} must match [a-z][a-z0-9_]*` |
| No key in `RESERVED_KEYS` | `key {k!r} is reserved` |
| Keys unique across reference and markers | `duplicate key {k!r}` |
| `1 <= len(markers) <= MAX_MARKERS` | `between 1 and 7 markers are required, got {n}` |
| Derived names (1.7) of different keys are pairwise disjoint and disjoint from `RESERVED_KEYS` | `key {a!r} collides with {b!r} through {name!r}` (for example markers `brdu` and `p_brdu`, or marker `embedding` through `embedding_model`) |
| Labels are nonempty strings, unique across reference and markers | `empty label for {k!r}` / `duplicate label {label!r}` |
| `crop_panels` is a permutation of `channel_keys` | `crop_panels must list each of {keys} exactly once` |
| A color is `#` plus six hex digits, or a list of three finite numbers in [0, 1] (`bool` rejected) | `invalid color {value!r} for {where}` |
| Each `badge_colors` entry has exactly `state`, `background`, `foreground`; `state` is `null` or an object with exactly the marker keys, each `0` or `1` (`bool` rejected); states are unique | `invalid badge state {value!r}` / `duplicate badge state {state!r}` |

`MarkerContract.from_json` requires exactly the keys `version`, `reference`, `markers`, `version == 1`, and the key rules, raising `MarkerConfigError(f"{where}: invalid marker_contract: {problem}")`.

`require_same_contract` passes only when reference and the ordered marker tuple are equal, so a same-length reorder fails. Otherwise:

```text
ContractMismatchError: Marker contract mismatch: {actual_from} records reference 'pi', markers ['edu', 'brdu'], but {expected_from} requires reference 'pi', markers ['brdu', 'edu']; marker order is part of the contract
```

### 1.5 Artifact kinds, recording and normalization (`records.py`)

```python
@dataclass(frozen=True)
class ArtifactKind:
    name: str                                  # used in messages, e.g. "barrage record"
    version_key: str                           # "artifact" or "schema"
    legacy: str                                # implies DEFAULT_CONTRACT
    current: str                               # requires CONTRACT_FIELD
    legacy_fields: Mapping[str, object] = field(default_factory=dict)   # supplied by normalize

    def recognizes(self, record: Mapping[str, object]) -> bool
    def contract(self, record: Mapping[str, object], *, where: str) -> MarkerContract
    def stamp(self, fields: Mapping[str, object], contract: MarkerContract) -> dict[str, object]
    def normalize(self, record: Mapping[str, object], *, where: str) -> dict[str, object]
```

- `recognizes`: true when `record.get(version_key)` equals `legacy` or `current`, so a missing or `null` version key is never recognized.
- `contract`: with the legacy version and neither `CONTRACT_FIELD` nor a `legacy_fields` key, returns `DEFAULT_CONTRACT`. With the legacy version and either present, raises `MarkerConfigError(f"{where}: {legacy} records must not carry {key}")`. With the current version, parses the field, and if the field is absent raises `MissingContractError(f"{where}: {name} {version_key}={current!r} has no marker_contract")`. Any other value, including a missing or `null` version key, raises `UnsupportedArtifactError(f"{where}: unsupported {name} {version_key}={value!r}; expected {legacy!r} or {current!r}")`.
- `stamp`: the only way a writer builds a new record from fields. `fields` must not contain `version_key` or `CONTRACT_FIELD` (`ValueError`). Returns `{version_key: current, **fields, CONTRACT_FIELD: contract.to_json()}`.
- `normalize`: the only normalization helper. It calls `contract` first, so every error above applies. A current record is returned as `dict(record)`. A legacy record becomes `{**record, **legacy_fields, version_key: current, CONTRACT_FIELD: DEFAULT_CONTRACT.to_json()}`.

Dict equality ignores key order, and every check in 2.2 compares dicts, so the position of `marker_contract` in a written record is not part of the contract. Writers that sort keys today (`write_json_atomic`, `zarr_source.py:66`) keep sorting them.

```python
PREPARED_SOURCE = ArtifactKind("prepared source completion", "artifact",
                               "dino-zarr-source-complete-v1", "dino-zarr-source-complete-v2")
BARRAGE         = ArtifactKind("barrage record", "artifact", "dino-zarr-barrage-v3", "dino-zarr-barrage-v4",
                               legacy_fields={"crop_panels": [["pi", "PI"], ["edu", "EdU"], ["brdu", "BrdU"]]})
FEATURE_CACHE   = ArtifactKind("feature cache", "schema", "dino-zarr-features-v2", "dino-zarr-features-v3")
ZARR_MODEL      = ArtifactKind("model bundle", "schema", "dino-zarr-model-v6", "dino-zarr-model-v7")
INTENSITY_MODEL = ArtifactKind("intensity-context model bundle", "schema",
                               "dino-knn-intensity-v1", "dino-knn-intensity-v2")
JOINT_MODEL     = ArtifactKind("joint model bundle", "schema", "dino-joint-dataset-v1", "dino-joint-dataset-v2")
INFERENCE       = ArtifactKind("inference record", "artifact", "dino-zarr-infer-v2", "dino-zarr-infer-v3")
CORTEX_RESCORE  = ArtifactKind("cortex rescore record", "artifact",
                               "dino-cortex-rescore-v1", "dino-cortex-rescore-v2")
SPATIAL_REVIEW  = ArtifactKind("RHU2 spatial review manifest", "artifact",
                               "rhu2-spatial-review-v1", "rhu2-spatial-review-v2")

MODEL_KINDS = (ZARR_MODEL, INTENSITY_MODEL, JOINT_MODEL)

def bundle_contract(bundle: Mapping[str, object], *, where: str) -> MarkerContract
    # the kind in MODEL_KINDS that recognizes bundle["schema"]; else UnsupportedArtifactError
def derive_bundle(bundle: Mapping[str, object], updates: Mapping[str, object], *,
                  where: str) -> dict[str, object]
    # {**kind.normalize(bundle, where=where), **updates} for that same kind
```

`derive_bundle` builds a new bundle from an existing one (decision 1). The result is in the current version with the input's contract: a `dino-knn-intensity-v1` input yields `dino-knn-intensity-v2` with `DEFAULT_CONTRACT`. `updates` must not contain `schema` or `CONTRACT_FIELD` (`ValueError`), and the input mapping is not modified. Its callers are `periodic apply` and `measure` and `bind_spatial_subtraction` (4.4).

No code encodes a legacy record from current fields; a resumed run keeps the record it loaded (2.3). Inference and barrage runs reuse their stored run record, and a rescore run reuses the items of its stored record whose keys `expected` has, once that record passes the normalized comparison of 2.2; reusing a feature cache keeps the metadata dict it loaded when entries are added (`zarr_train.py:325`), and `review_activation` updates an existing RHU2 `spatial.json` in place in the version it finds (`review_activation.py:49-57`). Appending to a legacy journal file is governed by 4.12.

### 1.6 Directory contracts and stored configs

```python
CONTRACT_FILES: tuple[str, ...] = ("complete.json", "run.json", "spatial.json", "markers.json")

def declared_contract(directory: Path) -> MarkerContract | None
def contract_of(directory: Path) -> MarkerContract
def resolve_directory(directory: Path, *, supplied: MarkerConfig | None = None,
                      supplied_from: str = "--markers") -> MarkerConfig
def save_config(config: MarkerConfig, directory: Path) -> Path
def marker_revision(directories: Iterable[Path]) -> str
```

`declared_contract` collects every contract the directory declares and requires them to agree.

1. `complete.json` and `run.json`, if present, are classified one by one.
   - A `complete.json` that does not decode (exactly `UnicodeDecodeError` or `json.JSONDecodeError`) or is not a JSON object is an opaque hand-made completion (`dinoapp/docs/data-format.md`) and is skipped. A `run.json` in that state raises `MarkerConfigError(f"{path}: run.json must be a JSON object")`, because only producers write it.
   - An object whose `artifact` is a string starting with `dino-zarr-source-complete-v` or `dino-zarr-barrage-v` goes through `PREPARED_SOURCE.contract` or `BARRAGE.contract`, so an unknown version in a family raises `UnsupportedArtifactError`.
   - Any other object (no `artifact`, a non-string `artifact`, or a foreign family such as `solarmap-native-barrage-v1`) is skipped when it has no `marker_contract` key. With the key it raises `UnsupportedArtifactError(f"{path}: unrecognized record {artifact!r} carries marker_contract")`.
2. `spatial.json`, if present, must be a JSON object that `SPATIAL_REVIEW` recognizes; its contract is `SPATIAL_REVIEW.contract(...)`.
3. `markers.json`, if present, contributes `load_config(...).contract`. An invalid file raises.
4. Contracts that differ raise `ContractMismatchError` naming both files. If none were found (a missing or empty directory, a legacy directory without records, a hand-made barrage without `markers.json`), the result is `None`.

`contract_of` returns `declared_contract(directory)`, or `DEFAULT_CONTRACT` when that is `None`. Readers of existing artifacts use `contract_of`. Writers use `declared_contract` for their destination, so a destination that declares nothing takes the contract of the run's inputs (4.5, 4.6).

`resolve_directory` returns the presentation config for the directory: `supplied` if given, else the directory's `markers.json`, else `default_config()`. The chosen config's contract must equal `contract_of(directory)`, checked by `require_same_contract` with `actual_from` set to `supplied_from`, the `markers.json` path, or `"packaged default markers.json"`. A supplied config therefore only asserts compatibility; it never reinterprets an artifact.

`save_config` writes `directory / "markers.json"` as `json.dumps(config_to_json(config), indent=2) + "\n"` through a temporary file and `os.link`, which never overwrites. If the file exists and `load_config(path) == config`, it does nothing; if the file exists and differs, it raises `MarkerConfigError(f"{path} differs from this run's marker configuration; supply that file with --markers or use a new output directory")`.

`marker_revision` returns `json.dumps([[str(directory), [mtime_ns or None for each name in CONTRACT_FILES]] for directory in directories])`, with `mtime_ns` from `os.stat(directory / name).st_mtime_ns` and `None` for a missing file. It is never persisted; reviewer caches and review revisions key on it (3.2).

### 1.7 Derived names (`names.py`)

Every persisted per-marker name derives from the key, and under the default contract each reproduces today's name.

| Function | Value | Persisted today in |
| --- | --- | --- |
| (key itself) | `k` | `labels.csv` columns (`label_store.py:19`), prediction call columns (`zarr_infer.py:340-341`, `cortex_score.py:70`), `display_max`/`vmax` keys, embedding tensor keys, `input_channel_names_by_target` keys, RHU2 `decision_thresholds` keys, joint metrics `targets` keys |
| `probability_column(k)` | `p_{k}` | predictions, labeled review |
| `prediction_column(k)` | `pred_{k}` | labeled review (`review_contracts.py:84`), joint `labeled_predictions.parquet` |
| `oof_probability_column(k)` | `oof_p_{k}` | `joint_train.py:278` |
| `saved_column(k)` | `saved_{k}` | dinoapp gallery frames (`native.py:154-155`) |
| `miscall_column(k)` | `{k}_miscall` | `native.py:97-100` |
| `rhu2_probability_column(k)` | `{k}_prob` | RHU2 tables, `spatial.json` `probability_columns` |
| `image_path_column(k)` | `{k}_image_path` | barrage `index.csv` (`sources.py:176`) |
| `cutoff_key(k)` | `{k}_cutoff` | bundles, inference and rescore records, labeled review manifest |
| `model_key(k)`, `scaler_key(k)` | `{k}_model`, `{k}_feature_scaler` | bundles |
| `metric_key(k, metric)` | `{k}_{metric}`, metric in `CLASSIFIER_METRICS = ("best_c", "cv_balanced_accuracy", "cv_folds", "validation_balanced_accuracy")` | bundles (`zarr_train.py:555-562`) |
| `positive_count_key(k)` | `{k}_positive` | rescore `complete.json` (`cortex_score.py:276-277`) |
| `stat_column(k, stat)` | `{k}_{stat}`, stat in `BARRAGE_STATS = ("mean", "min", "max", "median", "std")` | barrage `index.csv`, `candidates.parquet` |
| `intensity_column(k, stat="mean")` | `intensity_{k}_{stat}` | regionprops tables (`intensity_context.py:30-37`, `periodic.py:69-72`) |

The feature-cache payload `f"{target}|{cell}|..."` is unchanged (`zarr_train.py:203-208`).

Column order: wherever today writes per-marker blocks, the generic writer iterates markers in contract order and keeps the within-marker order. Inference predictions are `cell, dataset, roi, label, *p_k, *k` (`zarr_infer.py:337-341`); rescore rows interleave `p_k, k` per marker (`cortex_score.py:65-70`); barrage statistics are `k_mean … k_std` per marker (`zarr_barrage.py:284`, `:594-596`).

### 1.8 Labels, call codes, filters and text (`calls.py`)

```python
def validate_label(contract: MarkerContract, value: object) -> Label
    # sequence of len(contract.markers) entries, each 0/1 (bool accepted, returned as int);
    # else ValueError(f"Marker values must be a {n}-tuple of binary labels, got {value!r}")
def call_weights(contract: MarkerContract) -> tuple[int, ...]   # (2**(N-1), ..., 2, 1)
def encode_call(contract: MarkerContract, value: Label) -> int
def decode_call(contract: MarkerContract, code: int) -> Label | None   # None for uncalled_code
def uncalled_code(contract: MarkerContract) -> int                     # 2**N
def toggle_marker(contract: MarkerContract, value: Label, marker: str) -> Label
def matches_saved(value: Label, saved: SavedFilter) -> bool            # entries equal or None
def state_label(config: MarkerConfig, value: Label) -> str
def identity_name(config: MarkerConfig, value: Label | None) -> str
```

A call code is the binary number of the label with the first contract marker as the most significant bit; "no call" is `2**N`. Code arrays are `numpy.uint8`, and consumers vectorize as `values.astype(np.uint8) @ np.asarray(call_weights(contract), dtype=np.uint8)` with `values` of shape `(n, N)` in contract order. `MAX_MARKERS = 7` because codes `0 … 2**N - 1` and the sentinel `2**N` must fit in `uint8`; at N = 8 the sentinel would be 256. For the default contract the codes are `(0,0)→0`, `(0,1)→1`, `(1,0)→2`, `(1,1)→3`, uncalled `4`: today's `edu + 2 * brdu` with sentinel 4 (`native.py:363-366`, `:539-547`, `prod3.py:152-156`, `prod3_2.py:156-157`, `rhu2.py:155-159`). WF0 pins this in `test_native_prediction_loading_encodes_calls_as_edu_plus_twice_brdu` and `test_field_overlay_replaces_predictions_with_saved_labels_in_the_same_encoding`. No code path enumerates the `2**N` identities: filters are per marker, and colors are computed per distinct code present (1.9).

`state_label` joins `f"{label}{'+' if v else '−'}"` over markers in contract order with `" / "`, using U+2212 for minus. The default `(0, 1)` gives `"BrdU− / EdU+"` (`sources.py:102-108`).

`identity_name` reproduces `native.py:42` for the default contract:

| Value | Text |
| --- | --- |
| `None` | `"No saved call"` |
| no positives | `"Neither"` if N == 2, else `"All negative"` |
| all positive, N ≥ 2 | `"Dual positive"` if N == 2, else `"All positive"` |
| otherwise | `" + ".join(positive labels in contract order) + " only"` |

### 1.9 Colors and the field composite (`colors.py`)

```python
def hex_color(rgb: RGB) -> str                                    # "#%02x%02x%02x" of round(c * 255)
def identity_color(config: MarkerConfig, value: Label | None) -> RGB
def badge_colors(config: MarkerConfig, value: Label | None) -> tuple[str, str]   # (background, foreground) hex
```

`badge_colors` returns the `presentation.badge_colors` entry whose `state` equals `value`, as `hex_color` strings. For any other state the background is `hex_color(identity_color(config, value))` and the foreground is `"#202020"` when the background's relative luminance exceeds 0.5, else `"#ffffff"`. Relative luminance is `0.2126 R + 0.7152 G + 0.0722 B` over linearized channels, `c / 12.92` for `c <= 0.04045` and `((c + 0.055) / 1.055) ** 2.4` otherwise. For the default config, codes 0 to 4 give `("#404040", "#ffffff")`, `("#008f3a", "#ffffff")`, `("#ff00ff", "#ffffff")`, `("#ffffff", "#202020")`, `("#777777", "#ffffff")`, which equal `IDENTITY_BACKGROUNDS` and `IDENTITY_FOREGROUNDS` (`native_app.py:39-46`); the fallback states have luminance 0.051, 0.285 and 1.0.

`identity_color` returns `presentation.uncalled_color` for `None`, `presentation.negative_color` when every entry is 0, and otherwise, per RGB channel, `min(1.0, Σ marker.color[c])` over positive markers summed in contract order from `0.0`. For the default config, codes 0 to 4 give `(0.25, 0.25, 0.25)`, `(0, 1, 0)`, `(1, 0, 1)`, `(1, 1, 1)`, `(0.6, 0.6, 0.6)`, which equal `native.PALETTE` (`native.py:39-41`). Their `hex_color` values `#404040`, `#00ff00`, `#ff00ff`, `#ffffff`, `#999999` equal today's field marks (`native_app.py:908`).

The field composite needs numpy, so slice H implements it; the rule is fixed here. With `n_k = clip((image_k − lo_k) / (hi_k − lo_k), 0, 1)` in float64:

- reference view: `n_ref` repeated to three channels;
- marker view `k`: `n_k[..., None] * color_k`;
- identity and uncertainty backgrounds: start from `zeros(shape + (3,))`, add `n_k[..., None] * color_k` for each marker in contract order, then `np.minimum(·, 1.0)`.

For magenta BrdU and green EdU this gives `(b, e, b)`, `(b, 0, b)` and `(0, e, 0)` exactly. Each output channel receives one nonzero term, `x * 1.0` and `x * 0.0` are exact, adding `+0.0` is exact, and the clip is inactive. This reproduces `native.py:577-585` and `test_field_composite_is_brdu_magenta_edu_green`.

## 2. Recording, normalization and resumption

### 2.1 Artifacts

"Default" means `DEFAULT_CONTRACT`. Every new artifact is written in the current form. The legacy form is read through `normalize` and is written only by a run that continues with its own loaded legacy record (2.3).

| Artifact | Recognized by | Legacy form reads as | Current form | Notes |
| --- | --- | --- | --- | --- |
| `source.json` | `schema == "dino-zarr-source-v1"` (`zarr_source.py:244`, `:268`) | channel names only | unchanged | Carries no contract. `manifest_digest` and `write_json_atomic` sort keys (`zarr_source.py:55-58`, `:66`), so channel order affects neither `source_sha256` nor the bytes. |
| Prepared source `complete.json` | `PREPARED_SOURCE` | v1: default | v2 with top-level `marker_contract` | Every completing prepare writes v2 and `markers.json` beside `source.json`. |
| Prepared source without completion | `source.json` and no `complete.json` | `markers.json` if present, else nothing declared | n/a | Chunk partials hold geometry only, so a partial prepare completes in the current form (2.3). |
| Chunk partials | `artifact == "dino-zarr-chunk-v1"` (`zarr_prepare.py:93`, `:240`) | n/a | unchanged | Geometry only. |
| `display.json` (source cache) | `artifact == "dino-zarr-display-v1"` (`zarr_barrage.py:99`) | n/a | unchanged | `vmax` keys must equal the source channel set (decision 9). |
| Barrage `run.json`/`complete.json` | `BARRAGE` | v3: default, `crop_panels` PI \| EdU \| BrdU | v4 with `marker_contract` and `crop_panels: [[key, label], ...]` | New barrages write v4 and `markers.json`. A v3 partial run continues in v3 and gains no `markers.json` (2.3). |
| Hand-made barrage (`data-format.md`) | no recognized record | `markers.json` if present, else default | n/a | Crop columns are `combined_image_path`, `{k}_image_path`, `image_path`. |
| Foreign records in a crop directory | `run.json`/`complete.json` outside the dino families, e.g. `solarmap-native-barrage-v1` in `/working/chaichontat/rhu2-barrage-review` | skipped | n/a | Rejected when they carry `marker_contract` (1.6). |
| Corrected display `display.json` (`dino-corrected-display-v1`) | `schema` (`corrected_barrage.py:26-27`) | the barrage directory's contract | unchanged | Its `source` directory's contract must equal it. |
| `labels.csv` | owning directory, `contract_of(path.parent)` | columns `cell` plus markers, by name | same | Extra columns kept (`label_store.py:51-65`). Never rewritten for migration. |
| Change journal `cell-review/changes-*.jsonl` | `schema` of the file's first record | no `schema`: default, positional `[brdu, edu]` | `schema: "dino-review-journal-v1"` and `marker_contract` after `path`, before `time_utc` | Written, never replayed (`review_session.py:135-139`). A journal file keeps the form of its first record (4.12). |
| Backups `cell-review/<dir>-<session>.csv` | n/a | copies of `labels.csv` | unchanged | `contract_of(cell-review)` is default; the copy's owner is its parent barrage. |
| Feature cache `features.joblib` | `FEATURE_CACHE` on `metadata` | v2: default | v3 with `metadata["marker_contract"]` | Entry keys unchanged. A reused v2 cache keeps its v2 metadata when entries are added. |
| Direct-Zarr bundle | `ZARR_MODEL` | v6: default | v7 with top-level `marker_contract` | Never rewritten; its file SHA-256 identifies it downstream (`zarr_infer.py:487`). `bind_spatial_subtraction` of a v6 bundle returns a derived v7 bundle (4.4). |
| Intensity-context bundle | `INTENSITY_MODEL` | v1: default | v2 with top-level `marker_contract` | The bundles that `periodic apply/measure` write to new files (`periodic.py:285-287`, `:384-386`) and the bundle `bind_spatial_subtraction` returns (`spatial_pixels.py:98`) are derived: current version, the input's contract, input untouched (`derive_bundle`, 1.5). |
| Joint bundle | `JOINT_MODEL` | v1: default | v2 with top-level `marker_contract` | `joint_train.py:243`. Its deployed bundles are intensity-context v2. |
| Inference `contract.json`, `complete.json`, partition provenance | `INFERENCE` | v2: default | v3 with top-level `marker_contract` | Partition provenance is `{**run record, ...}` (`zarr_infer.py:411-416`, `:606-611`). |
| Cortex rescore `complete.json`, partition provenance | `CORTEX_RESCORE` | v1: default | v2 with top-level `marker_contract` | Partitions are `{**run record, ...}` (`cortex_score.py:108-114`). |
| Native prediction manifest with no `artifact` | no `artifact` key | default; rejected when it carries `marker_contract` | n/a | Read only by the native adapters (decision 5). |
| Labeled review | `artifact == "dino-labeled-review-v1"` (`review_contracts.py:76`) | default; rejected when it carries `marker_contract` | none | No writer exists; read only by `load_review_predictions` (4.2). |
| Review pointer `dino-review-current-v1` | `artifact` (`review_contracts.py:47`) | contract of the completion it selects | none | Unchanged. |
| Periodic field bundle | `targets` list (`periodic.py:127`) | its own marker order | unchanged | Exception to D8 (D12, decision 1): its ordered `targets` is its recorded contract and the authoritative coefficient order, checked against the active contract wherever it is bound (4.4). |
| `pixel_correction` identity | exact dict (`periodic_pixels.py:63-71`) | n/a | unchanged | Recorded in barrage and bundle records. |
| `spatial_subtraction` identity | exact dict (`spatial_pixels.py:38-44`, compared at `:49`) | target `brdu`, reference `pi` | unchanged | Only this pair exists (4.4). |
| Periodic `correction.json`, measure `run.json`/`complete.json`, parquet marker metadata | exact dicts (`periodic.py:238-245`, `:333-338`) | n/a | unchanged | Exception to D8 (D12): bound to the field bundle by hash and carry no contract of their own. |
| Embedding partitions `*.pt` | tensor keys | keys minus `cell_sha256` | same | Validation becomes generic (4.10). |
| RHU2 `spatial.json` | `SPATIAL_REVIEW` | v1: default | v2 with top-level `marker_contract` | `rhu2_prepare` writes v2 (decision 7); `review_activation` keeps the version it finds. |
| Frozen H5AD, RHU2 study, residual-BrdU artifacts | their own versions | BrdU/EdU only | unchanged | D4. |

### 2.2 Equality and identity checks

Expected records are built with `stamp` and are therefore current. A check marked "normalized" compares a stored record with a record this code built: it applies the owning kind's `normalize` to the stored record before comparing, because the two may be legacy and current forms of the same semantics. `normalize` also raises `MissingContractError` for a current-version record without its field, so a damaged new record fails instead of being treated as a mismatch. A check marked "raw" compares records of one run with each other and is exact, so the versions within a run must agree; the run's own record is first validated through its kind (`normalize` or `contract`), so a raw comparison never accepts a record its kind rejects.

| Site | Today | Rule |
| --- | --- | --- |
| Prepare resume, `zarr_prepare.py:286-304` | `source.json` dict equality, then completion field checks, then return | `source.json` unchanged. Before the completion branch can return, `declared_contract(out_dir)` reads `complete.json` and `markers.json` together, and a non-`None` result must equal `config.contract` (4.5). Then today's field checks. |
| Barrage display cache, `zarr_barrage.py:154-161` | field equality; `vmax` keys `{pi, edu, brdu}` | `vmax` keys equal `set(source.channels)`. |
| Barrage records, `zarr_barrage.py:508`, `:515` | whole-dict equality | normalized: `BARRAGE.normalize(existing) == expected`; same messages. Raw: a skip from a matching `complete.json` also requires an existing `run.json` to equal it. The run continues with the loaded `run.json` (4.6). |
| Feature cache, `zarr_train.py:496` | any mismatch resets the cache | Metadata whose `schema` `FEATURE_CACHE` recognizes is normalized; `FEATURE_CACHE.normalize(metadata) == expected` reuses the cache with its loaded metadata, and a recognized mismatch resets it as today. Unrecognized metadata resets it as today. `test_training_reuses_legacy_feature_cache_without_encoder` pins reuse of the frozen v2 cache. |
| Inference resume, `zarr_infer.py:518`, `:520` | whole-dict equality | normalized: `INFERENCE.normalize(contract.json) == expected` ("Inference output contract mismatch"). `run_record` is the loaded `contract.json`. Raw: `complete.json`, when present, equals `run_record` ("Inference completion contract mismatch"). A `complete.json` lies in the output directory, which today's check already requires to hold `contract.json` (`:515-519`). |
| Inference partitions, `zarr_infer.py:419` | whole-dict equality | raw: `metadata == {**run_record, partition fields}`, where `run_record` is the loaded `contract.json` if it exists, else `expected`. A partition in the other version fails. |
| Call provenance, `call_provenance.py:27`, `:45-48` | every completion key equals partition metadata | raw within one shard: every key of `complete.json` equals the partition's metadata as today, which includes `artifact` and, for a current shard, `marker_contract`. Each partition's metadata must also pass `INFERENCE.contract(metadata, where=str(path))`, which rejects a legacy partition carrying an extra `marker_contract` that the key-subset check alone would accept. Across shards, the `INFERENCE.contract` of every `complete.json` is equal, so v2 and v3 shards of the default contract combine. |
| Rescore completion and partitions, `cortex_score.py:225`, `:244`, `:117` | completion subset; partition equality | `expected = CORTEX_RESCORE.stamp(...)`. The recorded record is `complete.json` if present, else the first existing output partition by name. Normalized: every item of `expected` is in `CORTEX_RESCORE.normalize(recorded)`. `run_record` is `{k: v for k, v in recorded.items() if k in expected}`, which keeps the recorded version and drops the partition and completion fields (`:237`, `:271`), or `expected` when nothing is recorded. Partitions compare raw against `{**run_record, partition fields}`, and `complete.json` is `{**run_record, completion fields}` (`:271-279`). |
| Rescore inputs, `cortex_score.py:136` | shard source and model identity | additionally `INFERENCE.contract(manifest)` equals the contract `validate_feature_contract` returned (4.11). |
| Joint training inputs, `joint_train.py:88`, `:124` | none | `contract_of(tranche)` for every tranche and `INFERENCE.contract(complete)` for every shard equal the dataset's `bundle_contract(embedding bundle)`, before columns are selected; all datasets share one contract (4.11). |
| Bundle schema allow-lists, `zarr_infer.py:71-73`, `:104`, `:449` | `{v6, knn-v1}`, branch on knn | `ZARR_MODEL.recognizes or INTENSITY_MODEL.recognizes`; branch on `INTENSITY_MODEL.recognizes`. |
| `classifier_features.py:223-224` | both schemas equal knn-v1 | both `INTENSITY_MODEL.recognizes` and contracts equal (4.7). |
| `periodic.py:229-231`, `:319` | classifier schema equals knn-v1 | `INTENSITY_MODEL.recognizes`. |
| `dinoapp/src/dinoapp/rhu2.py:44` | `artifact == "rhu2-spatial-review-v1"` | `SPATIAL_REVIEW.recognizes`, and `SPATIAL_REVIEW.contract(manifest) == config.contract` (3.2). |
| `review_activation.py:20` | none | `SPATIAL_REVIEW.contract(manifest) == DEFAULT_CONTRACT`; the in-place update keeps the version (`:49-57`). |
| Native adapters | partial | each contract the adapter reads equals `config.contract` (4.14). Prediction partition metadata (`prod3_2.py:141-147`) is raw against its run record's `artifact` and passes `prediction_contract`. |
| `spatial_pixels.py:49` | spec equals the fixed identity | unchanged; bind and consumers also check the bundle contract (4.4). |
| `sources.py:149-151` | `source_id` includes a digest of `complete.json` | unchanged (pinned by `test_frozen_barrage_opens_with_todays_identity_and_rosters`). |

Prediction partition metadata is read by call provenance and the Prod3-2/Prod2-2 adapter, and both checks are above. Joint training and rescoring read inference partitions only for cells, labels and embeddings (`joint_train.py:106`, `cortex_score.py:93`); their inference record is each shard's `complete.json`, checked through `INFERENCE.contract` (4.11). Other provenance readers, such as the frozen `roc.py:185`, are unchanged.

Old records and bundles are never rewritten.

### 2.3 Resuming and extending existing artifacts

A record that already exists fixes the version of everything its run writes afterwards. Only after that record passes the normalized comparison of 2.2 does new code reuse it as the run record, as loaded (for rescore, its items whose keys `expected` has). The raw comparisons within the run are unchanged. This applies only to resumed artifacts. A derived artifact is a new file and is written in the current version through `derive_bundle` (1.5, 4.4).

| Run or artifact | Version fixed by | Later writes in that version |
| --- | --- | --- |
| Inference output directory | `contract.json` | partition provenance (`zarr_infer.py:606-611`), `complete.json` (`:625`) |
| Cortex rescore output directory | `complete.json`, else the first existing output partition | partitions (`cortex_score.py:108-114`), `complete.json` (`:280`) |
| Barrage directory | `run.json` (a matching `complete.json` skips the run) | `run.json` (`zarr_barrage.py:517`), `complete.json` (`:605`); a v3 barrage gets no `markers.json` |
| Feature cache | its `metadata` | the cache dumped with added entries (`zarr_train.py:325`), keeping the loaded metadata |
| Change journal file | its first record | appended records (4.12) |
| RHU2 `spatial.json` | itself | `review_activation`'s in-place update |

An interrupted v2 inference run therefore holds v2 partitions written by old code and v2 partitions written by new code, then a v2 `complete.json`; a fresh run of the same command in a new directory writes v3 throughout. A run that holds no versioned record yet is new and writes the current version: an empty output directory, a prepared source with only `source.json` and chunk partials, a barrage directory holding only a seeded `labels.csv`, or a `periodic apply`/`measure` output directory, which today's code never continues: `apply` refuses an existing directory (`periodic.py:210-211`), and `measure` skips a completed run whose model hash matches and refuses published outputs without a completion (`:348-362`). A legacy run continues only under the default contract, because its normalized record carries the default contract and the normalized comparison fails for any other.

## 3. How configuration reaches code

Operations receive an immutable `MarkerConfig` or `MarkerContract` argument. No module keeps a mutable marker global. Non-frozen modules drop `TARGETS` (`zarr_train.py:43`, `zarr_infer.py:47`, `intensity_context.py:11`, `classifier_features.py:16`, `periodic.py:22`).

### 3.1 Commands

| Command | `--markers PATH` | Contract from | Presentation from |
| --- | --- | --- | --- |
| `dino-zarr prepare` | optional; defines the new source | `--markers` or the packaged default; `declared_contract(--out-dir)`, when not `None`, must equal it | same; `save_config` beside `source.json` |
| `dino-zarr barrage` | optional; asserts | `contract_of(--source-dir)`; `declared_contract(--out-dir)`, when not `None`, must equal it | `resolve_directory(source_dir, supplied=...)`; `save_config` in `--out-dir` for a current-version run |
| `dino-zarr train` | optional; asserts | `contract_of` of every `--barrage-dir` and of the target source, all equal | none |
| `dino-zarr infer` | optional; asserts | `bundle_contract(model)`, which must equal `contract_of(--source-dir)` | none |
| `dino-zarr periodic fit` (click option) | optional; defines `targets` | `--markers` or the packaged default | n/a |
| `periodic apply`/`measure` | none | the field bundle's `targets` and the classifier's `bundle_contract`, which must agree; the classifier written to `--out-dir` is derived and current (4.4) | n/a |
| `dino clean-labels` | none | `contract_of(--barrage-dir)` under the lock | n/a |
| `dino.cortex_score`, `dino.joint_train`, `dino.review_activation` | none | the bundles and records they consume | n/a |

"Asserts" means the supplied contract must equal the artifact's (`require_same_contract`, `actual_from` naming the artifact, `expected_from="--markers"`), and a mismatch fails before any write. Prepared sources and barrages store `markers.json` only on paths that also write a current-version record, so a skipped completed run and a resumed v3 barrage write none. Training and inference outputs carry the contract only in their records.

### 3.2 dinoapp resolution per review target

dinoapp takes no marker input. Its native review targets are the `review_targets` (1.3) of the file named by `DINO_REVIEW_CONFIG`, read with `load_review_targets` on every script run; without that variable only the crop directory is offered. The **Review target** options are `("Crop directory", *target names)`. With `DINO_REVIEW_SOURCE="Native fields"`, `DINO_NATIVE_DATASET` must name a configured target, otherwise the app shows `Unknown review target: ...`. `app.ADAPTERS` maps each target's `adapter` (`prod2`, `prod3`, `prod2_2`, `prod3_2`) to its module; an unknown adapter is reported with the target name and the known adapters. Each adapter's `FieldSource(target)` takes its paths, cell prefix, raw channels and name from the target and keeps its file layout relative to `target.root`, `target.review_root` and, for `prod3_2` and `prod2_2`, `target.embedding_run`.

| Target | Config | Cross-checks |
| --- | --- | --- |
| Crop directory (`BarrageSource(root)`) | `self.config = resolve_directory(root)` | `spatial.json` is part of `contract_of(root)`. |
| Corrected barrage | inherited from `BarrageSource` | `contract_of(Path(spec["source"]))` equals `config.contract`; `DisplayPixels` checks its bundle internally (4.14). |
| Native review targets | `config = resolve_directory(default_annotation_path.parent)` | `AnnotationFiles` checks every annotation directory; every bundle, prediction manifest and inference record the adapter reads must equal `config.contract` (4.14). |
| RHU2 (crop directory with `spatial.json`) | as crop directory; `RoiSource.config` and `FieldCropSource.config` are the `SpatialSource`'s | `SpatialSource` requires `SPATIAL_REVIEW.recognizes`, `probability_columns == [rhu2_probability_column(k) for k in markers]`, `set(decision_thresholds) == set(markers)`; `RoiSource` requires `set(fused.attrs["key"]) == set(contract.channel_keys)` (`rhu2.py:137-139`). |

Every `labels.csv` write re-checks the owning directory's contract under `labels.csv.lock` (4.12). That check rejects a stale session but does not refresh its presentation, so the reviewer caches and review revisions also key on `marker_revision` (1.6) of the target's annotation directories:

- Crop directory: `app.get_barrage_source(root: str, marker_revision: str)` (`app.py:141-147`) with `marker_revision((root,))`. The review revision is `source.source_id + ":markers:" + marker_revision((root,))`, which covers `spatial.json` because it is one of `CONTRACT_FILES` and replaces today's `":spatial:{mtime_ns}"` suffix (`app.py:761-763`).
- Native targets: `native_app.get_source(source_type, target: ReviewTarget, marker_revision: str)` with `marker_revision(source_type.annotation_directories(target))`. Each adapter class provides the classmethod `annotation_directories(target: ReviewTarget) -> tuple[Path, ...]`, returning the directories under `target.review_root` that an instance for that target annotates, and `__init__` builds `annotation_paths` from it. The target is part of the cache key. The review revision is `source.source_id + ":markers:" + marker_revision(...)`.
- Native sampled fields: `native_app.get_field(source_type, target: ReviewTarget, marker_revision: str, seed, position, z_um, width_um)` is cached separately from `get_source`. The target and `marker_revision` are part of its cache key, and it calls `get_source(source_type, target, marker_revision)`. The app passes the target and revision it passed to `get_source` for the same render.
- RHU2: `rhu2_app.get_source(root: str, marker_revision: str)` (`rhu2_app.py:14-16`, called at `:20` and `app.py:780`) replaces its `spatial.json` mtime argument with `marker_revision((root,))`.

A changed revision makes `activate_target` cancel prefetches and drop galleries, cell queues and `workspace["field_crop_source"]` (`review_session.py:58-75`), and `rhu2_app` builds each `FieldCropSource` from the current `RoiSource` (`rhu2_app.py:100-102`). A captured `FieldCropSource` therefore never outlives the config it was built with. A changed contract additionally replaces the context (4.12).

## 4. Cross-slice API changes

Signatures are exact. "Unchanged" means as at `9042e7b`.

### 4.1 `review_contracts.combined_uncertainty` (H)

```python
def combined_uncertainty(probabilities: Mapping[str, ArrayLike], cutoffs: Mapping[str, float]) -> np.ndarray
```

`set(probabilities) == set(cutoffs)` and nonempty, else `ValueError("Classifier probabilities and thresholds must name the same nonempty markers")`. Each vector becomes `np.asarray(..., dtype=np.float64)`. Today's checks and messages follow (`review_contracts.py:22-36`): all 1-D with one nonempty shape, finite, within [0, 1], and every cutoff finite and strictly within (0, 1). It returns `1.0 / reduce(np.minimum, (np.abs(p[k] - c[k]) for k in probabilities))` under `np.errstate(divide="ignore")`, which is today's formula and gives `inf` at an exact cutoff.

### 4.2 `review_contracts.load_review_predictions` (H)

```python
def load_review_predictions(manifest_path: Path, *, model_sha256: str, cell_prefix: str,
                            contract: MarkerContract) -> pd.DataFrame
```

The version check stays exact and in this reader: `manifest.get("artifact") != "dino-labeled-review-v1"` raises today's `ValueError` (`review_contracts.py:76`), so a missing, `null` or other version fails. A manifest carrying `marker_contract` raises `MarkerConfigError(f"{manifest_path}: dino-labeled-review-v1 records must not carry marker_contract")`, and `require_same_contract(contract, DEFAULT_CONTRACT, expected_from="the review target", actual_from=str(manifest_path))` rejects any other requested contract. Today's remaining checks follow (`:76-101`) with columns `p_k`, `pred_k` and `cutoff_key(k)` per marker. `load_prediction_pointer` is unchanged. `dinoapp.review_predictions` keeps re-exporting it.

### 4.3 Crop rendering (B)

```python
def read_crop(reader: CellCropReader, row: pd.Series, *, size: int, pixel_correction=None) -> CellCrop  # unchanged
def render_crop(crop: CellCrop, *, cell: str, display_max: Mapping[str, float],
                panels: Sequence[tuple[str, str]]) -> Image.Image
```

`read_crop` calls `correct_channels(crop.channels, origin, pixel_correction=pixel_correction)` without a reference (`crop_rendering.py:41`). `render_crop` draws one axis per `(key, title)` in order, `vmax=display_max[key]`, titles `f"cell={cell}\nch={title} mean={mean:.1f}"`. Figure width is `BASE_PANEL_WIDTH * len(panels)` with `BASE_PANEL_WIDTH = 4.0` inches, so `output_width = round(4.0 * len(panels) * scale * FIGURE_DPI)` equals today's `BASE_FIGURE_SIZE[0]` result for three panels (`crop_rendering.py:22`, `:58`). It raises `ValueError` if `panels` is empty or names a channel absent from `crop.channels`. Callers pass `config.panels`: `zarr_barrage`, `native.FieldSource.cell_crop`, `prod3`, `rhu2.RoiSource.cell_crop`, `corrected_barrage`.

### 4.4 Pixel corrections (E)

```python
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

def correct_channels(channels: Mapping[str, np.ndarray], origin_zyx: tuple[int, int, int], *,
                     reference: str | None = None,
                     spatial_subtraction: ChannelSubtraction | None = None,
                     pixel_correction: PixelCorrection | None = None) -> dict[str, np.ndarray]
```

Order is unchanged (`pixel_corrections.py:17-25`). Subtraction first replaces `corrected[spatial_subtraction.target]` with `spatial_subtraction(channels[target], channels[reference], origin)`, using the uncorrected reference crop. `reference` names the crop channel holding the subtraction's reference pixels: `"pi"` for direct-Zarr crops, `"dapi"` in Prod3 (`prod3.py:211`) and Prod3-2. It is required when `spatial_subtraction` is given, else `ValueError("Spatial subtraction requires the crop channel holding its reference pixels")`. Periodic correction then replaces every channel whose name is in `pixel_correction.targets`, iterating `corrected` in insertion order (the pinned call order `edu` then `brdu` in `test_correct_channels_subtracts_brdu_then_corrects_markers`).

`SpatialPixels` (`spatial_pixels.py`) gains read-only properties `target == "brdu"` and `reference == "pi"`; `identity` uses them and is otherwise unchanged. `from_spec` keeps its dict-equality check, so a spec with any other target or reference raises "Unsupported spatial pixel subtraction contract".

```python
def require_spatial_subtraction_contract(contract: MarkerContract, *, artifact: str) -> None
def bind_spatial_subtraction(bundle: Mapping[str, Any], *, model_path: Path,
                             source_manifest: Mapping[str, Any]) -> dict[str, Any]   # signature unchanged
```

`require_spatial_subtraction_contract` requires `contract.reference == "pi"` and `"brdu" in contract.markers`, else `ValueError(f"{artifact}: spatial subtraction supports only BrdU against the pi channel verified as 405 nm DAPI; contract is {contract}")`. `bind_spatial_subtraction` calls it with `bundle_contract(bundle, where="model bundle")` before its existing checks, and returns `derive_bundle(bundle, {"spatial_subtraction": transform.identity}, where="model bundle")` in place of `{**bundle, ...}` (`spatial_pixels.py:98`), so a v6 or knn-v1 input yields a current-version bundle and the input mapping is unchanged. Consumers that load `spatial_subtraction` from a bundle call it with that bundle's contract: `zarr_infer._load_bundle` (D), `zarr_model.iter_model_image_batches` (C), `DisplayPixels` (H). `model_crop_reader` keeps the instance `subtraction = SpatialPixels.from_spec(spatial_subtraction)` that it builds and discards today (`zarr_model.py:150`) and appends `subtraction.reference` instead of the literal `"pi"` (`:151`). `iter_model_image_batches` checks that `spatial_subtraction.reference` is in `reader.channels` (`:266-267`). `read_model_images` passes `reference=spatial_subtraction.reference if spatial_subtraction is not None else None` (`:194`), because its correction branch also runs for periodic-only correction (`:183`). `crop_rendering.read_crop` passes no reference (`crop_rendering.py:41`), and `DisplayPixels.field_images` passes `reference=self.spatial.reference` (`display_pixels.py:71`).

`PeriodicPixels` keeps its fields and `identity`, and gains:

```python
@property
def targets(self) -> tuple[str, ...]: ...    # from the hash-checked field bundle via _load_field

def require_periodic_targets(correction: PeriodicPixels, contract: MarkerContract, *, artifact: str) -> None
    # correction.targets == contract.markers, else ContractMismatchError naming artifact
```

`_load_field` replaces the `["brdu", "edu"]` check (`periodic_pixels.py:24-25`): `targets` must be a nonempty list of unique valid keys and `coefficient.shape[1] == len(targets)`. `_gain_tile` reshapes to `(width, width, len(targets))` (`:40`). `__call__` raises `ValueError(f"Periodic pixel correction has no target {name!r}")` for names outside `targets` and indexes `targets.index(name)` (`:75-79`). A field bundle with `targets == ["edu", "brdu"]` therefore applies its first coefficient column to EdU. Bind sites call `require_periodic_targets`: `zarr_barrage --pixel-model` (B, `zarr_barrage.py:422-433`), `zarr_model.iter_model_image_batches` (C, `:268-275`), `DisplayPixels` (H, `display_pixels.py:37-42`), `periodic measure` (E).

`periodic.py`: `fit_field(frame, periods, folds, *, targets: Sequence[str])` reads `intensity_column(t)` per target and stores `targets=list(targets)` (`periodic.py:63-77`, `:127`); the `fit` command gains `--markers`. The field bundle, `correction.json` and the measure records keep today's form under D12 (decision 1). `apply` and `measure` take targets from the field bundle, replacing `:216-217`, `:250-276`, `:327-331`, `:367`, and when a classifier is given require `list(bundle["targets"]) == list(bundle_contract(classifier).markers)`. The classifier each writes to a new file is derived: `apply` dumps `derive_bundle(classifier_bundle, {"intensity_props": ..., "periodic_correction": record}, where=str(classifier))` (`:284-287`), and `measure` dumps `derive_bundle(bundle, {"intensity_props": ..., "pixel_correction": correction.identity}, where=str(classifier))` (`:384-386`). The classifier file read from `--classifier` is unchanged, and the hashes in `correction.json` and the measure records keep identifying the files they name.

### 4.5 Source and prepare (A)

```python
def build_source_manifest(*, labels_path: Path, channel_paths: Mapping[str, Path], offset_zyx: tuple[int, int, int],
                          required_channels: Sequence[str], dataset: str | None = None,
                          roi: str | None = None) -> dict[str, Any]
```

`required_channels` replaces the literal tuple (`zarr_source.py:190-196`). Missing channels raise today's `Missing required channel sources: [...]`, and extra channels are still accepted (`test_build_manifest_keeps_extra_channels_and_requires_brdu`).

`PrepareConfig` replaces `pi`, `edu`, `brdu` (`zarr_prepare.py:31-33`) with `channels: Mapping[str, Path]` and gains `config: MarkerConfig`. The CLI replaces `--pi/--edu/--brdu` (`:146-148`) with a repeated `--channel KEY=PATH` (split at the first `=`) and optional `--markers PATH`. Keys must be exactly `config.contract.channel_keys`, each once; violations go through `parser.error(f"--channel keys must be exactly {keys}: missing [...], unexpected [...]")`, and malformed values and duplicates are rejected the same way.

`main` order: build the manifest with `required_channels=config.contract.channel_keys`; the existing `source.json` check (`zarr_prepare.py:286-291`); then, before the completion branch can return (`:292-304`), `declared = declared_contract(out_dir)`, which reads `complete.json` and `markers.json` together and fails when they disagree, and a non-`None` `declared` must equal `config.contract` (`require_same_contract`, `expected_from` the `--markers` path or `"packaged default markers.json"`, `actual_from=str(out_dir)`). If `complete.json` exists: today's field checks and `status=skipped_complete` with no writes. Otherwise: write `source.json` if absent; `save_config(config, out_dir)`, which also rejects an existing `markers.json` with a different presentation; measure; write `PREPARED_SOURCE.stamp({"source_sha256", "cells", "cells_sha256"[, "measured_props_sha256"]}, config.contract)`. A partial output without `complete.json` holds no versioned record and completes in the current form (2.3).

### 4.6 Barrage (B)

```python
def parse_display_overrides(values: list[str], *, keys: Sequence[str]) -> dict[str, float]
def feature_columns(markers: Sequence[str]) -> tuple[str, ...]   # (k_mean, k_std) per marker
```

`parse_display_overrides` accepts `KEY=VALUE` with `KEY` in `keys` (the contract's channel keys), message `"Display maximum must be KEY=VALUE for one of {keys}"`, other rejections as today (`zarr_barrage.py:61-77`). `feature_columns` replaces `FEATURE_COLUMNS` (`:34-39`). `BarrageConfig` gains `config: MarkerConfig`; the CLI gains `--markers`.

`main` order:

1. Load the source; `config = resolve_directory(source_dir, supplied=...)`; every `config.contract.channel_keys` entry must be a source channel.
2. `declared = declared_contract(out_dir)`. When it is not `None`, `require_same_contract(config.contract, declared, expected_from=str(source_dir), actual_from=str(out_dir))` before any write. A missing or empty destination, or one holding only a seeded `labels.csv`, declares nothing and takes the source's contract.
3. Pixel model: `require_periodic_targets(pixel_correction, contract, artifact=str(cfg.pixel_model))`.
4. Display maxima as today (all source channels), overrides keyed by contract channel keys.
5. `expected = BARRAGE.stamp({**base, "crop_panels": [list(p) for p in config.panels]}, contract)`, where `base` is today's fields without `artifact` (`:487-505`).
6. An existing `complete.json` with `BARRAGE.normalize(complete) == expected`, and an existing `run.json` equal to it raw (same version and fields), prints `status=skipped_complete` and writes nothing; any other existing `complete.json` raises today's error (`:506-511`). A completed directory without `run.json` skips as today. An existing `run.json` must satisfy `BARRAGE.normalize(run) == expected`, else today's error (`:513-515`). `run_record` is the loaded `run.json` when it exists, else `expected`.
7. If `run_record` is current, `save_config(config, out_dir)`. Then today's writes, with `run_record` as `run.json` (`:517`) and `complete.json` (`:605`). An existing `labels.csv` must have `cell` and every marker column (`:519-525`).

A legacy partial barrage resumed with panels other than PI | EdU | BrdU therefore fails at step 6, because its normalized record carries the legacy panels. Candidate and index statistics loop over `contract.markers` (`:284`, `:594-596`); `--reuse-index` drops `combined_image_path` and every `image_path_column(k)` (`:601`); rendering passes `panels=config.panels`.

### 4.7 Classifier features and intensity context (C)

```python
# intensity_context
@classmethod
def IntensityContext.load(cls, path: Path, *, scale_zyx_um: np.ndarray, markers: Sequence[str]) -> IntensityContext
# fields and constructor unchanged; features_for_labels iterates self.means in insertion order

# classifier_features
def load_context(features: tuple[str, ...], marker_props: Path, dapi_props: Path, *,
                 scale_zyx_um: np.ndarray, markers: Sequence[str]) -> DapiIntensityContext | DapiRatioIntensityContext
def load_classifier_context(bundle: dict[str, object]) -> Context      # markers from bundle_contract(bundle)
def assemble_classifier_features(embeddings: Mapping[str, np.ndarray], context_features: Mapping[str, np.ndarray],
                                 *, target: str, mode: str, markers: Sequence[str]) -> np.ndarray
def validate_feature_contract(bundle: dict[str, object], embedding_bundle: dict[str, object]) -> MarkerContract
```

`IntensityContext.load` reads `["label", "centroid_z", "centroid_y", "centroid_x", *(intensity_column(k) for k in markers)]` and fills each mean by column name, replacing the positional mapping (`intensity_context.py:50`). The constructor stays because frozen RHU2 code builds contexts directly (`rhu2_train.py:190`, `rhu2_intensity.py:34`).

`assemble_classifier_features` requires `target in markers` (`ValueError(f"Unknown classifier target: {target}")`); `"marker"` mode is `(embeddings[target], context_features[target])`; `"joint"` mode is all embeddings in `markers` order then all contexts in `markers` order (`classifier_features.py:188-191`). Callers pass the order `validate_feature_contract` returns, never their own config. That function returns a `MarkerContract`, so callers pass `validate_feature_contract(bundle, embedding_bundle).markers`, a `tuple[str, ...]`, to `assemble_classifier_features`, `score_rows` and `require_embedding_markers` (`cortex_score.py:183`), and the Prod3-2 adapter compares the returned contract with its config (`prod3_2.py:71`).

`validate_feature_contract` requires both schemas to be recognized by `INTENSITY_MODEL`, then `require_same_contract(bundle_contract(bundle), bundle_contract(embedding_bundle), expected_from="classifier bundle", actual_from="persisted embeddings")`. Today's checks follow (`:222-247`) with expected width `EMBEDDING_WIDTH + C` (marker) or `len(markers) * (EMBEDDING_WIDTH + C)` (joint), `actual` iterating `scaler_key(k)`, and the message still starting "Classifier feature dimensions differ". It returns the bundle's contract. The DAPI contexts iterate `self.marker.means` instead of `TARGETS` (`:97-106`, `:127`). The DAPI scalar and ratio features keep reading `intensity_dapi_mean` from `dapi_props`, independent of the configured reference.

### 4.8 Training split and joint stratification (C)

```python
def joint_label(rows: pd.DataFrame, markers: Sequence[str]) -> pd.Series     # "_".join(str(v)), e.g. "1_0"
def split_training_rows(rows: pd.DataFrame, *, markers: Sequence[str], limit: int, test_fraction: float,
                        seed: int) -> tuple[pd.DataFrame, pd.DataFrame]
def split_dataset_rows(rows: pd.DataFrame, *, markers: Sequence[str], dataset_index: np.ndarray, datasets: int,
                       test_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]
```

`split_training_rows` keeps `training_split.py:11-54` with string strata. Candidates are `joint_label(rows, markers)`, then `rows[k].astype(str)` for each marker in contract order (omitted when N == 1). Today's feasibility rule (`:20`) and class-support check (`:45-49`) iterate `markers`: every marker needs at least two training rows of each class and both classes in validation. The error is `"No supported joint/per-marker stratification satisfies the requested limit and test fraction: need both classes of every marker in validation and at least 2 rows per marker class in training."` For the default contract the candidates and their order equal today's, so splits are unchanged (`test_training_split_stratifies_joint_then_brdu_then_edu`).

`split_dataset_rows` generalizes `joint_train.py:236-238`. Its first candidate is `dataset_index * 2**N + Σ rows[k_i] * 2**i`, with marker `i` in contract order at bit `i`, which for the default is today's `dataset_index * 4 + brdu + 2 * edu`. The next candidates are `dataset_index * 2 + rows[k]` for each marker in order. A candidate is used when every stratum has at least two rows, the stratum count is at most `min(test_size, len(rows) - test_size)`, and the resulting split gives every marker both classes in validation, at least two of each class in training, and at least two of each class within each dataset (the per-dataset `StratifiedKFold`, `joint_train.py:262-266`). Preconditions are checked explicitly; there is no `try/except` around `train_test_split`. It returns sorted index arrays, or raises `ValueError` naming the markers and datasets.

### 4.9 Training CLI, bundle and model input (C, D7)

`--brdu-input-channel/--edu-input-channel` (`zarr_train.py:372-385`) are replaced by a repeated `--input-channel TARGET=CH` plus optional `--markers`. Parsing happens after the contract is known:

```python
def resolve_input_channels(values: Sequence[str], contract: MarkerContract, *, slots: int) -> dict[str, tuple[str, ...]]
```

`TARGET` must be a contract marker. Per target, channels keep flag order, are unique valid keys, and number at most `slots` (`MODEL_INPUT_CHANNELS[hub_model]`). Under the default contract a target with no values gets `DEFAULT_INPUT_CHANNEL_NAMES` (`zarr_model.py:32`), as today (`zarr_train.py:405-408`). Under any other contract every marker must be given, else an error listing the missing targets. The result is ordered by contract markers. `thesis-internal/scripts/dino_train_405.sh:37-40` becomes `--input-channel brdu=brdu --input-channel brdu=pi --input-channel edu=edu --input-channel edu=pi`.

`TrainConfig` replaces the two name tuples with `input_channel_names_by_target: Mapping[str, tuple[str, ...]]` and gains `contract: MarkerContract`. `_load_training_rows(barrage_dir: Path, *, contract: MarkerContract)` and `_load_training_rows_many(barrage_dirs, *, contract)` require `cell` plus every marker in `labels.csv` and cast markers to int64 (`zarr_train.py:110-116`). The contract is the common `contract_of` of every barrage and the target source, checked before labels are merged or features extracted. Cache metadata is `FEATURE_CACHE.stamp(today's fields, contract)`; an existing cache is reused or reset per 2.2 and keeps its loaded metadata when reused (2.3). The bundle is `ZARR_MODEL.stamp(today's fields, contract)` with `scaler_key`, `model_key`, `metric_key` and `cutoff_key` per marker, scalers before models as today (`:587-590`).

`model_crop_reader` keeps its signature. `zarr_model` changes are listed in 4.4.

### 4.10 Embedding artifacts (D)

`write_embedding_atomic(artifact, path)` and `read_embedding_artifact(path)` keep their signatures (out-of-repo caller `thesis-internal/scripts/dino_reduce_umap.py:87`). `_validate_embedding_artifact` (`artifacts.py:170-186`) becomes structural: `cell_sha256` is a 32-byte `uint8` tensor, and at least one other key is present; every other key is a valid marker key whose tensor is CPU, `torch.bfloat16`, 2-D, and the same shape as the others.

```python
def require_embedding_markers(artifact: Mapping[str, torch.Tensor], markers: Sequence[str], *, path: Path) -> None
    # set(artifact) - {"cell_sha256"} == set(markers), else ValueError naming path and both key sets
```

Callers that know the contract call it: `zarr_infer._validate_partition` (`:424-429`), `joint_train` partition reads, `cortex_score._score_partition` (`:94-99`).

### 4.11 Inference, provenance, rescoring, clean labels (D, C)

- `zarr_infer._load_bundle(path) -> dict[str, Any]` keeps its signature and returns the bundle as loaded, not normalized: `model_schema` records the bundle's own `schema`, and `test_legacy_bundle_passes_inference_validation` pins `"dino-zarr-model-v6"`. Required keys come from `bundle_contract(bundle)`; spatial and periodic checks per 4.4. `main` requires `contract_of(source_dir)` and, if given, `--markers` to equal the bundle contract. Record fields: `cutoff_key(k)` per marker and `"embedding_targets": list(markers)`; `expected = INFERENCE.stamp(fields, contract)`. Resumption follows 2.2–2.3: `contract.json` is compared normalized and `complete.json` raw against the run record, which is written as `contract.json` when absent (`zarr_infer.py:551`), spread into partition provenance (`:411-416`, `:606-611`) and written as `complete.json` (`:625`). `_predict` writes `cell, dataset, roi, label`, every `p_k`, then every `k` (`:337-342`). Embedding dimension checks iterate markers, and the embedding partition stays `{**embeddings, "cell_sha256": ...}` (`:614-620`), which `RESERVED_KEYS` keeps collision-free.
- `call_provenance.load_calls(inference, expected_cells, source_hash) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]`. The dict holds call arrays keyed by marker in contract order. The contract is `INFERENCE.contract` of each shard's `complete.json`, all equal; shards may differ in version. Within a shard, each partition keeps today's raw check against every `complete.json` key (`call_provenance.py:47`) and also passes `INFERENCE.contract(metadata, where=str(path))` (2.2). It reads `cell` plus `k` and `p_k` per marker and checks calls against `cutoff_key(k)` (`call_provenance.py:44`, `:53-64`).
- `cortex_score.score_rows(rows, *, embeddings, bundle, context, markers: Sequence[str]) -> pd.DataFrame` writes `cell, dataset, roi, label`, then `p_k, k` interleaved per marker. `score_cortex` sets `markers = validate_feature_contract(bundle, embedding_bundle).markers` (`cortex_score.py:183`); `_partitions` requires `INFERENCE.contract(manifest)` of every shard to equal that contract (`:136`); `_init_worker` receives the run record and `markers` (`:83-88`); `_score_partition` calls `require_embedding_markers` and returns `(rows, {k: positives})`. It computes `classifier_embedding_dimension` as `EMBEDDING_WIDTH * len(markers)` in joint mode and `classifier_feature_count` from the first marker's scaler, records `cutoff_key(k)` and `positive_count_key(k)`, keeps `model_schema` and `classifier_feature_schema` equal to the loaded bundle's own `schema` (pinned as `dino-knn-intensity-v1` in `test_cortex_rescore_complete_and_partition_contracts`), stamps with `CORTEX_RESCORE`, and resumes per 2.2–2.3.
- `clean_labels.main`: under the existing lock, `contract = contract_of(barrage_dir)`; keeps `cell` plus `contract.markers` cast to int and drops other columns as today (`clean_labels.py:27-34`, pinned by `test_clean_labels_writes_canonical_columns`).
- `joint_train`: per dataset, `fit` loads the embedding bundle before the labels (today `read_labels` runs first, `joint_train.py:203-204`) and takes `contract = bundle_contract(bundle)`. `read_labels(dataset, *, contract)` requires `contract_of(tranche) == contract` for each tranche before reading `usecols=["cell", *markers]` (`:88`), and `labeled_embeddings(..., contract=...)` requires `INFERENCE.contract(complete)` of each shard to equal it (`:124`). All datasets share one contract. Splits come from `split_dataset_rows`, and the joint bundle is stamped with `JOINT_MODEL`. A deployed bundle is built from the embedding bundle without `schema` or `marker_contract`, then stamped with `INTENSITY_MODEL` (`:295-325`). Labels CSVs and metrics iterate markers.

### 4.12 dinoapp storage and session (F)

```python
# label_store.py
Label = dino_markers.Label

class LabelConflictError(ValueError): ...    # unchanged

def load_labels(path: Path, *, contract: MarkerContract | None = None) -> pd.DataFrame
def update_label(path: Path, cell: str, value: Label | None, *, contract: MarkerContract,
                 expected: Label | None | Literal["any"] = "any") -> Label | None
```

`load_labels` with `contract=None` uses `contract_of(path.parent)`; WF0 test bodies call `load_labels(path)`. A missing file gives an empty frame with columns `["cell", *markers]`. Today's validation (`label_store.py:17-28`) runs over the contract's markers, with the message "must contain binary marker values".

`update_label` validates `value` with `validate_label` before locking. Under `labels.csv.lock` (`:49-50`) it first runs `require_same_contract(contract, contract_of(path.parent), expected_from="this review session", actual_from=str(path.parent))`, so a stale session gets `ContractMismatchError` and the file is untouched. Then the read-merge-replace of `:51-65` runs over `list(contract.markers)`. A new file's header is `cell,*markers`, and extra columns are kept.

```python
# sources.py
@dataclass(frozen=True)
class Edit:                       # unchanged fields
    path: Path
    cell: str
    previous: Label | None
    value: Label

class AnnotationFiles:
    def __init__(self, paths: tuple[Path, ...], default: Path, *, contract: MarkerContract): ...
    contract: MarkerContract
    values: dict[str, Label]
    owners: dict[str, Path]
    def destination(self, cell: str) -> Path: ...                                    # unchanged
    def assign(self, cell: str, value: Label, *, expected: Label | None) -> Edit: ...   # passes contract
    def undo(self, edit: Edit) -> None: ...                                           # passes contract

class ReviewSource(Protocol):
    target_id: str
    source_id: str
    display_name: str
    config: MarkerConfig
    annotation_paths: tuple[Path, ...]
    default_annotation_path: Path
    subsets: tuple[str, ...]
    datasets: tuple[str, ...]
    crop_markers: tuple[str, ...]      # ordered subset of ("combined", *config.display_markers)
    def review_rows(self, *, dataset: str) -> pd.DataFrame: ...
    def correction_rows(self, live_labels: Mapping[str, Label], *, identity: SavedFilter,
                        subset: str, dataset: str) -> pd.DataFrame: ...
    def crop(self, cell: str, marker: str) -> Crop: ...   # "combined" or a contract marker, else ValueError

class NativeFieldSource(Protocol):     # implemented by slice H adapters
    config: MarkerConfig
    display_name: str
    annotation_paths: tuple[Path, ...]
    default_annotation_path: Path
    gallery_predictions: pd.DataFrame
    origin: np.ndarray
    scale: np.ndarray
    def cell_id(self, label: int) -> str: ...
    def cell_crop(self, label: int) -> tuple[bytes, int]: ...
    def marker_crop(self, label: int, marker: str) -> tuple[bytes, int]: ...
    def field_at_cell(self, label: int, width_um: float) -> Any: ...
```

`AnnotationFiles.__init__` requires `contract_of(path.parent) == contract` for each path before loading. `correction_rows` keeps the keyword `identity`, now a `SavedFilter`; a full tuple keeps today's exact-match meaning (`identity=(0, 1)` in `test_frozen_barrage_opens_with_todays_identity_and_rosters`). `BarrageSource(root)` keeps its one-argument constructor, sets `config = resolve_directory(root)`, and applies today's per-view rule (`sources.py:136-139`) over `("combined", *config.display_markers)`, which gives `("combined", "edu", "brdu")` for the default. `NativeSource.crop` dispatches `"combined"` to `field_source.cell_crop` and marker keys to `field_source.marker_crop`, replacing `edu_cell_crop`/`brdu_cell_crop` (`sources.py:276-284`); `NativeSource.config` is `field_source.config`. `toggle_marker` and `state_label` are removed from `sources.py` in favor of `dino_markers` (1.8).

```python
# review_session.py
def get_context(source_id: str) -> dict[str, Any]          # unchanged; new keys "contract": None, "drafts": {}
def activate_target(target_id: str, *, revision: str, contract: MarkerContract) -> dict[str, Any]
def journal(state_dir: Path, session_id: str, record: dict[str, Any], *, contract: MarkerContract) -> None
def save_assignment(annotations, state_dir, context, cell, value, *, expected, record) -> Edit   # unchanged
def undo_edit(annotations, context, state_dir, edit) -> None                                  # unchanged
```

When the stored context's `contract` is set and differs, `activate_target` cancels its pending prefetches as a revision change does (`review_session.py:58-75`), discards the context, and creates a new one: new `session_id`, empty history, galleries, queues, drafts, workspace and native state. Positional edits, pending events and drafts cannot cross contracts. Revision changes behave as today, and the context records `contract`.

`review_session.JOURNAL_SCHEMA = "dino-review-journal-v1"`. Journal records in a new file are `{**record, "cell", "previous", "value", "path", "schema", "marker_contract", "time_utc"}`, with `previous`/`value` as lists in contract order or `null`, `schema` equal to `JOURNAL_SCHEMA` and `marker_contract` equal to `contract.to_json()`; `record` must not contain `schema` or `marker_contract` (`ValueError`). A journal file keeps the form of its first record (2.3). When `changes-<session>.jsonl` already exists, `journal` reads its first line and classifies it:

- `schema == JOURNAL_SCHEMA`: the record must carry `marker_contract` (else `MissingContractError` naming the file), and `MarkerContract.from_json` of it must equal `contract`. Appended records take the new form.
- no `schema` key: a legacy journal. The record must not carry `marker_contract` (else `MarkerConfigError` naming the file), `contract` must equal `DEFAULT_CONTRACT`, and appended records omit both `schema` and `marker_contract`, as today.
- any other `schema` value: `UnsupportedArtifactError` naming the file.

A contract mismatch raises `ContractMismatchError` naming the file. `save_assignment` and `undo_edit` pass `annotations.contract` and run this check before they change `labels.csv` (today they journal after the write, `review_session.py:155-167`), so a mismatch writes nothing.

### 4.13 Reviewer UI state model, D2 (G)

Cell review (`app.review_barrage_cell`, also used for captured queues):

- Sidebar: "Dataset" as today, and "Cells" with options `("Unsaved", "Saved", "All cells")`, default "Unsaved". When "Saved", one select per marker in `config.display_markers` order, labeled `config.label(k)`, with options `("Any", f"{label}+", f"{label}−")`, default "Any", together forming a `SavedFilter`. Widget names `cell_saved:{k}`. Queue signature `(source.source_id, contract, dataset, cells_option, saved_filter)`.
- The panel shows the saved state as `state_label(config, saved)` or "No saved call".
- With exactly two markers, four buttons (`st.button`) follow in the order `(0, 0), (0, 1), (1, 0), (1, 1)`, labeled `state_label(config, value)`, with key `assign-{session_id}-{cell}-{value}:{current}:{view_token}`; the button for the saved value is `type="primary"`. A click calls `save_assignment(..., value=<tuple in contract order>, expected=<saved value>, record={"action": "assign", "source_id": ..., "dataset_filter": ...})`, which writes every marker in one locked replace, and advances to the next cell. Clicking the saved value writes nothing, sets the notice "<cell> is unchanged; nothing was written.", and still advances.
- With three or more markers, one `−`/`+` control per marker in `display_markers` order edits a draft. Drafts live in `context["drafts"][cell] = {"expected": Label | None, "values": dict[str, int | None]}`, seeded from the saved value (all `None` for an unsaved cell). While any value is `None` the panel shows the caption "Set every marker to save this cell." and writes nothing. As soon as every marker is set and the tuple differs from the saved value, the panel calls `save_assignment` with `expected=draft["expected"]` and the same record, and stays on the cell. A draft is discarded after a save, on navigation away from the cell, on context reset, and when the saved value no longer equals `expected` (with a notice).
- `LabelConflictError` and `ContractMismatchError` set the notice and write nothing.
- An unsaved cell never displays or filters as all-negative; all-negative exists only after an explicit save of `(0,) * N`. Unsaved cells never match a marker filter.

Correction galleries: one view per `crop_markers` entry after "combined", labeled `f"{config.label(k)} correction gallery"` (default EdU, then BrdU, as `app.py:807`). The joint "Saved state" select (`app.py:463-467`) becomes per-marker filter selects, widget names `gallery_saved:{k}`, default all "Any". A card toggles only its gallery's marker with `toggle_marker` and saves the full tuple. Gallery signature `(source_id, contract, marker, subset, dataset, saved_filter)`. Training miscalls (`native_app`): "Disagreement" options are `("Either marker" if N == 2 else "Any marker", *labels)`, mapped to `target: str | None`; the joint identity select becomes per-marker filters.

Native field panel (`native_app.review_field`): with exactly two markers, four buttons in the order `(0, 1), (1, 0), (1, 1), (0, 0)` are labeled `identity_name(config, value)` (EdU only, BrdU only, Dual positive, Neither under the default contract), have key `field-assign:{view_token}:{value}`, and are disabled when the selected cell is not editable. A click saves immediately when the value differs from the saved value, with `expected=selection["expected"]`, and writes nothing otherwise. With three or more markers, one `−`/`+` control per marker (key `field-mark:...`) uses the same draft model as cell review: the call is saved as soon as every marker is set and differs from the saved value, and the caption "Set every marker to save this cell." shows while any marker is unset. "Undo last edit" follows the controls. A rejected save (`LabelConflictError`, `ContractMismatchError`) clears the selection and sets the notice. The first click on an unsaved cell with a model call still saves `decode_call(contract, code)` (`:743-769`). Badge text uses `identity_name`, field marks use `hex_color(identity_color(...))`, and the badge's background and text colors come from `badge_colors(config, value)` (1.9), replacing `IDENTITY_BACKGROUNDS` and `IDENTITY_FOREGROUNDS` (`native_app.py:39-46`, used at `:942-943`).

Widget keys include the marker key and view token, for example `f"cell-mark:{session_id}:{cell}:{k}:{view_token}"`. Every render calls `activate_target(..., contract=source.config.contract)`. `GALLERY_IDENTITIES` (`app.py:44`, `native_app.py:258-262`) is removed.

### 4.14 Dataset adapters and the shared prediction loader (H)

```python
# dinoapp.native
@dataclass(frozen=True)
class ModelCalls:
    calls: np.ndarray            # uint8 indexed by segmentation label; uncalled_code where absent
    uncertainty: np.ndarray      # float32 indexed by label; NaN where absent
    uncertainty_log_vmax: float

def load_model_calls(frame: pd.DataFrame, *, dataset: str, contract: MarkerContract,
                     cutoffs: Mapping[str, float], cell_prefix: str,
                     uncertainty_log_vmax: float | None = None) -> ModelCalls
def prediction_contract(manifest: Mapping[str, object], *, where: str) -> MarkerContract
def classifier_caption(model_id: str, config: MarkerConfig, cutoffs: Mapping[str, float]) -> str
def code_colors(config: MarkerConfig, codes: np.ndarray) -> np.ndarray   # (len(codes), 3) float64, per distinct code
PALETTE: np.ndarray        # code_colors(default_config(), np.arange(5, dtype=np.uint8)); kept as a public name
def render_marker_crop(crop: CellCrop, marker: str, display_max: float, *, label: str) -> bytes
def prefetch_marker_crop(source: NativeFieldSource, cell: str, marker: str) -> None
def training_gallery(predictions, live_labels, *, contract: MarkerContract, split: str,
                     identity: SavedFilter) -> pd.DataFrame
def training_miscalls(predictions, live_labels, *, contract: MarkerContract, split: str,
                      target: str | None, identity: SavedFilter) -> pd.DataFrame
def render_field(field: Field, calls: np.ndarray, *, config: MarkerConfig, mode: str, selected: int | None,
                 border_alpha: float, boundary_alpha: float, contrast: Mapping[str, tuple[float, float]],
                 uncertainty: np.ndarray | None = None, uncertainty_log_vmax: float | None = None) -> str
```

`load_model_calls` performs the checks the adapters share. It requires columns `label`, `cell`, and `k`, `p_k` per marker, unique positive labels, `cell == f"{cell_prefix}|{label}"`, non-null binary calls, and finite probabilities in [0, 1]. Error messages start with `dataset`. Calls are encoded with `call_weights`, and uncertainty comes from `combined_uncertainty`. When `uncertainty_log_vmax` is `None`, it requires at least one finite uncertainty ("No finite model uncertainty scores are available", `native.py:376-378`) and sets `uncertainty_log_vmax = np.quantile(np.log1p(finite), 0.999)`. When it is given, that value is used and no finite score is required. RHU2 passes its snapshot's cohort-wide scale (`rhu2.py:162`), because exact-cutoff probabilities can make every score of one ROI infinite and RHU2 accepts such an ROI today (`rhu2.py:149-162`).

Adapter-specific checks stay in each adapter: Prod3-2 provenance and coverage (`prod3_2.py:125-155`, cutoffs iterate markers), and RHU2's threshold-derived calls. RHU2 builds the frame it passes with `p_k = rows[rhu2_probability_column(k)]` and `k = p_k >= decision_thresholds[k]`.

`prediction_contract` reads a manifest with no `artifact` key as `DEFAULT_CONTRACT` (decision 5) unless it carries `marker_contract`, which raises `UnsupportedArtifactError(f"{where}: unversioned prediction manifest carries marker_contract")`. Otherwise `INFERENCE` or `CORTEX_RESCORE`, whichever recognizes it, gives the contract, and any other version raises `UnsupportedArtifactError`.

Every adapter compares each contract it reads with `config.contract`, so a bundle and a manifest are both checked when both exist:

| Adapter | Contracts checked |
| --- | --- |
| Prod2 (`native.py`) | `prediction_contract` of the pointer's manifest (`native.py:346-348`); the labeled review through `load_review_predictions(contract=...)` (`:386-389`) |
| Prod3 (`prod3.py`) | the `DisplayPixels` bundle (internally, below); the pointer's manifest (`prod3.py:141-142`); the labeled review (`:175`) |
| Prod3-2, Prod2-2 (`prod3_2.py`) | the `DisplayPixels` embedding bundle; with a pointer, the manifest and `validate_feature_contract(bundle, embedding_bundle)` of the selected classifier (`prod3_2.py:60-71`); without one, `prediction_contract` of the reembedding `complete.json` and `INFERENCE.contract` of each shard's `complete.json` (`:56-57`, `:128`); every prediction partition's metadata (`:141-147`), whose `artifact` must equal its run record's (the pointer's manifest, or the shard's `complete.json`) and whose `prediction_contract` must equal `config.contract`; the labeled review (`:182`) |
| RHU2 (`rhu2.py`) | `SPATIAL_REVIEW.contract(spatial.json)` and the channel and column checks of 3.2 |

`Field.images` is keyed by contract channel keys, and `Field.anatomy_channel` is removed (the single-channel modes are used only by tests). `render_field` modes are `"Cell identities"`, `"Model uncertainty"`, or a channel key, with the composite of 1.9; identity borders use `code_colors(config, calls)`. Each native target declares its raw-channel mapping as `raw_channels` in its review target (1.3), for example `{"561": "pi", "BrdU": "brdu", "EdU": "edu"}` for Prod2 and Prod3 and `{"DAPI": "pi", "BrdU": "brdu", "EdU": "edu"}` for Prod3-2 and Prod2-2; the Prod3 and Prod3-2 adapters add the crop-only `"dapi"` channel. RHU2 uses the fused `key` attributes (`rhu2.py:137-139`). `FieldSource.contrast` becomes `dict[str, tuple[float, float]]`. `field_calls` encodes saved labels with `encode_call` and fills `uncalled_code`. `FieldSource.config` comes from `resolve_directory(default_annotation_path.parent)` and must serve the target's `raw_channels` (`adapter_config`), and each adapter class provides `annotation_directories(target)` (3.2). `RoiSource.config` is its `SpatialSource`'s config, and `FieldCropSource` exposes `config` from its `RoiSource`.

`toggle_brdu`, `toggle_edu`, `render_edu_crop`, `render_brdu_crop`, `prefetch_edu_crop`, `prefetch_brdu_crop`, `edu_cell_crop`, `brdu_cell_crop` and `IDENTITIES` are removed. `marker_crop` caches with `lru_cache(maxsize=128)`.

`DisplayPixels(model_path, source_dir)` keeps its signature, because `build_corrected_barrage` stubs it. Internally it requires `bundle_contract(bundle) == contract_of(source_dir)`, applies `require_spatial_subtraction_contract` and `require_periodic_targets`, and `field_images` returns `{**images, **{k: corrected[k] for k in markers}}`, passing `reference=self.spatial.reference` (`display_pixels.py:71`). `crop(crop, origin, *, reference)` is unchanged. `CorrectedBarrageSource` calls `read_crop(self.reader, row, size=size)` and `self.display_pixels.crop(crop, origin, reference=self.config.reference.key)` with today's call shapes.

`rhu2_prepare` writes `SPATIAL_REVIEW.stamp(fields, DEFAULT_CONTRACT)`, that is `rhu2-spatial-review-v2` with `marker_contract` (`rhu2_prepare.py:96-102`), with names from `default_config()` and uncertainty from the mapping form of `combined_uncertainty`. `rhu2_app` shows one vmax percentile input per marker with widget names `rhu2_{k}_vmax_percentile`, which are today's names for the default.

## 5. Slices

### 5.0 Rules for every slice

- A slice edits only its files and their tests (plan section 5). It reports a needed change elsewhere instead of making it. Section 6 assigns the out-of-repo callers.
- WF0 assertions on frozen fixtures never change. Assertions on freshly written default-config artifacts change only as listed in the table below, each by adding `marker_contract` or the new version string, and the owning slice lists every such edit in its report. Otherwise a slice adapts only the WF0 invocation helpers and builders assigned to it below, so each has one owner even when two slices in a wave touch one file.
- Within a wave, a slice that calls an API changed by a sibling slice (in WF3, B calls E's correction APIs, and the tests of B and E build manifests with A's `build_source_manifest`; C↔D in WF4; F, G, H and I in WF5, where G, H and I call F's storage and session APIs, F's `NativeSource.crop` calls H's `marker_crop`, I's `review_activation.py` calls H's `combined_uncertainty`, F's `test_sources.py` and `test_review_session.py` call G's gallery APIs in `app.py` and `native_app.py`, and H's `test_native.py` calls G's `native_app.new_gallery_state` and `update_gallery_card`) writes the call against this spec and tests its own logic with fakes. Tests that exercise the sibling's implementation are listed in its report as integration-dependent and are run at the wave gate.
- Across waves, a slice changes APIs whose callers, test fakes, fixtures or builders lie in files owned by a slice of another wave, earlier or later (table below). Both full suites run at every gate (plan D6), so the changing slice's report gives, for each affected caller, fake, fixture and builder outside its files, the smallest edit that passes `DEFAULT_CONTRACT`, `DEFAULT_CONTRACT.markers` or `default_config()` (or adds a legacy `schema` to a schema-less test bundle) and otherwise keeps today's behavior. The main session applies these edits at the changing slice's integration. In a later-wave file the later owner replaces the edit with this spec's form; in an earlier-wave file the edit is final, because that owner has finished. The table lists the cases found at `9042e7b`; the changing slice reports any other it finds. These edits are not compatibility aliases: no old name or signature is kept.

| Changed by | API | Callers in other-wave files | Bridging edit |
| --- | --- | --- | --- |
| A | `build_source_manifest(..., required_channels=...)` | `test_zarr_infer.py:46` (D) | `required_channels=DEFAULT_CONTRACT.channel_keys` |
| B | `render_crop(..., panels=...)` | `native.py:453`, `prod3.py:226`, `rhu2.py:240`, `corrected_barrage.py:69` (H) | `panels=default_config().panels` |
| C | `split_training_rows(..., markers=...)` | `train.py:396` (I) | `markers=DEFAULT_CONTRACT.markers` (I's final form) |
| C | `IntensityContext.load(..., markers=...)` | `test_periodic.py:97` (E, WF3) | `markers=DEFAULT_CONTRACT.markers` (final) |
| C | `iter_model_image_batches` reads the bundle contract to bind periodic and spatial corrections (4.4) | the schema-less bundle in `test_periodic_pixels.py:97-103` (E, WF3); the bound bundle in `test_spatial_pixels.py:59-66` (E, WF3) | the bundle gains `schema="dino-zarr-model-v6"` (final); in `test_spatial_pixels.py` the `schema` E adds to `original` for `bind_spatial_subtraction` already satisfies this |
| F | `update_label(..., contract=...)` | `test_clean_labels.py:59` (C, WF4), which submits `update_label` to a thread | `contract=DEFAULT_CONTRACT` (final) |

`test_reused_index_refreshes_reviewer_paths` (`test_zarr_barrage.py:302`, B, WF3) also calls F's API, through `BarrageSource(root)` and `crop(cell, marker)`. Both keep their signatures (decision 14, 4.12), and the barrage it opens records the default contract, so it needs no edit.

- Every slice adds custom-contract tests, using a three-marker contract with reference `dapi` (for example markers `edu, brdu, pcna`) or a two-marker reorder `["edu", "brdu"]`, and a same-length-reorder rejection test wherever it consumes an artifact.

WF0 assertions that change under decisions 1 and 21. `DEFAULT_JSON` below is the literal `{"version": 1, "reference": "pi", "markers": ["brdu", "edu"]}`.

| Test (file) | Assertion today | Assertion after | Owner |
| --- | --- | --- | --- |
| `_assert_reproduces_legacy_bundle`, called by `test_training_reuses_legacy_feature_cache_without_encoder` and `test_fresh_training_reproduces_legacy_cache_and_bundle` (`dino/tests/test_characterization_train_infer.py:384-392`) | `set(bundle) == set(legacy_bundle) == EXPECTED_BUNDLE_KEYS`; every key except `created_at_utc` and `training_barrage_dirs` equal between the fresh and frozen bundle | `set(legacy_bundle) == EXPECTED_BUNDLE_KEYS` and `set(bundle) == EXPECTED_BUNDLE_KEYS \| {"marker_contract"}`; `schema` leaves the equality loop and is asserted as `bundle["schema"] == "dino-zarr-model-v7"`, with `bundle["marker_contract"] == DEFAULT_JSON` | C |
| `test_fresh_training_reproduces_legacy_cache_and_bundle` (same file, `:477`) | `cache["metadata"] == legacy_cache["metadata"] == EXPECTED_CACHE_METADATA` | `legacy_cache["metadata"] == EXPECTED_CACHE_METADATA` and `cache["metadata"] == {**EXPECTED_CACHE_METADATA, "schema": "dino-zarr-features-v3", "marker_contract": DEFAULT_JSON}` | C |
| `test_fresh_inference_on_legacy_bundle_reproduces_legacy_run` (same file, `:609-617`) | the rerun's `contract.json` and `complete.json` equal the frozen run's; `read_parquet_artifact` of each rerun partition equals the frozen partition's | each rerun record equals `{**frozen, "artifact": "dino-zarr-infer-v3", "marker_contract": DEFAULT_JSON}`; each rerun partition's provenance equals the frozen provenance with the same two changes, and its row count is unchanged | D |
| `test_cortex_rescore_complete_and_partition_contracts` (same file, `:817-845`, `:903-967`) | `RESCORE_CONTRACT_KEYS` without `marker_contract`; `"artifact": "dino-cortex-rescore-v1"` in the checked subset | `RESCORE_CONTRACT_KEYS` gains `"marker_contract"`, which extends both the completion key set and the partition provenance check; the checked subset has `"artifact": "dino-cortex-rescore-v2"` and `"marker_contract": DEFAULT_JSON` | D |
| `test_new_session_backs_up_labels_before_its_first_edit_and_journals_undo` (`dinoapp/tests/test_characterization_review.py:625`) | `[list(record) for record in records] == [EDIT_KEYS[record["action"]] for record in records]` | each key list is `[*EDIT_KEYS[action][:-1], "schema", "marker_contract", "time_utc"]`, and every record has `record["schema"] == "dino-review-journal-v1"` and `record["marker_contract"] == DEFAULT_JSON`; `EDIT_KEYS` itself is unchanged because `test_frozen_journal_records_positional_brdu_edu_pairs` reads it | F |

No other WF0 assertion changes. The fresh-barrage tests compare records through `_assert_fresh_records`, which already ignores `artifact` and extra keys (`test_characterization_barrage.py:183-196`), and the `legacy_partial_run` case of `test_rebuild_preserves_legacy_annotations` continues in v3 (2.3). `test_prepare_reproduces_frozen_output_from_frozen_inputs` checks selected completion keys only. `test_interrupted_legacy_inference_resumes_missing_partition`, `test_completed_legacy_inference_resumes_as_complete`, `test_completed_legacy_barrage_resumes_as_skipped` and `test_resumed_session_appends_to_the_frozen_journal_and_keeps_its_backup` resume legacy runs, which keep their version.

WF0 helper and builder ownership:

| File | Helper or builder | Owner |
| --- | --- | --- |
| `dino/tests/test_characterization_source.py` | `run_prepare`, `build_manifest` | A |
| `dino/tests/test_characterization_barrage.py` | `_parse_display_max`, `_feature_columns`, `_render_combined_crop`, `_read_crop` (wraps the plain-function fake in an object exposing `targets=("brdu", "edu")`) | B |
| | `_correct_channels` (wraps fakes exposing `target="brdu"`, `reference="pi"`, `targets=("brdu", "edu")`; passes `reference="pi"`) | E |
| | `_load_training_rows` (WF4; passes `contract=DEFAULT_CONTRACT`), `_load_training_sources` (WF4) | C |
| `dino/tests/test_characterization_train_infer.py` | `_train_cli`, `_intensity_features`, `_split`, `_assemble`, `_validate_feature_contract` | C |
| | `_infer_args`, `_score_rows`, `_load_calls` (returns `calls["brdu"], calls["edu"], records`), and the builder `_write_rescore_inputs`, whose shard `complete.json` gains `"artifact": "dino-zarr-infer-v2"` because rescoring now reads the inference record's contract (4.11) | D |
| | `_combined_uncertainty` | H |
| `dinoapp/tests/test_characterization_review.py` | `_update`, `_annotations`, `_barrage_annotations`, `_save`, `_undo` | F |
| | `_toggle`, `_render_marker_crop`, `_load_native_calls`, `_overlay_calls` (sets `source.config = default_config()`), `_render_field_pixels` (contract-keyed images, mapping contrast, `config=`) | H |

### WF2: `dino_markers`

Provides section 1 and the `pyproject.toml` changes. Tests: the packaged default equals 1.2 and `DEFAULT_CONTRACT`; every rule in 1.4 (8 markers, reserved keys `labels`, `cell_sha256` and `embedding` through `embedding_model`, uppercase key, leading underscore, `brdu`/`p_brdu` collision, duplicate label, non-permutation panels, `True` as a color component, unknown key, badge state with a missing or extra marker, duplicate badge state); config and contract JSON round trips, with and without `badge_colors`, with every color written as a triple; `require_same_contract` rejects `["edu", "brdu"]` against `["brdu", "edu"]` with both orders in the message; each `ArtifactKind` (legacy read, current read, current without the field raising `MissingContractError` naming the file, legacy with the field, legacy barrage carrying `crop_panels`, unknown version, missing version key, `stamp` for default and for `["edu", "brdu"]`); `derive_bundle` turns a knn-v1 bundle into knn-v2 with the default contract and the updates, keeps a v2 bundle's custom contract, leaves its input unchanged, and refuses `schema` or `marker_contract` in the updates; `normalize` of a v3 barrage adds the legacy panels and the default contract; `bundle_contract`; `declared_contract` is `None` and `contract_of` is default for a missing and an empty directory; `contract_of` for a legacy prepared source, a v3 barrage, a hand-made directory with an undecodable `complete.json` and a custom `markers.json`, a `solarmap-native-barrage-v1` `run.json` (skipped), the same record carrying `marker_contract` (raises), a `run.json` that is not a JSON object (raises), a v3 record plus a reordered `markers.json` (mismatch), a v4 record plus a matching `markers.json`, an unknown family version, `spatial.json` v1 and v2; `resolve_directory` with a supplied reordered config (raises) and with a custom-contract v4 record but no `markers.json` (raises); `save_config` no-op and conflict; `marker_revision` changes when `markers.json` is written; names under both contracts; call codes round trip for N = 1 … 7, default codes 0 … 4; `identity_color` default equals `native.py:39-41` exactly; `hex_color` default list; `badge_colors` for codes 0 … 4 of the default equals `native_app.py:39-46`, and a three-marker state without an override uses its identity color and luminance foreground; `state_label` and `identity_name` default strings; `matches_saved`. Reports the editable install of `markers/` into `multi` and `seq` for the main session.

### A: Source and prepare (WF3)

Owns `zarr_source.py`, `zarr_prepare.py`, `test_zarr_source.py`, `test_zarr_prepare.py`. Provides 4.5 and prepared-source records with `markers.json`. Consumes WF2. Tests: default and three-marker prepares write `dino-zarr-source-complete-v2` with `marker_contract` and `markers.json`; `source.json` bytes do not depend on the config; `--channel` errors (missing, unexpected, duplicate, malformed, invalid key); resume of a default output with `["edu", "brdu"]` raises `ContractMismatchError` naming `complete.json` before any write; a completed output whose `markers.json` disagrees with `--markers` fails before the skip returns; a v2 `complete.json` without `marker_contract` raises `MissingContractError`; a legacy completed output skips with its file set and bytes unchanged and no `markers.json` added (`test_prepare_resumes_frozen_legacy_output_without_rewriting` checks only the files present before the run); a partial output with only `source.json` and chunk partials completes as v2.

### B: Barrage and rendering (WF3)

Owns `zarr_barrage.py`, `crop_rendering.py`, `test_zarr_barrage.py`. Provides 4.3 and 4.6 and barrage records with `markers.json`. Consumes WF2, A's records through `contract_of`, and E's `PeriodicPixels.targets`, `require_periodic_targets` and optional `reference` (integration-dependent). Tests: a fresh default barrage writes v4 with `marker_contract`, the default `crop_panels` and `markers.json`; three-marker barrage `index.csv` and `candidates.parquet` columns; a fresh three-marker destination and one holding only a seeded `labels.csv` accept the source's contract; a v3 partial `run.json` resumes in v3 and writes no `markers.json`; a completed directory whose `run.json` differs in version from a matching `complete.json` is not skipped; resuming a v3 partial run with custom `crop_panels` is rejected; four-panel render is 512 × 128 at size 16 with titles in `crop_panels` order; `--display-max` accepts `dapi`/`pcna` and rejects `pi` under the three-marker contract; `--markers ["edu", "brdu"]` against a default source rejected; a reordered `markers.json` in the source against an existing default barrage rejected; resume with changed `crop_panels` rejected; an existing `labels.csv` lacking a marker column rejected; `--pixel-model` with mismatched periodic targets rejected; display cache with an extra source channel round-trips.

### E: Pixel corrections (WF3)

Owns `pixel_corrections.py`, `periodic_pixels.py`, `periodic.py`, `spatial_pixels.py`, `test_periodic_pixels.py`, `test_periodic.py`, `test_spatial_pixels.py`. Provides 4.4. Tests: `correct_channels` with three markers and a subtraction object; subtraction without `reference` rejected; periodic-only correction with no `reference`; Prod3-style crops (`pi` and `dapi` channels) subtract against `dapi`; `SpatialPixels.from_spec` with target `edu` or reference `dapi` rejected; `bind_spatial_subtraction` rejects a bundle whose contract lacks `brdu` or has reference `dapi` (existing tests gain a `schema` in their bundles), and returns a current-version bundle with the input's contract while leaving the input mapping unchanged; a field bundle with `targets ["edu", "brdu"]` applies coefficient column 0 to EdU; three-target reshape; a name outside `targets` rejected; coefficient width mismatch rejected; `periodic fit --markers` writes three targets; `apply`/`measure` reject a classifier whose contract markers are reordered relative to the field targets, and from a v1 classifier write a `dino-knn-intensity-v2` classifier with the default contract while the input file stays byte-identical; the field bundle and `correction.json` keep today's form.

### C: Training (WF4)

Owns `zarr_train.py`, `zarr_model.py`, `classifier_features.py`, `training_split.py`, `intensity_context.py`, `joint_train.py`, `clean_labels.py`, `test_zarr_train.py`, `test_zarr_model.py`, `test_classifier_features.py`, `test_intensity_context.py`, `test_joint_train.py`, `test_clean_labels.py`, and the out-of-repo `thesis-internal/scripts/dino_neighbor_eval.py` and `thesis-internal/scripts/dino_train_405.sh` (section 6). Provides 4.7–4.9 and the `clean_labels`/`joint_train` parts of 4.11. Before changing `joint_train.py`, it freezes small `dino-knn-intensity-v1` and `dino-joint-dataset-v1` bundles as fixtures (plan section 5) and adds read tests for them. Consumes A and B records, E's correction APIs, D's `require_embedding_markers` (integration-dependent). Tests: three-marker training on stub encoders with `--input-channel` for each marker writes a v7 bundle and v3 cache metadata; a reused v2 cache extended with new entries keeps its v2 metadata; v3 cache metadata without `marker_contract` raises `MissingContractError`; a non-default contract with a marker missing from `--input-channel` rejected; two barrages with reordered contracts rejected before feature extraction; a target source with a different contract rejected; `read_model_images` with periodic-only correction passes no reference; joint assembly with markers `["edu", "brdu"]` reverses the column blocks; `validate_feature_contract` returns the contract and rejects a classifier whose contract order differs from its embedding bundle; joint width for three markers; three-marker `split_training_rows` fallback order; `split_dataset_rows` reproduces today's `joint_train` split on a default fixture and falls back per marker when the joint strata are infeasible; `IntensityContext.load` maps by name with reordered parquet columns; `clean_labels` keeps three marker columns; `joint_train` stamps v2 bundles for a three-marker fit, rejects a tranche whose `markers.json` reorders the markers before reading its columns, and rejects a shard inference record whose contract differs from its embedding bundle. Reports the 5.0 edits to E's `test_periodic.py:97` and `test_periodic_pixels.py:97-103`.

### D: Inference and provenance (WF4)

Owns `zarr_infer.py`, `call_provenance.py`, `cortex_score.py`, `artifacts.py`, `test_zarr_infer.py`, `test_cortex_score.py`, `test_artifacts.py`, and the out-of-repo `thesis-internal/scripts/dino_render_calls.py`, `thesis-internal/test/test_dino_call_provenance.py`, `thesis-internal/thesis/gam/gam/uv_soft_joint.py` and `thesis-internal/thesis/scripts/prepare_uv_gam.py` (section 6, D10, D11). Provides 4.10 and the inference, provenance and rescore parts of 4.11. Consumes E's `require_spatial_subtraction_contract` (integrated in WF3) and C's `validate_feature_contract`, `assemble_classifier_features`, `IntensityContext.load` and `zarr_model` (integration-dependent). Tests: three-marker inference writes v3 records in `contract.json`, `complete.json` and partition metadata with grouped columns; a v3 `contract.json` with a v2 partition is rejected (an interrupted v2 run resuming with v2 partitions and a v2 completion is `test_interrupted_legacy_inference_resumes_missing_partition`); a v3 `contract.json` with a v2 `complete.json` of the same default semantics is rejected; `--markers` reorder and bundle/source contract mismatch rejected; a v7 bundle without `marker_contract` fails naming the bundle; `load_calls` returns three arrays in contract order, combines v2 and v3 shards of the default contract, rejects shards with different contracts, and rejects a v2 partition carrying an extra `marker_contract`; rescore with three markers interleaves columns and writes `pcna_positive` in a v2 completion; a rescore resumed from v1 partitions continues in v1, and one resumed from a v1 `complete.json` with a partition removed rewrites that partition with v1 provenance that carries no completion fields; rescore rejects a shard inference record whose contract differs from the embedding bundle; structural embedding validation accepts three keys and rejects unequal shapes and invalid keys; `require_embedding_markers` rejects a missing marker.

### F: Annotation storage (WF5)

Owns `dinoapp/src/dinoapp/label_store.py`, `sources.py`, `review_session.py`, `dinoapp/docs/data-format.md`, `test_label_store.py`, `test_sources.py`, `test_review_session.py`. Provides 4.12. Consumes WF2, and G's gallery APIs and H's `marker_crop` (integration-dependent). Tests: three-marker `labels.csv` create and edit with extra columns kept; value length and binary checks; stale session (directory `markers.json` replaced by a reordered one after `AnnotationFiles` loads) raises `ContractMismatchError` from `update_label` and leaves the file byte-identical; `AnnotationFiles` rejects a directory whose contract differs; a new journal file under a custom contract carries `schema` and `marker_contract` after `path` and before `time_utc` on every record (the default case is `test_new_session_backs_up_labels_before_its_first_edit_and_journals_undo` with the assertion update above, and appending to a legacy journal without either field is `test_resumed_session_appends_to_the_frozen_journal_and_keeps_its_backup`); appending to a legacy journal under a custom contract raises; a first record with `schema` but no `marker_contract` raises `MissingContractError`, one with `marker_contract` but no `schema` raises `MarkerConfigError`, and an unknown `schema` raises `UnsupportedArtifactError`, each before `labels.csv` changes; `activate_target` with a new contract gives a new session id and empty history and drafts; `BarrageSource` for a hand-made three-marker directory follows `display_markers` and `{k}_image_path`; `correction_rows` with `None` entries; `NativeSource.crop` dispatches to `marker_crop`. Basic barrage review without `dino` is G's `test_plain_crop_workspace_runs_without_native_or_compute_dependencies`. Updates `data-format.md` for `markers.json`, contract-derived `labels.csv` columns, `{k}_image_path`, the rule that a hand-made `run.json` or `complete.json` must not carry `marker_contract`, and the journal field. Reports the 5.0 edit to C's `test_clean_labels.py:59`.

### G: Reviewer UI (WF5)

Owns `dinoapp/src/dinoapp/app.py`, `native_app.py`, `test_workspace.py`, `test_native_selection.py`, `test_native_dependencies.py`, `test_native_cache.py`. Provides 4.13 and the app and `native_app` cache keys and revisions of 3.2. Consumes F, H (integration-dependent) and `dino_markers`. Tests (Streamlit `AppTest`) with a three-marker hand-made barrage: per-marker controls with the caption "Set every marker to save this cell." while any marker is unset; setting the last marker writes `(1, 0, 1)` in one row without advancing; "Unsaved" excludes a saved all-negative cell; per-marker filters in cell review and galleries; a gallery card toggles only its marker; a stale-session write shows the mismatch notice and leaves `labels.csv` unchanged; replacing `markers.json` with one that changes only a label rebuilds the barrage source and shows the new label; for a native dataset, replacing an annotation directory's `markers.json` and revisiting the same sampled field (same seed, position, z and width) rebuilds it through `get_field` with the new config; badge background and text follow `badge_colors`; crop-directory review without `dino` is `test_plain_crop_workspace_runs_without_native_or_compute_dependencies`. With two markers, a cell review button saves and advances and clicking the saved value writes nothing and advances; a field button saves in one click and clicking the saved value writes nothing.

### H: Dataset adapters (WF5)

Owns `native.py`, `prod3.py`, `prod3_2.py`, `prod2_2.py`, `rhu2.py`, `rhu2_app.py`, `rhu2_prepare.py`, `corrected_barrage.py`, `display_pixels.py`, `dino/dino/review_contracts.py`, `test_native.py`, `test_prod3_2.py`, `test_prod3_2_review.py`, `test_prod3_gallery.py`, `test_rhu2.py`, `test_display_pixels.py`, `dino/tests/test_review_contracts.py`. Provides 4.1, 4.2, 4.14, the `NativeFieldSource` implementations, adapter `config`, `annotation_directories`, `marker_crop` and `contrast`, and the `rhu2_app` cache key of 3.2. Consumes F (integration-dependent), B (`render_crop`), E (`correct_channels`, `require_periodic_targets`, `require_spatial_subtraction_contract`), C (`validate_feature_contract`, `prod3_2.py:71`). Tests: `combined_uncertainty` for three markers, exact-cutoff `inf` and key-set mismatch; `load_review_predictions` rejects a manifest with no `artifact`, with `artifact: null` and with an unknown version, one carrying `marker_contract`, and a non-default requested contract; `load_model_calls` with three markers and each error; `load_model_calls` with a given `uncertainty_log_vmax` accepts a frame whose scores are all infinite; an RHU2 ROI whose probabilities all sit at the thresholds loads with the snapshot's scale; default `code_colors` equals `PALETTE`; three-marker composite with clipping equals the 1.9 formula; `render_field` reference and marker modes; `marker_crop` per marker; an adapter rejects an annotation directory, a prediction manifest, a classifier bundle or a shard inference record whose contract differs from its config; `prediction_contract` rejects an unversioned manifest that carries `marker_contract`; Prod3-2 rejects a prediction partition whose `artifact` differs from its run record's or whose contract differs from its config; `DisplayPixels` rejects a bundle whose contract has reference `dapi`; Prod3-2 provenance iterates markers; RHU2 rejects a `spatial.json` whose `probability_columns` disagree with the contract; `rhu2_prepare` writes a v2 snapshot with the default contract; a `FieldCropSource` built after a config change carries the new config.

### I: Frozen paths and docs (WF5)

Owns legacy `train.py`, `infer.py`, `roc.py`, `rhu2_train.py`, `rhu2_embed.py`, `rhu2_predict.py`, `legacy_labeler.py` (call sites only), `dino/dino/review_activation.py`, `test_rhu2_activation.py`, `test_legacy_labeler.py`, and all READMEs. Adapts under `DEFAULT_CONTRACT`: `train.py:396` passes `markers=DEFAULT_CONTRACT.markers`; `legacy_labeler.py:578`, `:649`, `:677`, `:752`, `:903` pass `contract=DEFAULT_CONTRACT` (`load_labels(path)` at `:330` is unchanged); `review_activation.py:36` calls `combined_uncertainty({"brdu": ..., "edu": ...}, {"brdu": thresholds["brdu"], "edu": thresholds["edu"]})`, and `activate` requires `SPATIAL_REVIEW.contract(manifest) == DEFAULT_CONTRACT` and keeps the snapshot's version. `rhu2_train.py`, `rhu2_embed.py`, `rhu2_predict.py`, `rhu2_intensity.py`, `infer.py` and `roc.py` need no change, because the shared calls they make keep their signatures. Documents `markers.json` including `badge_colors`, `--markers`, `--channel`, `--input-channel`, the record versions, normalization, the resume rule and its distinction from derived bundles (the classifiers `periodic apply`/`measure` write, `dino/README.md:255-325`, and the bundle `bind_spatial_subtraction` returns, `:440-458`, are written in the current version), and the reviewer controls in `dino/README.md` and `dinoapp/README.md`. Tests: the existing suites for these modules pass, and an activation of a v1 and of a v2 snapshot each keeps its version.

## 6. Out-of-repo callers

These files lie outside the dinoinfer worktrees. The owning slice writes the exact change in its report; the main session applies it to thesis-internal at the WF4 integration gate (D10, D11), runs `test/test_dino_call_provenance.py` there, and runs `ruff check --output-format=concise` on each changed Python file and `scripts/check_imports.py --no-baseline` for the changed imports, per thesis-internal's AGENTS.md. No compatibility alias is added.

| Caller | Change | Owner |
| --- | --- | --- |
| `thesis-internal/scripts/dino_train_405.sh:37-40` | `--input-channel brdu=brdu --input-channel brdu=pi --input-channel edu=edu --input-channel edu=pi` (D7, 4.9) | C |
| `thesis-internal/scripts/dino_neighbor_eval.py` | drop `TARGETS` from the `dino.zarr_train` import (`:17-23`); after loading the bundle (`:236`), `contract = bundle_contract(bundle, where=str(args.model))`; `_load_training_rows(args.barrage_dir, contract=contract)` (`:241`); pass `contract.markers` to `_measure_context` and use it where `TARGETS` was (`:60`, `:75`, `:276`, `:287`) | C |
| `thesis-internal/scripts/dino_render_calls.py:103` | `calls, prediction_records = load_calls(...)`, then `brdu, edu = calls["brdu"], calls["edu"]`; `:104-111` unchanged | D |
| `thesis-internal/test/test_dino_call_provenance.py` | the fixture `metadata` (`:16-25`), shared by `complete.json` (`:26`) and the partition metadata (`:29-31`), gains `artifact="dino-zarr-infer-v2"`, because `INFERENCE.contract` rejects an unversioned completion or partition; `:38` becomes `calls, records = load_calls(...)` with assertions on `calls["brdu"]` and `calls["edu"]`; `:47` and `:53` are unchanged | D |
| `thesis-internal/thesis/gam/gam/uv_soft_joint.py` | in `load_labeled_classifier_data` (`:83-88`), whose science stays BrdU/EdU, first `require_same_contract(DEFAULT_CONTRACT, bundle_contract(bundle, where="classifier bundle"), expected_from="uv_soft_joint (BrdU/EdU)", actual_from="classifier bundle")`; then pass `markers=DEFAULT_CONTRACT.markers` to `assemble_classifier_features` (`:138-143`). `load_classifier_context(bundle)` (`:132`) keeps its call (D11) | D |
| `thesis-internal/thesis/scripts/prepare_uv_gam.py:128-129` | in the prediction-override branch only (`:124-134`), replace the two literal version comparisons with `CORTEX_RESCORE.recognizes(prediction_record)` and `INTENSITY_MODEL.recognizes({"schema": prediction_record.get("model_schema")})`, raising today's "The selected pulse prediction artifact has an invalid contract" when either fails, then require `CORTEX_RESCORE.contract(prediction_record, where=str(prediction_complete_path)) == DEFAULT_CONTRACT` through `require_same_contract`. The partition-count, model-hash and labeled-cells-hash checks stay. The default branch (`:135-140`) pins the prediction completion by its recorded file hash and is unchanged (D11) | D |
| `thesis-internal/scripts/dino_reduce_umap.py:87` | none; `read_embedding_artifact(path)` keeps its signature | n/a |

## 7. Open questions for the author

None. The author settled the GAM callers (D11) and the periodic records (D12) on 2026-10-15.
