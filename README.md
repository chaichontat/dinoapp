# dinoinfer

Cell image review, classifier fitting, and inference live in two Python
packages that share a third, dependency-free package of marker definitions:

| Directory | Distribution | Commands | Documentation |
| --- | --- | --- | --- |
| `dinoapp/` | `dinoapp` | `dinoapp`, `dinoapp-legacy` | [Reviewer](dinoapp/README.md) |
| `dino/` | `thesis-dino` | `dino`, `dino-zarr` | [Fitting and inference](dino/README.md) |
| `markers/` | `dino-markers` | none | [Marker configuration](dino/README.md#marker-configuration) |

The reviewer uses DINO's public source readers, pixel corrections, rendering, and
review contracts. DINO owns training, inference, and model activation. Editing
labels in the reviewer does not automatically retrain a model or run inference.

## Markers

Both packages read their markers from one configuration, `markers.json`: one
reference channel and one or more binary markers, each with a key, a label, and a
color. The packaged default is the reference `pi` with the markers `brdu` and
`edu`, and it reproduces the names, columns, and files of a BrdU/EdU barrage.
`dino-markers` implements the configuration and the artifact records that bind
every prepared source, barrage, label file, feature cache, model bundle, and
prediction run to the ordered marker keys it was made with. Marker order is part
of that contract. [dino/README.md](dino/README.md#marker-configuration) describes the
format, the commands that take `--markers`, and the artifact versions.

## Install

Run these commands from the repository root. The uv workspace resolves
`dino-markers`, `thesis-dino`, and `dinoapp` from their directories and `squisher`
and `squisher-segment` from [GitHub](https://github.com/chaichontat/squisher).
Basic crop review supports Python 3.10 or newer; native review and DINO require
Python 3.12 or newer:

```bash
uv sync --package dinoapp                        # crop review
uv sync --package dinoapp --extra native         # native field review
uv sync --package thesis-dino --extra compute    # preparation, fitting, inference
```

The `compute` sync installs `dinov2` from the fork
[chaichontat/dinov2](https://github.com/chaichontat/dinov2) at commit `76d1757`,
which is `facebookresearch/dinov2` commit `7b187bd` with the Cell-DINO folders
packaged. The root `pyproject.toml` overrides its 2023 training pins.
[dino/README.md](dino/README.md#installation) describes the checkpoint.

With pip, `dino-markers` is not on PyPI and must be installed from `markers/` in
the same command. When the native or compute extras need `squisher` and `dinov2`,
install them from GitHub first, at the commits the uv workspace pins, because the
PyPI packages of those names are unrelated.
Install `dinov2` with `--no-deps`, because its `setup.py` pins a 2023 Torch stack:

```bash
python -m pip install -e ./markers -e ./dinoapp
python -m pip install \
  "squisher @ git+https://github.com/chaichontat/squisher@4cd1423aaacedda1fe3f4326411455e271eaf872#subdirectory=squisher" \
  "squisher-segment @ git+https://github.com/chaichontat/squisher@4cd1423aaacedda1fe3f4326411455e271eaf872#subdirectory=segment"
python -m pip install --no-deps \
  "dinov2 @ git+https://github.com/chaichontat/dinov2@76d1757bbb72097b7f0126c4d0081f4fea483e67"
python -m pip install -e ./markers -e './dino[compute]' -e './dinoapp[native]'
```

The `[legacy]` extras serve the H5AD pipeline (`dino train`, `infer`, `barrage`,
`run`, `roc`, `rf-*`, `model-sweep`) and `dinoapp-legacy`. That pipeline also
needs the fishtools package, which requires NumPy 1 and cannot share an
environment with squisher (NumPy 2.4 or newer). Run it in a separate environment
with fishtools installed ([legacy install](dino/README.md#installation)).

## Preparing data

The direct Zarr workflow takes a segmentation label Zarr and one fused OME-Zarr
per channel on one voxel grid; the label grid may be a crop of it, placed by
`--offset-zyx`. It runs in the `compute` environment:

1. Write a `markers.json` if the markers are not BrdU and EdU with a PI
   reference ([format](dino/README.md#marker-configuration)). Keys must avoid the
   reserved names in `dino_markers.RESERVED_KEYS`, and marker order is part of
   every artifact's contract.
2. `dino-zarr prepare --labels LABELS --channel KEY=PATH ... --offset-zyx Z Y X
   --out-dir SRC [--markers markers.json]` binds one channel to each key of the
   reference and every marker. `--offset-zyx` is required, `0 0 0` when the label
   and fused grids coincide.
3. `dino-zarr barrage --source-dir SRC --out-dir BARRAGE` samples cells and
   renders their crops. The barrage takes its markers from the source.
4. Review the barrage in `dinoapp` (**Review target → Crop directory**), which
   writes `labels.csv` with one 0/1 column per marker.
5. `dino-zarr train --barrage-dir BARRAGE --out-dir MODEL` fits one classifier
   per marker. Outside the default markers, give each marker its input channels
   with `--input-channel MARKER=CHANNEL`.
6. `dino-zarr infer --source-dir SRC --model-dir MODEL --out-dir CALLS` writes
   `p_k` and the binary call `k` for every cell and marker.

A consuming command takes its markers from the artifacts it reads; `--markers`
there only asserts that they match. [dino/README.md](dino/README.md#direct-zarr-workflow)
gives complete commands, sharding, and resume behavior. Native review of the
datasets listed in a review configuration (`DINO_REVIEW_CONFIG`) is described in
[dinoapp/README.md](dinoapp/README.md#native-spatial-review), and the lab's
datasets and runs in [docs/lab-notes.md](docs/lab-notes.md).

```bash
export DINO_REVIEW_BARRAGE_DIR=/path/to/barrage
dinoapp --server.port 8502
dino-zarr train --help
dino-zarr infer --help
```

## Development

Install the development extras in the environment with the required local
dependencies:

```bash
python -m pip install --no-deps -e ./markers -e './dino[dev]' -e './dinoapp[dev]'
python -m pytest dino/tests/test_cli.py dino/tests/test_zarr_cli.py dinoapp/tests/test_label_store.py
ruff check path/to/changed_file.py
```

The test directories of all three packages are configured at the root. Run focused tests for
the behavior being changed. To build both source distributions and wheels:

```bash
uv build --all-packages
```

Reviewer documentation and format details are in `dinoapp/`; DINO documentation
and tests are in `dino/`; marker definitions and their tests are in `markers/`.
