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

The `compute` sync installs `dinov2` from `facebookresearch/dinov2` at the commit
that provides Cell-DINO (`7b187bd`), and the root `pyproject.toml` overrides its
2023 training pins. Model loading also reads the checkout's `hubconf.py` and
requires the installed package to be that checkout, so install a clone of the same
commit over it after each sync;
[dino/README.md](dino/README.md#installation) describes the checkpoint:

```bash
git clone https://github.com/facebookresearch/dinov2 /path/to/dinov2
git -C /path/to/dinov2 checkout 7b187bd4df8efce2cbcbbb67bd01532c19bf4c9c
uv pip install --no-deps -e /path/to/dinov2
```

With pip, `dino-markers` is not on PyPI and must be installed from `markers/` in
the same command. The PyPI `squisher` and `dinov2` are unrelated packages, so
install squisher from GitHub and the DINOv2 clone first when the native or compute
extras need them:

```bash
python -m pip install -e ./markers -e ./dinoapp
python -m pip install \
  "squisher @ git+https://github.com/chaichontat/squisher#subdirectory=squisher" \
  "squisher-segment @ git+https://github.com/chaichontat/squisher#subdirectory=segment"
python -m pip install --no-deps -e /path/to/dinov2
python -m pip install -e ./markers -e './dino[compute]' -e './dinoapp[native]'
```

The `[legacy]` extras serve the H5AD pipeline (`dino train`, `infer`, `barrage`,
`run`, `roc`, `rf-*`, `model-sweep`) and `dinoapp-legacy`. That pipeline also
needs the fishtools package, which requires NumPy 1 and cannot share an
environment with squisher (NumPy 2.4 or newer). Run it in a separate environment
with fishtools installed.

## Preparing data

The direct Zarr workflow takes a segmentation label Zarr and one fused OME-Zarr
per channel, all on the same voxel grid (or offset into the label grid with
`--offset-zyx`). It runs in the `compute` environment:

1. Write a `markers.json` if the markers are not BrdU and EdU with a PI
   reference ([format](dino/README.md#marker-configuration)). Keys must avoid the
   reserved names in `dino_markers.RESERVED_KEYS`, and marker order is part of
   every artifact's contract.
2. `dino-zarr prepare --labels LABELS --channel KEY=PATH ... --out-dir SRC
   [--markers markers.json]` binds one channel to each key of the reference and
   every marker.
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
production datasets (`DINO_REVIEW_SOURCE="Native fields"`) is described in
[dinoapp/README.md](dinoapp/README.md).

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
conda run -n seq pytest dino/tests/test_cli.py dino/tests/test_zarr_cli.py dinoapp/tests/test_label_store.py
conda run -n seq ruff check dino/dino/roc.py
```

The test directories of all three packages are configured at the root. Run focused tests for
the behavior being changed. To build both source distributions and wheels:

```bash
uv build --all-packages
```

Reviewer documentation and format details are in `dinoapp/`; DINO documentation
and tests are in `dino/`; marker definitions and their tests are in `markers/`.
