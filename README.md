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

Run these commands from the repository root. `dino-markers` is a core dependency
of both packages, so install it from `markers/` in the same command. Basic crop
review supports Python 3.10 or newer:

```bash
python -m pip install -e ./markers -e ./dinoapp
```

Native review and DINO require Python 3.12 or newer. Install the three local
packages in the same environment:

```bash
python -m pip install -e ./markers -e ./dino -e './dinoapp[native]'
```

For fitting and encoder execution, install `./dino[compute]`; H5AD and microscopy
workspace commands also require `./dino[workspace]`. See the package documentation
for the local DINOv2, fishtools, and squisher dependencies and checkpoint setup.
The uv workspace resolves `thesis-dino` from `dino/` and `dino-markers` from `markers/`.

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
