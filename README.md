# dinoinfer

Cell image review, classifier fitting, and inference live in two independently
installable Python packages:

| Directory | Distribution | Commands | Documentation |
| --- | --- | --- | --- |
| `dinoapp/` | `dinoapp` | `dinoapp`, `dinoapp-legacy` | [Reviewer](dinoapp/README.md) |
| `dino/` | `thesis-dino` | `dino`, `dino-zarr` | [Fitting and inference](dino/README.md) |

The reviewer uses DINO's public source readers, pixel corrections, rendering, and
review contracts. DINO owns training, inference, and model activation. Editing
labels in the reviewer does not automatically retrain a model or run inference.

## Install

Run these commands from the repository root. Basic crop review supports Python
3.10 or newer:

```bash
python -m pip install -e ./dinoapp
```

Native review and DINO require Python 3.12 or newer. Install both local packages
in the same environment:

```bash
python -m pip install -e ./dino -e './dinoapp[native]'
```

For fitting and encoder execution, install `./dino[compute]`; H5AD and microscopy
workspace commands also require `./dino[workspace]`. See the package documentation
for the local DINOv2, fishtools, and squisher dependencies and checkpoint setup.
The uv workspace resolves `thesis-dino` from `dino/` for native reviewer installs.

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
python -m pip install --no-deps -e './dino[dev]' -e './dinoapp[dev]'
conda run -n seq pytest dino/tests/test_cli.py dino/tests/test_zarr_cli.py dinoapp/tests/test_label_store.py
conda run -n seq ruff check dino/dino/roc.py
```

Both package test directories are configured at the root. Run focused tests for
the behavior being changed. To build both source distributions and wheels:

```bash
uv build --all-packages
```

Reviewer documentation and format details are in `dinoapp/`; DINO documentation
and tests are in `dino/`.
