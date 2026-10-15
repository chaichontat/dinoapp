# DINO cell classification

`thesis-dino` trains and runs frozen Cell-DINO feature extractors with one logistic-regression head per binary marker. The packaged [marker configuration](#marker-configuration) defines the markers BrdU and EdU.

The separate `dino-zarr` executable reads large fused and masked-label Zarrs directly. Its commands take the marker configuration. The `dino` commands for H5AD and workspace inputs classify BrdU and EdU only.

The package owns the complete barrage and classification workflow under one `dino` command:

- `dino barrage`: sample cells and render the labeling barrage
- `dino clean-labels`: normalize and deduplicate the label table
- `dino train`: train and evaluate a model bundle from a labeled barrage
- `dino infer`: classify cells from one dataset or ROI
- `dino label`: launch the [`dinoapp`](../dinoapp/README.md) reviewer
- `dino model-sweep`: compare classical classifiers on barrage features
- `dino rf-search`: tune the classical random-forest classifier
- `dino rf-train`: fit the classical random-forest model bundle
- `dino run`: run inference across several datasets
- `dino roc`: render ROC curves from inference outputs
- `dino mnist`: run the standalone DINOv3 MNIST validation experiment

## Installation

The base installation provides source and crop readers, fitted pixel corrections,
review artifact validation, and crop rendering. It does not install the reviewer,
Torch, DINOv2, or OSQP. Native review requires Python 3.12 or newer. The marker definitions come from the
`dino-markers` package in the repository's `markers/` directory, a core dependency
that is installed from there alongside this package.

Use `[compute]` for encoder execution, spatial fitting, direct-Zarr preparation,
and pixelwise regionprops measurements. The uv workspace resolves `squisher` and
`squisher-segment` from GitHub and `dinov2` from the fork
[chaichontat/dinov2](https://github.com/chaichontat/dinov2) at commit `76d1757`
(`facebookresearch/dinov2` commit `7b187bd` with the Cell-DINO folders packaged).
Run this command from the `dinoinfer` repository root:

```bash
uv sync --package thesis-dino --extra compute
```

Use `[legacy]` for the H5AD pipeline (`dino train`, `infer`, `barrage`, `run`,
`roc`, `rf-*`, `model-sweep`). It also needs the `fishtools` checkout, which
supplies the microscopy workspace, crop, and barrage readers. fishtools requires
NumPy 1 and squisher NumPy 2.4 or newer, so install it with `./dino[legacy]` in a
separate environment.

Cell-DINO requires the selected Cell-DINO checkpoint and `dinov2` installed from a git commit; models and transforms load from the installed package. Model bundles record that installation as `dinov2_source` (git URL and commit), and inference refuses a bundle whose `dinov2_source` differs from the installed `dinov2`. Bundles and feature caches that record only `model_source_sha256` load without a source check. Outside the uv workspace, install it with `pip install --no-deps "dinov2 @ git+https://github.com/chaichontat/dinov2@76d1757bbb72097b7f0126c4d0081f4fea483e67"`; `--no-deps` preserves the environment's Torch stack. The optional MNIST validation command requires DINOv3 to be importable in the environment.

## Marker configuration

Commands that name markers read them from a marker configuration, a JSON file
with one reference channel and one to seven binary markers. The packaged default
`dino_markers/markers.json` defines the reference `pi` and the markers `brdu` and
`edu`:

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

A key starts with a lowercase letter and continues with lowercase letters, digits,
or underscores. Keys are unique across the reference and the markers, may not
equal a reserved column or record name (`dino_markers.RESERVED_KEYS`, for example
`cell`, `label`, `labels`, and `cell_sha256`), and may not derive the same column
or record name as another key or a reserved name (`brdu` with `p_brdu`, `embedding`
with `embedding_model`). Labels are nonempty and unique. A
color is `#rrggbb` or a list of three numbers in [0, 1].

`presentation` holds display settings only. `crop_panels` lists the reference and
every marker once and sets the panel order of combined barrage crops. It is
independent of the marker order that defines labels and models.
`negative_color` colors a call with every marker negative, and `uncalled_color`
colors a cell without a saved call. A positive state blends its markers' colors by
adding them per channel and clipping at 1. The optional `badge_colors` list sets
the reviewer badge of named states: `state` maps every marker key to `0` or `1`,
and `null` is "no saved call". A state without an entry uses its blended color,
with a dark or white text color chosen from its luminance.

A three-marker configuration with the reference `dapi` is:

```json
{
  "version": 1,
  "reference": {"key": "dapi", "label": "DAPI"},
  "markers": [
    {"key": "edu", "label": "EdU", "color": "#00ff00"},
    {"key": "brdu", "label": "BrdU", "color": "#ff00ff"},
    {"key": "pcna", "label": "PCNA", "color": "#00ffff"}
  ],
  "presentation": {
    "crop_panels": ["dapi", "edu", "brdu", "pcna"],
    "negative_color": "#404040",
    "uncalled_color": "#999999"
  }
}
```

### Marker contract

The marker contract is the reference key and the ordered marker keys, for example
`{"version": 1, "reference": "pi", "markers": ["brdu", "edu"]}`. Every artifact
that depends on the markers records it, and each reader compares the recorded
list with the active one. Labels, colors, and panel order are presentation and
never invalidate features or models, but marker order does. A reordered list of
the same markers is a different contract, and every command fails on it with a
message naming both orders. A marker count above seven is rejected because call
codes are `uint8`.

Everything persisted per marker derives from its key. The label column is the
key `k`. Predictions use `p_k` and `k`, bundles use `k_cutoff`, `k_model`, and
`k_feature_scaler`, barrage indexes use `k_image_path` and `k_mean`, `k_min`,
`k_max`, `k_median`, `k_std`, and regionprops tables use `intensity_k_mean`. Under
the default configuration these are the `brdu` and `edu` names. A call code is the
label read as a binary number with the first contract marker as the most
significant bit, and "no call" is `2**N`; for the default configuration the code is
`edu + 2 * brdu` and "no call" is 4.

### Commands

Without `--markers`, a command that creates an artifact uses the packaged default.
A command that consumes an existing artifact takes its contract from the
artifact. There, `--markers PATH` only asserts compatibility: it must equal the
artifact's contract, and it never reinterprets the artifact.

| Command | `--markers PATH` | Contract from |
| --- | --- | --- |
| `dino-zarr prepare` | Defines the new source; `--channel KEY=PATH` supplies each channel | The file, or the packaged default; an output directory that already declares a contract must equal it |
| `dino-zarr barrage` | Asserts | The source; an output directory that already declares a contract must equal it |
| `dino-zarr train` | Asserts | Every `--barrage-dir` and the target source, all equal |
| `dino-zarr infer` | Asserts | The model bundle, which must equal the source's |
| `dino-zarr periodic fit` | Defines the fitted targets, in order | The file, or the packaged default |
| `dino-zarr periodic apply`, `measure` | none | The field bundle's `targets` and the classifier's contract, which must agree |
| `dino clean-labels` | none | The barrage directory, under the label-file lock |
| `python -m dino.cortex_score`, `dino.joint_train`, `dino.review_activation` | none | The bundles and records they read |

A source prepared with `--markers` carries that file as `markers.json` beside
`source.json`, and a barrage carries the resolved configuration as `markers.json`
in its directory. A reader that finds a `markers.json` uses its presentation. A
directory without one uses the packaged default. A different `markers.json` is
rejected when its contract differs from the contract the directory's records
declare. A new prepare or barrage run fails when the directory's `markers.json`
differs from the run's configuration in any way, including presentation; supply that file with `--markers` or
use a new output directory. Every
`labels.csv` write also checks the directory's contract under the label-file lock.

### Artifact records

Every new artifact carries the current version of its family and a
`marker_contract` field, under the default contract too. A current record without
the field fails with an error naming the artifact. An existing artifact is never
rewritten. The reader normalizes a legacy record in memory to the default
contract before it compares records:

| Artifact | Current version | Legacy version, read as the default contract |
| --- | --- | --- |
| Prepared source `complete.json` | `dino-zarr-source-complete-v2` | `dino-zarr-source-complete-v1` |
| Barrage `run.json`, `complete.json` | `dino-zarr-barrage-v4`, with `crop_panels` | `dino-zarr-barrage-v3`, with panels PI, EdU, BrdU |
| Feature cache metadata | `dino-zarr-features-v3` | `dino-zarr-features-v2` |
| Model bundle | `dino-zarr-model-v7` | `dino-zarr-model-v6` |
| Intensity-context bundle | `dino-knn-intensity-v2` | `dino-knn-intensity-v1` |
| Joint bundle | `dino-joint-dataset-v2` | `dino-joint-dataset-v1` |
| Inference `contract.json`, `complete.json`, partition metadata | `dino-zarr-infer-v3` | `dino-zarr-infer-v2` |
| Cortex rescore `complete.json`, partition metadata | `dino-cortex-rescore-v2` | `dino-cortex-rescore-v1` |
| RHU2 `spatial.json` | `rhu2-spatial-review-v2` | `rhu2-spatial-review-v1` |
| Reviewer change journal | `dino-review-journal-v1` | records without `schema` |

A directory without records, such as a hand-made barrage, has the default
contract unless it holds a `markers.json`. The periodic field bundle,
`correction.json`, and the measure records keep their versions: the field
bundle's ordered `targets` list is their recorded contract.

A run that already holds a versioned record continues in that version. An
interrupted inference, rescore, or barrage run, an extended feature cache, an existing
change journal, and an existing RHU2 `spatial.json` keep the version they were
started in, so no run mixes versions. A legacy run can continue only under the default contract. A new
output directory writes the current version throughout. Records of one run are
compared with each other exactly, so partitions and completion records must agree
in version with their run record.

A derived artifact is a new file built from an existing one, and it is written in
the current version with the input's contract while the input is left
untouched. The classifiers that `dino periodic apply` and `measure` write and the
bundle that `dino.spatial_pixels.bind_spatial_subtraction` returns are derived: a
`dino-zarr-model-v6` or `dino-knn-intensity-v1` input yields a v7 or v2 bundle
with the default contract. Source files, chunk partials, `display.json` caches, and
embedding partitions carry no marker identity and are not stamped.

The legacy H5AD pipeline (`dino train`, `infer`, `barrage`, `rf-*`, `model-sweep`,
`roc`, `run`), the RHU2 training, embedding, and prediction modules, the residual
BrdU analyses, and the BrdU/DAPI spatial subtraction fitting classify BrdU and
EdU only and run under the default contract.

## Inputs

Training expects a barrage directory containing `index.csv` and `labels.csv`. Generate and label it with:

```bash
dino barrage --out-dir output/brdu_edu_barrage
DINO_REVIEW_BARRAGE_DIR=output/brdu_edu_barrage dino label
```

The reviewer is the [`dinoapp`](../dinoapp/README.md) package in this
repository's `dinoapp/` directory. The
`dino label` compatibility command runs `python -m dinoapp` using the current
Python interpreter. Install the reviewer separately in that environment. It
discovers datasets from the barrage index and uses explicit
`{key}_image_path` columns for each marker (`brdu_image_path` and `edu_image_path`
by default) and `combined_image_path` when present. Choose **Review target → Prod2** or **Prod3** for spatial
field and model-miscall modes; native dependencies load only after selection.
The adapters, renderer, label store, legacy labeler, and reviewer tests are
maintained in `dinoapp`; no reviewer source files are retained in this package.
The native mode uses this package's existing crop readers and barrage renderer.

From the repository root, install `dinoapp` in the same environment:

```bash
conda run -n multi python -m pip install --no-deps -e './dinoapp[native]'
DINO_REVIEW_SOURCE="Native fields" DINO_NATIVE_DATASET=Prod3 \
  conda run -n multi dinoapp --server.port 8502
```

For every labeled cell, training reads its registered fused image and segmentation, masks the crop to the target cell, and places each marker's input channels in the model channels. The frozen backbone produces features; separate cross-validated logistic regressions classify each marker. The held-out test fraction defaults to 20%.

Both trainers use fixed-seed
stratification policies on the joint label, then on each marker in contract order
(BrdU, then EdU by default), checking partition capacity
and `--limit`. A supported split needs both classes for each marker in the
held-out partition and at least two of each class in training; otherwise
training fails before feature extraction.

Barrage rebuilds preserve annotated cells only when the previous index proves
the same dataset/ROI identities. `dino clean-labels --apply` holds the reviewer's
label-file lock while reading, backing up the exact original bytes, and
publishing the cleaned table.

## Usage

```bash
dino train --barrage-dir output/brdu_edu_barrage

dino infer \
  --h5ad /path/to/cells.h5ad \
  --model-path output/brdu_edu_barrage/brdu_edu_cell_dino_hpa_vitl16_logreg.joblib \
  --fused-zarr /path/to/fused.zarr \
  --segmentation-zarr /path/to/segmentation.zarr \
  --dataset DATASET

dino run --h5ad /path/to/cells.h5ad

dino roc \
  --labels-csv output/brdu_edu_barrage/labels.csv \
  --dinoinfer-dir /path/to/dinoinfer \
  --bundle-path output/brdu_edu_barrage/brdu_edu_cell_dino_hpa_vitl16_logreg.joblib
```

Inference writes `brdu_prob` and `edu_prob`. For a single dataset/ROI, pass the
exact fused and segmentation stores together so inference does not rediscover
them from global workspace roots. Outputs include model and input provenance
and refuse to overwrite an existing destination. Legacy `dino infer --skip-existing`
assumes the image and segmentation stores are immutable. Use a fresh output
destination after changing either store; this path does not detect in-place
pixel or label changes. Feature caches are bound to
the checkpoint, the installed `dinov2` commit, preprocessing, and masked-cell inputs;
bundles created before that metadata was added must be retrained.

Held-out ROC evaluation requires the bundle's saved `train_cells` and
`test_cells`; counts alone cannot reconstruct membership. Supplying a bundle
also requires each prediction artifact to identify that same model. Score
integration requires finite probabilities in [0, 1] for every requested cell.

From this `dino/` package directory, build the source distribution and wheel with:

```bash
uv build
```

## Dependency architecture

Native `dinoapp` adapters consume the public runtime APIs in `thesis-dino>=0.2.0`:
`zarr_source` opens sources and reads crops, `pixel_corrections` applies fitted
fields, `crop_rendering` renders cells, `review_contracts` validates selected
predictions, and `classifier_features` validates the saved feature layout.
The reviewer owns navigation and annotation edits; DINO owns training,
inference, and activation. DINO does not import the reviewer.

`load_source` verifies stored source contents and live image geometry.
`open_review_source` verifies manifest identity and array metadata for native
display. Both preserve their own validation requirements. Corrected display
and inference share subtraction followed by periodic gain application;
inference then performs its existing preprocessing and masking.

Fitting and saved-artifact validation share `spatial_evaluation` rules.
Validation does not import the fitting runner. Fitted spatial models retain
their `dino.spatial_subtraction.SpatialSubtraction` serialization identity.

## Direct Zarr workflow

Prepare the globally reconciled masked labels once, then generate the barrage without per-cell Zarr reads:

```bash
dino-zarr prepare \
  --labels /working/eduseg/20260808-prod2/cp-large/masked/labels.zarr \
  --channel pi=/working/eduseg/20260808-prod2/234-3-514561638/fusion-405-l0-g20-d40/fused.ch0.ome.zarr \
  --channel edu=/working/eduseg/20260808-prod2/234-3-514561638/fusion-514-l0-t1k/fused.ch0.ome.zarr \
  --channel brdu=/working/eduseg/20260808-prod2/234-3-514561638/fusion-638-l0/fused.ch1.ome.zarr \
  --offset-zyx 0 4304 2911 \
  --out-dir /working/eduseg/20260808-prod2/dino-src

dino-zarr barrage \
  --source-dir /working/eduseg/20260808-prod2/dino-src \
  --out-dir /working/eduseg/20260808-prod2/dino-barrage
```

`--channel KEY=PATH` is repeated once for each key of the marker contract, the
reference and every marker, so the keys must equal that set exactly. With
`--markers PATH`, the file defines the source (for example the three-marker
configuration above with `--channel dapi=... --channel edu=... --channel brdu=...
--channel pcna=...`). Preparation writes `dino-zarr-source-complete-v2` with the
contract and a `markers.json` beside `source.json`; `source.json` and its digest do
not depend on the configuration. The barrage takes its contract from the source,
renders one panel per `crop_panels` entry, and writes `markers.json`,
`{key}_image_path` and `{key}_mean`, `_min`, `_max`, `_median`, and `_std` columns
for every marker. `--display-max KEY=VALUE` overrides the rendered maximum of any
contract channel key.

For native postprocessed labels, `prepare --measured-props /path/to/props.parquet`
reuses the regionprops table. The label store must record its indexed
`volumes.npy` reference; preparation verifies complete label coverage, matching
areas, finite centroids within the grid, and valid largest-area slices. The
completion record binds the property file bytes. Use `--dataset NAME --roi NAME`
together to specify the cell namespace when the storage path does not follow
the dataset/ROI layout. A new segmentation requires a new ROI namespace.

Launch `dinoapp` or the `dino label` compatibility command, choose **Review target → Crop directory**, and enter
`/working/eduseg/20260808-prod2/dino-barrage` in the **Barrage directory** field. Once
annotations are complete:

```bash
dino-zarr train \
  --barrage-dir /working/eduseg/20260808-prod2/dino-barrage \
  --out-dir /working/eduseg/20260808-prod2/dino-model

dino-zarr infer \
  --source-dir /working/eduseg/20260808-prod2/dino-src \
  --model-dir /working/eduseg/20260808-prod2/dino-model \
  --out-dir /working/eduseg/20260808-prod2/dino-infer
```

`--input-channel TARGET=CHANNEL` names the source channels that fill a marker's
model input slots. It is repeated, in flag order, for each slot of a target. The
target must be a marker of the contract and the channel a key of the reference or
a marker, unique within the target, and no more than the encoder's input slots.
Under the default contract a target without the flag uses the channels `brdu` and
`edu`. Under any other contract every marker must be given its channels. This
example pairs each marker with `pi`:

```bash
dino-zarr train --barrage-dir BARRAGE --out-dir MODEL \
  --input-channel brdu=brdu --input-channel brdu=pi \
  --input-channel edu=edu --input-channel edu=pi
```

Training writes a `dino-zarr-model-v7` bundle and `dino-zarr-features-v3` cache
metadata with the contract, and `dino-zarr infer` writes
`dino-zarr-infer-v3` records, as listed under [Artifact records](#artifact-records).
Prediction partitions hold `cell`, `dataset`, `roi`, `label`, then `p_k` for each
marker in contract order, then each marker's binary call `k`.

For large sources, run one chunk-preserving shard per GPU. Each label chunk is
assigned wholly to one shard, so workers do not duplicate the expensive label
decompression:

```bash
CUDA_VISIBLE_DEVICES=0 dino-zarr infer ... \
  --shard-count 2 --shard-index 0 --out-dir dino-infer/shard-000 &
CUDA_VISIBLE_DEVICES=1 dino-zarr infer ... \
  --shard-count 2 --shard-index 1 --out-dir dino-infer/shard-001 &
wait
```

The two `predictions/` Parquet datasets are disjoint and together contain every
requested cell. Read them as partitioned datasets instead of concatenating ten
million rows in memory. For recovery granularity, use more shards than GPUs
(for example, `--shard-count 16`) and dispatch two shard indices at a time;
completed shard directories are skipped when inference is rerun with `--resume`.

Inference preserves the prepared spatial order and precomputes each batch's
exact label-plus-channel Zarr chunk footprint before issuing I/O. It fetches
each required compressed chunk once through a persistent eight-worker pool.
The defaults `--batch-size 32 --cache-gb 4 --prefetch-workers 8` are measured
for the current BF16-SDPA and fused-Zarr layout. A plan that cannot fit the
cache fails before decompression. Results are committed every 100,000 cells
under `predictions/`; rerunning with `--resume` skips completed partitions.

Source manifests identify stored level-0 content, including encoded image and
label chunks. Preparing or loading a source streams every stored file under each
selected array through SHA-256 in 1 MiB blocks, without decompressing chunks.
This costs a full read of the compressed source on each command invocation;
input stores must remain unchanged throughout the scan and processing. Changes
to pixels, labels, metadata, or encoded storage invalidate the existing source
identity and its preparation, barrage, model, and feature-cache contracts. Older
metadata-only manifests must be regenerated in a new preparation directory,
followed by the dependent barrage/model workflow; they are not silently upgraded.

Inference publishes `contract.json` before processing. It binds the marker
contract, the prepared table bytes, requested input and cell order, model bundle, source content,
external intensity-context table when used,
classification cutoffs, shard, precision, and partition layout. Resume validates
saved prediction metadata and cell IDs and, when requested, BF16 embedding shape
and ordered cell identity, including for completed runs. A valid interrupted
partition pair can be finished without replacing its existing file. Changed
contracts, malformed partitions, legacy outputs lacking a run contract, and
`--no-resume` on an occupied directory fail while preserving existing files; use
a new output directory for a different run. Preparation likewise preserves
published partial files and can finish an interrupted moment/plane pair.

Training extracts frozen-model features with native bf16 weights and inputs by
default and disables xFormers so the production attention path is PyTorch SDPA.
Pass `--feature-precision float32` for an explicit fp32 run.
`dino-zarr infer --store-embeddings` requires a bf16 bundle; fp32 bundles
support prediction-only inference.
Preprocessing remains float32 before the transformed batch is cast to the model
dtype. Precision and native execution mode are bound to the feature cache,
stored in the model bundle, and enforced by direct-Zarr inference. Regenerate
older autocast caches and model bundles before using them with this path.

Use `--feature-cache /path/to/features.joblib` to reuse embeddings across
annotation-only refits. Cache metadata includes the source, model hashes,
channels, preprocessing, and precision; a contract change invalidates the
entries instead of mixing incompatible embeddings. Without this option, the
cache remains under `<out-dir>/features.joblib`.

The barrage displays the raw crops of the reference and each marker as the panels of `presentation.crop_panels`, PI, EdU, then BrdU by default (raw 405/514/638 nm for the production source). The masked labels only define cell pixels and outlines. Training selects a separate probability cutoff for each marker on the held-out set; inference retains probabilities alongside the binary calls.

### Read inference calls

`dino.call_provenance.load_calls(inference, expected_cells, source_hash)` validates
every shard against its `complete.json` and returns `(calls, records)`. `calls` maps
each marker key, in contract order, to its binary call array over the requested
cells, and `records` lists the shard records. The contract is the one the shards'
`complete.json` files record, and all shards must agree; a `dino-zarr-infer-v2` and
a `dino-zarr-infer-v3` shard of the default contract combine. A caller reads
`calls["brdu"]` and `calls["edu"]` by key.

### Shared periodic intensity correction

`dino periodic` (also `dino-zarr periodic`) fits one XY gain field with equal
sampling across explicit Z slabs. It uses periodic cubic tensor splines, anatomical
adjustment (`u`, `v`, ventricular distance `r_vz` in µm, and Z), and original-tile
grouped cross-validation to select 5 or 9 knots with ridge alpha 10.

```sh
dino periodic fit --cells fit-population.parquet \
  --period-xy-um 221.0358259 331.5537389 \
  --scale-zyx-um 0.6 0.28780706498795744 0.28780706498795744 \
  --z-edge 100 --z-edge 450 --z-edge 800 --z-edge 1150 \
  --z-edge 1500 --z-edge 1850 --z-edge 2200 \
  --cells-per-z 20000 --out-dir periodic-fit

dino periodic apply --model periodic-fit/model.joblib \
  --props props.parquet --classifier classifier/model.joblib \
  --scale-zyx-um 0.6 0.28780706498795744 0.28780706498795744 \
  --out-dir periodic-applied
```

`fit` takes `--markers` and fits one gain field per marker of the configuration,
and the field bundle records the ordered `targets`. Its coefficient columns follow
that order. `apply` and `measure` take the targets from the field bundle and require
them to equal the markers of the classifier's contract.

Fit population columns: unique `label`, `centroid_z/y/x` in pixels, `u`, `v`,
`r_vz` in µm, original acquisition `tile_index`, and `intensity_<key>_mean` for
each target (`intensity_brdu_mean` and `intensity_edu_mean` by default). Select the intended anatomical population before fitting;
only cells within the specified half-open Z range are sampled. Derive periods
from recorded acquisition tile placement. XY phases use centroid times voxel
spacing, with zero origin. Each Z slab must populate every held-out tile fold.

Apply streams all cells, correcting available min/mean/max intensity statistics
before the existing intensity-context loader computes cell and neighbor features.
Use `props-corrected.parquet` for subsequent context-model refits. Supplying
`--classifier` writes `model.joblib` bound to that table for `dino-zarr infer
--model-dir periodic-applied`; coefficients and embeddings are preserved. The
written classifier is a derived artifact: it is in the current version with its
input's contract, and the input file is unchanged. The
classifier must already reference the supplied uncorrected props and spacing.
Parquet provenance rejects repeat correction. Existing fitted shared-field
artifacts with the same spline/period/coefficient fields are accepted by apply.
Fit outputs include sampled cells/folds and CV metrics; apply records input hashes.

### Pixelwise correction for regionprops and embeddings

Use `dino periodic measure` to recompute intensity properties over corrected
pixels and bind the same lazy correction to the embedding input:

```sh
dino periodic measure --source-dir dino-source \
  --model periodic-fit/model.joblib --classifier classifier/model.joblib \
  --scale-zyx-um 0.6 0.28780706498795744 0.28780706498795744 \
  --workers 8 --out-dir pixel-model
dino-zarr infer --source-dir dino-source --model-dir pixel-model \
  --out-dir infer-pixel --store-embeddings
```

The source must have aligned image/label grids. The classifier must reference
that source and the same voxel spacing. Gain values are evaluated on the
original image grid before regional min/mean/max or DINO preprocessing, using
float32 multiplication. The shared model is applied to each of its targets (BrdU and EdU
by default), with raw source
pixels read through existing chunk caches. A bounded in-memory 960×960 gain-tile
cache reuses the field across Z; corrected image volumes and gain maps are never
materialized. Regionprops measurements retain their existing resumable partial
measurement tables. `pixel-model` contains measured `props.parquet`, a bound
`model.joblib`, and provenance/completion records. Classifier coefficients are
preserved for reembedding; refit classifiers using the new embeddings and props.
Like `apply`, `measure` writes a derived classifier in the current version with its
input's contract.

Completed pixel-measurement runs verify their recorded output digests before
reuse. A directory containing final properties or a final model without
completion is ambiguous and requires a fresh output directory. Measured
properties carry the periodic-correction marker and cannot be corrected again.

The older `periodic apply` command scales existing statistics at cell centroids.
`periodic measure` computes the statistics from pixelwise-corrected intensities.

### Nested smooth subtraction fitting

The installed owner `dino.spatial_subtraction_workflow` runs the established
paired raw-voxel workflow for an explicit directory and filename prefix:

```sh
python -m dino.spatial_subtraction_workflow benchmark \
  --output-dir thesis/output/prod2_brdu_dapi_subtraction --prefix prod2
python -m dino.spatial_subtraction_workflow run \
  --output-dir thesis/output/prod2_brdu_dapi_subtraction --prefix prod2
```

Use `multi` and persistent tmux for fitting. Package callers construct
`WorkflowConfig(output_dir=Path(...), prefix="prod2")` and call `benchmark(config)`
or `run(config)`. The original prod3 `smooth_subtraction.py benchmark|run`
launcher calls these APIs with its existing directory/prefix.

Inputs are `<prefix>_cortex_samples.parquet`,
`<prefix>_cortex_global_training_samples.parquet`, and four supplied
`<prefix>_cortex_cv_fold<N>.training.parquet` / `.predictions.parquet` pairs
(N=0–3). Required columns are exact native `z_index`, `y_index`, `x_index`,
physical `x_um`, `y_um`, `z_um`, and raw `brdu`, `dapi`; outer predictions also
provide the established global comparator in `global_predicted`. Legacy prod3
training tables join physical coordinates from the saved samples by native
indices, preserving row order. The producer supplies positive-support,
trimmed training rosters capped at 50,000. This owner neither resamples nor
recomputes trimming, and never reads raw volumes.

The fixed method uses q=0.05, three X/four Y cubic knots, nonnegative alpha,
the exact pinball plus second-difference penalty, and lambda grid
`[0.001, 0.01, 0.1]`. Three inner folds hold out entire 250 µm XY blocks,
stratified on 100 µm Z bands with seed 83261. Inner bases fit only inner
training rows; observation-weighted inner pinball loss selects lambda, with
exact ties favoring the larger value. No outer test observations enter
selection. Final-model selection runs independently on the saved global
training roster. Up to three independent candidates run concurrently.
OSQP default absolute/relative tolerances remain 0.0005/0.000001, caps 60,000
iterations/240 seconds, with the existing exact `solved` and reconstructed
raw-feasibility/objective-gap gates. The benchmark uses the established
3,000-row sample (seed 932, lambda 0.01).

Numerical precision can be set explicitly with
`--solver-absolute-tolerance` and `--solver-relative-tolerance`, or the matching
`WorkflowConfig` fields. These settings apply uniformly to the benchmark,
inner candidates, outer fits and final fit; the statistical objective and
independent acceptance gate remain unchanged. Prod2's 3,000-row benchmark
required tighter precision:

```sh
python -m dino.spatial_subtraction_workflow run \
  --output-dir thesis/output/prod2_brdu_dapi_subtraction --prefix prod2 \
  --solver-absolute-tolerance 5e-5 --solver-relative-tolerance 1e-7 \
  --solver-max-iterations 180000 --solver-time-limit 600
```

At those settings the exact failed benchmark passed with status `solved`,
4,075 iterations, 0.7195 s solver time, raw primal infeasibility 1.28786e-11,
relative objective gap 5.57510e-15 and mean-objective gap 2.27374e-13 raw
intensity units. The preceding default-precision result had been rejected by
the unchanged 1e-5 relative-gap gate. A cached fit made at looser precision
than requested raises an explicit error rather than being silently reused or
refitted. Accepted fits at equal or tighter precision remain reusable; prod3
default settings are preserved.

Iteration and time budgets are explicit `WorkflowConfig.solver_max_iter` /
`solver_time_limit` fields and CLI `--solver-max-iterations` /
`--solver-time-limit` options. Defaults remain 60,000 iterations and 240 s.
Increasing these budgets preserves already accepted checkpoints; it does not
change the objective, precision or acceptance gate. Prod2 outer fold 1,
inner fold 2 at lambda 0.1 exhausted the default iteration cap. Its exact
33,161-row training subset passed with a 180,000 iteration/600 s budget at
the same tighter tolerances: status `solved`, 68,125 iterations, 227.3825 s
solver time, 227.4970 s fit wall time, raw primal infeasibility 0.0496028893,
relative gap 6.531607e-8 and mean-objective gap 0.00000262611 raw intensity
units. Benchmark process peak RSS was 416,428 KiB. Subsequent prod2 fits use
the larger explicit budget; prior accepted models retain their own recorded
solver budgets and are reused exactly.

Outputs use `<prefix>_smooth_*`: inner-fold assignments, each candidate's
model/solver/prediction/metric checkpoints, tuning tables and selection JSON;
four outer models/predictions/metrics; combined held-out predictions and
XY/depth metrics; `<prefix>_smooth_final_model.joblib` and the run record.
Existing accepted checkpoints and selections resume in their original names.
The directory must belong to one fixed input roster; this workflow does not
rebind existing checkpoints to changed input tables. The launcher owns logs,
completion signaling and upstream source provenance.

Validate the saved fits without refitting:

```sh
python -m dino.spatial_subtraction_validation \
  --output-dir thesis/output/prod2_brdu_dapi_subtraction --prefix prod2
```

The package API is `validate(output_dir: Path, prefix: str)`. An explicit
`destination_dir=Path(...)` (CLI `--destination-dir`) redirects the validation
outputs to an existing directory while leaving the input artifacts untouched.
The original prod3 validation launcher uses this same owner.

Validation checks all 45 inner candidates, four outer models and the final
model against their exact training domains, sample counts, nonnegative finite
alpha, finite offsets and solver acceptance. It recomputes each saved model's
scaled pinball-plus-roughness objective and checks its recorded objective and
primal/dual gap. It reconstructs the deterministic
inner folds, verifies whole-block separation and observation-weighted smoothing
selection, recomputes inner metrics and exact saved predictions, and verifies
that paired outer test values and global predictions remain unchanged. Outer
train/test XY blocks must be disjoint, and the combined test roster must match
the four saved outer results. Outputs are `<prefix>_smooth_validation.json`,
`_numerical_validation.tsv`, `_pooled_regions.tsv` and `_pooled_depth.tsv`.
The upstream producer separately validates raw sampling, trimming and supplied
fold-preparation provenance; the model validator does not infer missing
preparation records.

### Smooth raw BrdU subtraction for reembedding

Bind a saved `dino.spatial_subtraction.SpatialSubtraction` to a copied inference
bundle through `dino.spatial_pixels.bind_spatial_subtraction`. It returns a derived
bundle in the current version with the input's contract and leaves the supplied
bundle unchanged. The result adds the `spatial_subtraction` contract and preserves
the encoder, classifiers, context properties and existing periodic correction.
Subtraction supports only BrdU against the `pi` channel verified as 405 nm DAPI:
a bundle whose contract has another reference or no `brdu` marker, and a
subtraction of another target or reference, are rejected:

```python
from pathlib import Path
import joblib
from dino.spatial_pixels import bind_spatial_subtraction

corrected = bind_spatial_subtraction(
    bundle,
    model_path=Path("prod3_smooth_final_model.joblib"),
    source_manifest=source.manifest,
)
joblib.dump(corrected, output_model_dir / "model.joblib")
```

The supplied bundle must reference the supplied source identity. A source
metadata refresh must first verify unchanged raw stores, grids, labels and cell
roster and record the old/new source provenance in its run setup. The binding
requires matching BrdU and `pi` physical OME grids with ZYX micrometer axes and
recorded scale then translation. The source producer must verify `pi` as true
405 nm DAPI. Physical XY is the full fused crop index times spacing plus the
recorded physical translation, including the label-to-fused crop offset.

The existing lazy reader adds raw `pi` to crop reads and chunk prefetching when
this contract is bound; it does not add a channel to the encoder. BrdU becomes
`clip(rint(raw BrdU - alpha(X,Y)*raw pi - beta(X,Y)), 0, 65535).astype(uint16)`.
Pixels with either raw marker or reference zero remain zero. This matches the
reviewed smooth crop application. Existing float32 periodic gains follow the
subtraction; the existing highpass, quantization and target mask then run
unchanged. EdU retains its existing path. No corrected volume is materialized.
The installed model owner retains signed residuals for diagnostics.

Run the existing `dino-zarr infer --model-dir ... --store-embeddings` command.
For direct batch inference, pass `spatial_subtraction=bundle.get("spatial_subtraction")`
to `model_crop_reader`. The bound artifact digest, full geometry and operation
are recorded in the model bundle. Changing the binding changes the inference
resume identity; a mutated subtraction artifact is rejected before completed
partitions can be skipped. Classifier outputs remain incidental to reembedding;
binding does not refit or activate a classifier.

### Joint classifier across datasets

`python -m dino.joint_train --config config.json --output OUT --workers N` refits the
marker-specific logistic classifiers on the labels of several datasets that
share one encoder (the trainer checks the encoder settings match). Embedding
coefficients are shared. Each dataset has its own regularized intercept, as in
the RHU2 section offsets, and its own coefficients on the intensity terms named
by the config's `context_features`: either `log1p_cell_mean`,
`log_cell_over_local_mean` and `log1p_dapi_mean`, or those three plus
`robust_z_log_marker_over_dapi_k100`. The fourth term is the cell's
log((marker + 1)/(DAPI + 1)) centered on the median and divided by 1.4826 × MAD
of that ratio over its 100 nearest cells, a local background reference that
tolerates the labeled minority of a neighborhood. The
validation holdout is stratified by dataset and the joint label of the contract
markers, falling back to dataset and each marker in order, and one
cutoff per marker is chosen on it. `metrics.json` also reports out-of-fold
metrics per dataset. Within one dataset the intercept and other-dataset columns
are constant, so `OUT/deploy/<dataset>/model.joblib` stores the exactly
collapsed `dino-knn-intensity-v2` bundle for that dataset with the marker
contract of the embedding bundle; the scorers read it directly. Example: `/working/eduseg/prod2-prod3-joint/config.json`.
`--workers` sets the processes for reading labeled embeddings and for the
single-threaded logistic fits (C search, final and out-of-fold fits); results
are identical for any worker count.

`python -m dino.cortex_score` takes the arguments of
`full-run2/score-cortex.py` plus `--workers N` and writes the same prediction
parquets with their provenance and `complete.json`, scoring partitions in
parallel worker processes. Its prediction files hold `p_k` and `k` for each marker, in the
contract order of the classifier bundle, and `complete.json` records
`dino-cortex-rescore-v2` with `marker_contract` and a `<key>_positive`
count for each marker. It does not copy embeddings into the output
(`stores_embeddings: false`); they stay in the source embedding run. Each worker
loads the bundle and kNN intensity context once. PCA-projected bundles are
rejected. Source shards and partitions must match their recorded counts and
layout. Reusing an output directory requires the same scoring contract and
exact output partition roster; changed selections require a fresh directory.
Same-contract partial runs can resume.

### RHU2 preparation and activation

`python -m dino.rhu2_train prepare` resolves physical spacing for each ROI from
its acquisition metadata or explicit `--z-step-um` and records `spacing_by_roi`
in `prepared.json`. Measurement and fitting require this prepared spacing;
an explicit fitting value must agree. Scoring uses the fitted bundle's
`spacing_by_roi` so its neighbor geometry matches training.

The RHU2 spatial snapshot is `rhu2-spatial-review-v2` with the default contract,
because it holds the frozen release's `brdu_prob` and `edu_prob` columns. Activation
requires that contract, accepts a v1 or v2 snapshot, and keeps the snapshot's version.

RHU2 rescoring marks inference incomplete before publishing ROI tables and
marks it complete only after all ROIs pass validation. Activation stages tables
in a fresh directory and switches the reviewer manifest after validation.
Previously active tables remain available if activation fails.

### Simple hierarchy with grouped nested evaluation

`dino.hierarchical_train` provides a bounded alternative for corrected cached
features: `X` contains the shared encoder embedding followed by exactly three
intensity context terms, with the same basis for every dataset. The hierarchical
family fits shared embedding and intensity coefficients and a global intercept.
Each dataset adds only three intensity coefficient deviations and one intercept
deviation. `lambda0` penalizes shared slopes; `lambda_delta` penalizes all four
local adjustments. An infinite local penalty gives a fully pooled classifier.
The base features are standardized on training rows before local columns are
constructed; masked local columns receive no additional standardization.

`nested_evaluate(X, y, dataset_index, groups, datasets=D, family="hierarchical")`
uses four outer and three inner `StratifiedGroupKFold` partitions by default.
Each split keeps spatial groups together and requires both labels in every
dataset's training and test rows. Inner macro-dataset OOF log loss selects from
`lambda0=(1,10,100)` and `lambda_delta=(1,10,100,inf)`; numerical ties favor more
pooling, then stronger shared regularization. Each outer training fold selects
one cutoff per dataset from its selected model's inner OOF predictions cast to
float32. The returned `Evaluation` stores outer OOF `probabilities`, rowwise
`cutoffs`, rowwise `fold`, dataset-specific confusion counts and metrics, and
inner selections. Supplying the same `outer_folds` and seed to `family="joint"`
compares the existing joint feature parameterization under the same nested
splits, shared penalty grid, and local cutoff rule. The joint family scales its
entire existing design within each training fold. Historical nonnested OOF
results do not provide this controlled comparison.

`fit_final` selects penalties and dataset cutoffs through grouped OOF predictions
on all supplied labels, then fits all labels. Its `FinalFit.fitted.collapse(d)`
returns an ordinary `StandardScaler` and `LogisticRegression` over the original
embedding and three context columns, preserving probabilities for dataset `d`.
These objects and `FinalFit.cutoffs[d]` can populate a `dino-knn-intensity`
deployment bundle consumed by `dino.cortex_score`.
The estimator and evaluation APIs do not activate an application model.
