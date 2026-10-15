# dinoapp

`dinoapp` lets you inspect cell images and save manual calls for binary markers
in a browser. The markers are BrdU and EdU unless the target's
[`markers.json`](#markers) defines others. Review individual cells, correct one
marker across a gallery, or inspect cells in their spatial context. A **barrage** is
a directory of cell crops with an index connecting each image to its cell.

dinoapp is one of two packages in the `dinoinfer` repository. The other,
[`thesis-dino`](../dino/README.md) in `dino/`, generates barrages, trains
classifiers, and runs inference.

## Quick start: label cell crops

Use Python 3.10 or newer on Linux or another POSIX system. Run these commands in
the Python environment where you want to install and launch the app:

```bash
git clone https://github.com/chaichontat/dinoapp.git dinoinfer
cd dinoinfer/dinoapp
python -m pip install -e ../markers -e .

export DINO_REVIEW_BARRAGE_DIR=/path/to/barrage
dinoapp --server.port 8502
```

Replace `/path/to/barrage` with your crop directory, then open the URL printed
by Streamlit. You can also launch with `python -m dinoapp`; extra arguments are
forwarded to Streamlit.

1. In the sidebar, choose **Review target → Crop directory** and check **Barrage directory**.
2. Choose **Review scope → Dataset cells** and **Review view → Cell review**.
   Correction galleries require saved labels and can be empty on a new target.
3. Choose a **Dataset** and **Cells → Unsaved**.
4. Inspect the combined crop. With two markers (BrdU and EdU by default), choose
   the cell's call from four buttons, one per combination (for example
   **BrdU− / EdU+**); the saved call is highlighted. A click saves the call and
   advances to the next cell, and clicking the call that is already saved writes
   nothing and advances. With three or more markers, set each marker with its `−`
   or `+` control. The cell is saved as soon as every marker has a value that differs
   from the saved call, without advancing. A cell with no saved call is never shown or
   filtered as all-negative; that state exists only after you save every marker as `−`.
5. Use **Undo last edit** to reverse the most recent edit for the current review
   target. Its history is shared across crop, gallery, and spatial views in this
   browser session. Switching targets keeps their histories separate; restarting
   the app clears session history. Each target also keeps its filters, loaded
   queues, field settings, and selected cell when you switch away and return.
   When a source reloads with a new model or display revision, its review queues
   reload while preserving history for the same annotation files.

Labels are saved to `labels.csv` inside the barrage directory. It is created on
the first saved label if it does not exist.

## Markers

dinoapp takes no marker input. It reads the markers of each review target from the
target's own directory: the `markers.json` in the crop or annotation directory, or
the packaged default (reference `pi`, markers `brdu` and `edu`) when the directory
has none. `markers.json` sets each marker's key, label, and color, the crop panel
order, and the badge colors; its format is described in the
[thesis-dino README](../dino/README.md#marker-configuration). The control labels,
filters, badges, field colors, and `labels.csv` columns follow it.

The artifacts that `thesis-dino` writes record the ordered marker keys, and the
directory's records and `markers.json` must agree. A hand-made barrage without
records has the default contract unless it holds a `markers.json`. Every write to
`labels.csv` first checks that the directory still has the contract the session
was opened with. If it changed, the write is rejected, `labels.csv` is left
untouched, and the next render opens a new session with its own history and
backup. Editing `markers.json` while the app runs, including a change of labels or
colors, rebuilds that target's sources and fields on the next render.

## Choose a target, scope, and view

**Review target** opens a crop directory or one of the native review targets
configured in the file named by `DINO_REVIEW_CONFIG` ([Review targets](#review-targets)).
**Review scope** chooses the dataset roster, model training disagreements, or a
queue you captured from a field or gallery. **Review view** offers the views
supported by that target and scope.

| View | Available with | What you do |
| --- | --- | --- |
| **Cell review** | Combined crops in a crop directory or native dataset; captured cells | Assign every marker from one combined crop. |
| **EdU correction gallery**, **BrdU correction gallery** | Targets with that marker's crops | Click a crop to flip that marker while preserving the others. |
| **Spatial field** | Native datasets or a crop directory with an RHU2 `spatial.json` | Select and annotate cells in their spatial context. |
| **Correction gallery** in **Training miscalls** scope | Native datasets | Review ranked model disagreements; each marker has its own card. |

There is one correction gallery for each marker that has crops, titled
`<label> correction gallery`, in the order of the configuration's display markers
(EdU, then BrdU, by default).

In a field, **Review selected cell** opens its crop and **Review cells in this
field** captures a fixed queue of eligible cells. These actions create a
**Captured cells** scope. A gallery's **Cell to inspect → Review cell** opens
combined crops separately from its marker-edit actions. Neither navigation nor
queue capture saves a label.

In cell review, **Show in field** opens the cell at its recorded crop plane and
native location. **Return to cell review** restores its queue position;
**Return to previous view** restores the gallery or exact originating field.
**Return to sampled fields** resumes the saved seed/position or RHU2 branch cursor.
Located fields keep that sampling cursor separate. Only a real field click
confirms an unannotated model call; navigating to a cell leaves it unannotated.
Captured RHU2 crops retain the originating field's marker contrast.

Crop directories have spatial access only through their explicit `spatial.json`
contract. A `display.json` binds corrected pixels; it does not link that directory's
annotations to a separately opened native target. Matching numeric labels do not
establish a relationship between targets.

Correction galleries match one filter per marker against current manual labels.
Each filter is **Any**, `<label>+`, or `<label>−`, and together they select the
saved states to show; a cell without a saved call never matches a marker filter.
Their queues stay fixed while you edit: a corrected cell remains in its
loaded position until you click **Refresh matches**. **Cell review** has the
same behavior; click **Refresh cells** to rebuild its dataset queue. Its **Cells**
option is **Unsaved**, **Saved**, or **All cells**, and **Saved** adds the same
per-marker filters. To rebuild
a captured field queue, return to the field and choose **Review cells in this field**.

Native galleries also have an **Examples** filter. **Training** and **Validation**
refer to the selected model's actual fitting roster. **All labeled** in correction
galleries can include saved annotations on cells outside that roster. The
**Training miscalls** scope uses the model's training review rows; a cell may have
a separate card for each marker's disagreement, and the **Disagreement** filter
selects one marker or any marker.

## Barrage input files

The directory must contain `index.csv` and the crop images it references.
An example index is:

```csv
cell,rank,dataset,roi,brdu_image_path,edu_image_path,combined_image_path
sample:1|10,0,sample,1,images-brdu/0.png,images-edu/0.png,images-combined/0.png
sample:1|20,1,sample,1,images-brdu/1.png,images-edu/1.png,images-combined/1.png
```

Each marker has its own image column, `<key>_image_path`: `brdu_image_path` and
`edu_image_path` here, and for example `pcna_image_path` with a `pcna` marker in
`markers.json`.

`cell` must be non-null and unique; `rank` must be unique and determines review
order. Keep cell identifiers exactly as generated. `dataset` enables dataset
filtering; `roi` adds context to crop captions. Image paths resolve from the
barrage directory and must stay inside it. An `image_path` column can supply
the crop when a marker-specific path is absent.

If `labels.csv` already exists, it must contain unique, non-null `cell` values
and a binary column (`0` or `1`) for each marker key (`brdu` and `edu` by default).
Edits preserve other cells and extra columns, including annotations outside the
displayed index.

[docs/data-format.md](docs/data-format.md) gives the full input and output
format, the image column each review mode needs, and a check to run before
reviewing.

To generate a barrage from microscopy data, use `thesis-dino`: `dino barrage`
for H5AD cell metadata, or `dino-zarr prepare` followed by `dino-zarr barrage`
for fused and masked-label Zarrs. See the [thesis-dino README](../dino/README.md).

## Run inference

The main dinoapp reviews images and saved calls. Training and inference run
through the [`thesis-dino`](../dino/README.md) package in `dino/`, which requires Python 3.12 or newer,
its dependencies, and the installed Cell-DINO backbone and checkpoint. Saving
manual labels in the app does not retrain a classifier or regenerate predictions.

Choose the inference workflow that matches your trained model:

| Input and model | Command | Output |
| --- | --- | --- |
| A source prepared with `dino-zarr prepare` and a model from `dino-zarr train` | `dino-zarr infer` | Partitioned Parquet files in `predictions/`, with `p_<key>` and `<key>` for each marker (`p_brdu`, `p_edu`, `brdu`, and `edu` by default). |
| H5AD cell metadata and a bundle from `dino train` | `dino infer` | A Parquet table containing `brdu_prob` and `edu_prob`. |

For an existing direct-Zarr source and model, run:

```bash
conda run -n multi dino-zarr infer \
  --source-dir /path/to/prepared-source \
  --model-dir /path/to/trained-model \
  --out-dir /path/to/inference-output
```

The model directory must contain `model.joblib`. Inference checks the recorded
source and feature settings and that the model and the source have the same
marker contract, applies the model's saved marker cutoffs, and
writes `complete.json` with `marker_contract` when the run finishes. Resume is enabled by default:
rerun the same command to continue an interrupted run. Use a new output directory
when changing the model or inputs.

For one H5AD dataset/ROI with explicit image stores, run:

```bash
conda run -n multi dino infer \
  --h5ad /path/to/cells.h5ad \
  --model-path /path/to/trained-bundle.joblib \
  --dataset DATASET --roi ROI \
  --fused-zarr /path/to/fused.zarr \
  --segmentation-zarr /path/to/segmentation.zarr \
  --output-path /path/to/inference.parquet
```

Replace the placeholders with the dataset, ROI, and stores matching the model's
inputs. The H5AD workflow requires the `fishtools` package as well as Cell-DINO.
Use `dino-zarr infer --help` or `dino infer --help` for additional options.

### Display new predictions in the reviewer

Generating an inference output does not select it in dinoapp. Native adapters
read the prepared model selection in `cell-review/current.json` under their
review directory. That selection binds the classifier, completed cortical
predictions, and labeled training review artifact. A generic inference output
alone does not provide this selection; use the dataset's model preparation and
activation workflow, then restart the app to reload its cached source. The
sidebar shows the active classifier identity and marker cutoffs.

RHU2 spatial review reads probabilities and cutoffs from its prepared spatial
snapshot. Its model activation workflow updates `spatial.json` after checking
full cohort coverage and cell coordinates, while preserving manual annotations
and cortical branch membership. Reload the RHU2 view after activation to load
the selected predictions. dinoapp has no inference or model activation button.

## Native spatial review

Native review targets use prepared local microscopy datasets at the paths named
in a review target configuration. To use them, those datasets must be accessible
on the machine running the app. Installing the app does not download or prepare
them.

Native review requires Python 3.12 or newer and, in the same environment,
`thesis-dino` from this repository's `dino/` directory, `dino-markers` from
`markers/`, and the local `squisher` package. The native extra declares these
dependencies; it uses DINO's source readers, fitted corrections, review artifact
validation, and renderer. Encoder execution and fitting dependencies are only
needed when running DINO's compute commands. On the configured lab host, use the
existing `multi` environment:

```bash
conda run -n multi python -m pip install --no-deps \
  -e /home/chaichontat/squisher/squisher \
  -e ../markers \
  -e ../dino
conda run -n multi python -m pip install -e '.[native]'
DINO_REVIEW_CONFIG=/path/to/markers.json \
DINO_REVIEW_SOURCE="Native fields" DINO_NATIVE_DATASET=Prod3 \
  conda run -n multi dinoapp --server.port 8502
```

### Review targets

The native targets are the `review_targets` list of a marker configuration file.
`DINO_REVIEW_CONFIG` names that file; the app reads it on every render. Without
`DINO_REVIEW_CONFIG`, **Review target** offers only **Crop directory**. With
`DINO_REVIEW_SOURCE="Native fields"`, `DINO_NATIVE_DATASET` selects the initial
target by name and must name a configured target. Each entry names its adapter,
which owns the dataset's file layout below `root` and `review_root`:

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
    "uncalled_color": "#999999"
  },
  "review_targets": [
    {
      "name": "Prod3",
      "adapter": "prod3",
      "root": "/data/prod3",
      "review_root": "/data/prod3/dino-review",
      "cell_prefix": "prod3-realigned-v1",
      "raw_channels": {"561": "pi", "BrdU": "brdu", "EdU": "edu"}
    }
  ]
}
```

| Key | Meaning |
| --- | --- |
| `name` | Name shown in **Review target**; unique in the file. |
| `adapter` | `prod2`, `prod3`, `prod2_2`, or `prod3_2`. |
| `root` | Dataset directory holding the microscopy, segmentation, and surface files the adapter reads. |
| `review_root` | Directory holding the annotation directories, the model selection `cell-review/current.json`, and the default review state. |
| `cell_prefix` | Namespace of the dataset's cell identifiers, `<cell_prefix>\|<label>`. |
| `raw_channels` | Raw image array names mapped to the file's reference and marker keys, one array per key. |
| `embedding_run` | Re-embedding run directory; required by `prod3_2` and `prod2_2`. |

Relative paths resolve against the directory containing the file; absolute paths
are used as written. Each target's `raw_channels` must cover the channel keys of
this file and of the target's annotation directories. `review_targets` is read only
by the reviewer: `thesis-dino` ignores it and never stores it in a data
directory, and the `markers.json` of a crop or annotation directory supplies only
its markers. The [data format](docs/data-format.md#review-targets) lists the
validation rules.

In the sidebar, choose a configured **Review target**, then
choose **Review view → Spatial field**. The sidebar shows the selected classifier
and its cutoff for each marker. The adapters read the reference and markers from the
`markers.json` of the target's default labels directory, and they reject a
classifier, prediction manifest, or inference record whose marker contract differs.

Use **Position**, **Previous**, and **Next** to sample fields; adjust **Field
width (µm)** and **Z (µm)** to inspect the surrounding tissue. Choose **Cell
identities** or **Model uncertainty** in **Right panel view**.

Clicking an unannotated cell with a model call immediately confirms and saves
that call. Clicking an already annotated cell selects it for inspection.
With two markers, four buttons (**EdU only**, **BrdU only**, **Dual positive**,
**Neither** by default) set the selected cell's call; a click saves it immediately
unless it equals the saved call, which writes nothing. With three or more markers,
use the `−` or `+` control of each marker; the call is saved as soon as every marker
is set and differs from the saved call. The badge names the selected cell's state.
**Undo last edit** reverses the most recent saved edit.

### Native save locations

Existing annotations are updated in the file that contains them. New annotations
go to the adapter's default file below, under the target's `review_root`.

| Adapter | Annotation directories | Default labels file |
| --- | --- | --- |
| `prod2` | `barrage-cortex`, `barrage-vz-svz`, `barrage-pixel` | `barrage-pixel/labels.csv` |
| `prod3` | `barrage-cortex-v1`, `barrage-vz-svz-v1` | `barrage-vz-svz-v1/labels.csv` |
| `prod2_2`, `prod3_2` | `barrage-cortex-v1` | `barrage-cortex-v1/labels.csv` |

**Review state directory** defaults to the target's `review_root`.
`DINO_REVIEW_LABEL_ROOT` sets its initial value. This setting controls where
`cell-review/` backups and change journals are written; annotation files remain
at the locations selected by the dataset adapter.

### Corrected images and model review

Prod3 displays BrdU after its saved smooth BrdU/DAPI subtraction and periodic
intensity correction. DAPI uses the recorded 405 nm source; the 561 nm anatomical
image is a separate display channel. Spatial fields and selected-cell crops use
the same pixel correction.

Prod2-2 uses the reviewed ventricular surface, native segmentation, true DAPI,
corrected BrdU, and saved hierarchical probabilities. Its active classifier and
training review artifact are selected by `cell-review/current.json` under its
review directory.

Prod3-2 uses the recovered 405 nm DAPI fusion and postprocessed segmentation under
`/warm/prod3-2-405-local3d-ls-z128-core80`, with the short-window 514 nm EdU fusion
and baseline predictions under the target's `embedding_run`.
Its spatial view uses the reviewed ventricular surface. The transferred baseline
was trained on Prod2 cells: **Training miscalls** scope, **Training**, and **Validation**
are empty for that baseline's Prod3-2 review. Use **All labeled** in a correction
gallery to review saved Prod3-2 annotations against its predictions.

The Prod3-2 barrage at
`/working/eduseg/prod3-2/dino-prod3-2/barrage-cortex-v1` also displays corrected
crops through **Crop directory**. Its `display.json` selects native corrected pixels for
both marker galleries and combined cell review. This requires the native
dependencies and recorded source data; the index, labels, and original PNGs
are preserved.

## RHU2 spatial review

Open a prepared RHU2 barrage through **Crop directory**, then select **RHU2 spatial
fields**. This mode appears when the barrage contains `spatial.json`.

Choose a **ccf_adjusted ROI** by dataset, acquisition ROI, and cortical branch
(`cortex` or `cortex2`). The selector includes only branches with cells in the
prepared cortical cohort, ordered by fewest manual annotations first. **Position**
reproducibly samples a cell within that branch; its centroid determines the field's
Z plane. **Next** and **Previous** stay within the branch. Changing branches resets
Position, while changing field width retains the sampled cell.

Only cortical cohort cells are colored and selectable. Other cells can still be
visible in the raw images. The field has a 20 µm scale bar, and Z is shown as a
native image plane. Selected-cell details show model probabilities and the saved
identity. Clicking an unannotated cell saves its model call as described above.

One **`<label>` vmax percentile** input per marker (**BrdU vmax percentile** and
**EdU vmax percentile** by default) adjusts that marker's upper display limit
using pixels in the current field (default 99.8th percentile).
The same limits apply to the selected-cell crop. These controls change display
contrast only.

All RHU2 edits go to the review directory's `labels.csv`. Images and segmentation
are opened read-only, and saved annotations take precedence over model calls.

### Prepare RHU2 metadata

Preparation requires an existing native barrage conversion with `complete.json`
and a validated RHU2 release. Use the native dependencies in the `multi`
environment on the configured host:

```bash
conda run -n multi python -m dinoapp.rhu2_prepare \
  --review-root /working/chaichontat/rhu2-barrage-review \
  --release /working/chaichontat/ccf_release_rhu2_map2

DINO_REVIEW_SOURCE=Barrage \
DINO_REVIEW_BARRAGE_DIR=/working/chaichontat/rhu2-barrage-review \
  conda run -n multi dinoapp --server.port 8504 --server.headless true
```

Run preparation once: it refuses to overwrite an existing spatial snapshot.
The snapshot is `rhu2-spatial-review-v2` with the default BrdU/EdU contract,
because it holds the frozen release's `brdu_prob` and `edu_prob` columns; the app
requires its probability columns and thresholds to match that contract.
It records coordinates, existing probabilities, and cortical branch membership;
it does not fit a model or copy images. Model calls use a probability cutoff of
0.5 for each marker in the initial snapshot; an activated model supplies its own
cutoffs through `spatial.json`. Branch membership lives in `ccf_branches.parquet`,
separately from model prediction tables.

## Saving, undo, and common problems

Each annotation is saved immediately. Before the first edit to an existing labels
file in a session, the app creates a CSV backup under `cell-review/`. Saved edits
and successful undos are recorded in `changes-<session>.jsonl` there. Each record of a new
journal file holds the cell, its previous and new values as lists in marker order (or
`null`), the label path, `schema` (`dino-review-journal-v1`), `marker_contract`, and
the time. A journal that was started without `schema` and `marker_contract` keeps its
form and continues only under the default contract. Undo history
and loaded queues belong to the current Streamlit session; labels persist on disk.

| What you see | What to do |
| --- | --- |
| An empty correction gallery | Set the per-marker filters to a saved state that has labels (**Any** matches both calls). For a new barrage, start with **Cell review → Cells → Unsaved**. For a transferred native model, try **All labeled**. |
| An edited cell stays in the queue | Click **Refresh matches** or **Refresh cells** when you want to rebuild the queue from current labels. |
| A message that a cell changed on disk | The edit or undo was rejected because another edit changed that cell. Reload the view before trying again. |
| A marker contract mismatch | The directory's `markers.json` or records changed the ordered marker keys after the session opened, or the artifacts of one target disagree. Nothing was written. Reload to open a new session, and restore the intended `markers.json` or the matching artifacts. |
| Missing `index.csv` or crop | Check **Barrage directory** and the image paths in `index.csv`. |
| Native import or missing-file error | Check the native extras, local packages, and recorded dataset paths in the environment running the app. |
| No **Spatial field** view for RHU2 | Open the prepared review directory containing `spatial.json`. |

## Legacy labeler and development

The random-forest-assisted labeler is available as `dinoapp-legacy` after
`python -m pip install -e '.[legacy]'`. If `thesis-dino` is installed, `dino label`
launches the main dinoapp. Barrage generation, training, inference, and periodic
intensity correction belong to [`thesis-dino`](../dino/README.md). The legacy
labeler assigns BrdU and EdU only.

Run the following commands from this `dinoapp/` package directory. For development,
install `.[dev,native,legacy]` and the local native packages, then
run focused tests and lint checks:

```bash
python -m pip install -e ../markers -e '.[dev,native,legacy]'
pytest tests/test_sources.py tests/test_label_store.py
ruff check .
```

## License

[MIT](LICENSE)
