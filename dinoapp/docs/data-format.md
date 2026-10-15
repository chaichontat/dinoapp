# Barrage data format

dinoapp reviews a **barrage**: one directory holding cell crop images and an
index that connects each image to its cell. You prepare the index and crops;
the app writes your marker calls back into the same directory. Without a
`markers.json` the markers are BrdU and EdU with the reference channel PI;
`markers.json` declares any other set.

```text
my-barrage/
├── index.csv             required; one row per cell
├── images-brdu/0.png     crops referenced by index.csv (any layout inside the directory)
├── images-edu/0.png
├── images-combined/0.png
├── labels.csv            optional at first; created on the first saved call
├── markers.json          optional; the marker configuration
└── complete.json         optional
```

Open the directory in the sidebar under **Review target → Crop directory**, or set
`DINO_REVIEW_BARRAGE_DIR=/path/to/my-barrage` before launching `dinoapp`.

## index.csv

`index.csv` is a comma-separated table with a header row. Only `cell` and
`rank` are required. Columns not listed here are ignored.

| Column | Required | Content |
| --- | --- | --- |
| `cell` | yes | Cell identifier, read as text. Must be non-empty and unique. Saved calls are matched to this exact string. The caption shows the part after the last `\|`, so `sample:1\|10` appears as "Cell 10". |
| `rank` | yes | Integer review order, unique across the file. Cells are shown in ascending rank; captions show `rank + 1`. |
| `dataset` | no | Group name. Adds a **Dataset** filter to the sidebar. |
| `roi` | no | Region or field name, shown in the crop caption. |
| `<key>_image_path` | per view | Crop shown in the correction gallery of the marker with that key, for example `brdu_image_path` for the **BrdU correction gallery**. One column per marker in `markers.json`. |
| `combined_image_path` | per view | Crop shown in **Cell review**. |
| `image_path` | no | Crop used for any marker whose own path column is absent or empty. |

Each review view needs only its own image column: a barrage with only
`combined_image_path` supports **Cell review**, and one with only
`edu_image_path` supports the **EdU correction gallery**. Correction galleries
appear in the order of the combined crop panels in `markers.json`
(`presentation.crop_panels`, reference channel omitted). A missing path is
reported when that cell's crop is first displayed, not when the barrage opens.

```csv
cell,rank,dataset,roi,brdu_image_path,edu_image_path,combined_image_path
sample:1|10,0,sample,1,images-brdu/0.png,images-edu/0.png,images-combined/0.png
sample:1|20,1,sample,1,images-brdu/1.png,images-edu/1.png,images-combined/1.png
```

## Crop images

Crops are PNG files. Paths in `index.csv` are relative to the barrage directory
and must resolve inside it after following symbolic links; a path that leaves
the directory is rejected.

The app displays each file exactly as stored. It does not adjust contrast,
draw outlines, or resize crops, so render the view reviewers should see when
you generate the images, for example with the target cell outlined. Correction
galleries scale every crop to the same column width, so crops of one size keep
the gallery rows aligned.

## markers.json

Optional. `markers.json` holds the marker configuration of the directory: the
reference channel, the ordered list of binary markers, their display labels and
colors, and the order of the combined crop panels. Without it the directory
uses the packaged default (reference `pi`, markers `brdu` and `edu`).

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
    "negative_color": [0.25, 0.25, 0.25],
    "uncalled_color": "#999999"
  }
}
```

The reference and marker keys, in this order, are the marker contract: the
columns of `labels.csv` and the image column prefixes derive from the marker
keys, and the order of `markers` is the order of the label tuple. Reordering
the markers is a different contract. Labels, colors and panel order are
presentation only.

## labels.csv

dinoapp stores calls in `labels.csv` in the barrage directory and creates the
file with the first saved call. To start from existing calls, provide it in
this format (the default contract, whose markers are `brdu` and `edu`):

```csv
cell,brdu,edu
sample:1|10,1,0
sample:1|20,0,1
```

| Column | Content |
| --- | --- |
| `cell` | Identifier matching `index.csv`. Non-empty and unique. |
| `<key>` | One column per marker in `markers.json`, in marker order. `1` for positive, `0` for negative. The default markers give `brdu` and `edu`. |

The three-marker `markers.json` above gives `cell,edu,brdu,pcna`. A saved call
always sets every marker of the cell in one write, so an unsaved cell is absent
from the file and is never an all-negative row.

The app refuses to open a labels file with missing marker columns, missing or
duplicate cells, or values other than `0` and `1`. Every edit rewrites the file
sorted by `cell`. Extra columns and rows for cells absent from `index.csv` are
kept; new rows leave the extra columns empty.

Every write re-checks the marker contract of the directory under
`labels.csv.lock`. A review session opened under one contract cannot write
after `markers.json` or a dino record in the directory has changed the
contract; the write fails and `labels.csv` is left as it was. Existing
`labels.csv` files are never rewritten to migrate them.

## Files the app writes

The barrage directory must be writable. Besides `labels.csv`, the app creates:

- `labels.csv.lock`, which serializes edits from concurrent sessions, and a
  short-lived `labels.csv.tmp` during each save;
- `cell-review/`, holding a backup of `labels.csv` taken before a session's
  first edit and a `changes-<session>.jsonl` journal of edits and undos.

Each journal record lists `previous` and `value` as marker lists in contract
order, or `null`. A journal file started by this version carries, on every
record, `schema` (`dino-review-journal-v1`) and `marker_contract` (the contract
JSON), placed after `path` and before `time_utc`. A journal file whose first
record has no `schema` is a legacy journal of positional `[brdu, edu]` pairs;
it continues in that form, and only under the default contract. Journal files
are written and never replayed.

## complete.json and run.json

Optional. The app does not use the contents of a hand-made `complete.json`
beyond its bytes: when the file changes, the app treats the barrage as a new
version and rebuilds its review queues, which is useful when you regenerate
crops in place. A hand-made `run.json` or `complete.json` must not carry a
`marker_contract` field, because only dino's own records
(`dino-zarr-source-complete-v*` and `dino-zarr-barrage-v*`) declare one; a
foreign record that carries the field is rejected. A `run.json` must be a JSON
object. A barrage written by `dino-zarr barrage` already holds these records and
a `markers.json`.

## Check a barrage before reviewing

Opening a barrage validates `index.csv`, `markers.json` and its agreement with
the dino records in the directory, and `labels.csv`. This check also opens every
crop that the index references:

```bash
python - /path/to/my-barrage <<'PY'
import sys
from pathlib import Path

from dinoapp.label_store import load_labels
from dinoapp.sources import BarrageSource

source = BarrageSource(Path(sys.argv[1]))
load_labels(source.root / "labels.csv")
for marker in source.crop_markers:
    for cell in source.rows["cell"]:
        source.crop(cell, marker)
    print(f"{marker}: {len(source.rows):,} crops found")
PY
```

## Other review sources

Native dataset targets, **Spatial field** (`spatial.json`), and corrected
barrage display (`display.json`) read microscopy stores, segmentations, and
model outputs prepared by the separate `thesis-dino` pipeline. They are
described in the [README](../README.md) and are not part of the barrage format.

## Review targets

A marker configuration file may carry a top-level `review_targets` list, the
native datasets that **Review target** offers. The reviewer reads it only from
the file named by `DINO_REVIEW_CONFIG`; a barrage's own `markers.json` supplies
only its markers, and `thesis-dino` never writes `review_targets` into a data
directory. Without `DINO_REVIEW_CONFIG` the reviewer offers only **Crop
directory**.

Each entry is an object with exactly these keys:

| Key | Rule |
| --- | --- |
| `name` | Non-empty string, unique in the file. |
| `adapter` | Non-empty string naming a reviewer adapter: `prod2`, `prod3`, `prod2_2`, or `prod3_2`. The app rejects any other name when the target is opened. |
| `root`, `review_root` | Non-empty path strings. |
| `cell_prefix` | String; cell identifiers are `<cell_prefix>\|<label>`. |
| `raw_channels` | Object mapping raw image array names to channel keys, one array for each of the file's reference and marker keys. |
| `embedding_run` | Optional non-empty path string; required by the `prod3_2` and `prod2_2` adapters. |

A relative path resolves against the directory that contains the file; an
absolute path is used as written. A file without `review_targets` has no review
targets. An invalid entry is reported with the file path and the entry's
position, for example `review_targets[0]: unexpected keys ['dataset']`.


## Connected review

The index remains the dataset crop roster. Capturing field cells creates a
session-local queue; it does not add rows to `index.csv` or a model's training
review table. Navigation preserves the queue cursor and records actual field
geometry separately from its sampling cursor. It does not write CSV labels,
backups, or change journals.

Native targets resolve complete cell identifiers against their segmentation
namespace and query saved crop geometry for the requested label. Missing or
duplicate geometry is an error. Cells outside a training gallery can be opened
without inventing a training split or tranche.

An RHU2 `spatial.json` binds the opened directory to its ROI-qualified cortical
cohort and shared `labels.csv`. Captured spatial crops use their ROI's native
largest-area plane and preserve the originating field's display contrast.
Ordinary indexed directories remain crop-only.
