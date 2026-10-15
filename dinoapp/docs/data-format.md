# Barrage data format

dinoapp reviews a **barrage**: one directory holding cell crop images and an
index that connects each image to its cell. You prepare the index and crops;
the app writes your BrdU and EdU calls back into the same directory.

```text
my-barrage/
├── index.csv             required; one row per cell
├── images-brdu/0.png     crops referenced by index.csv (any layout inside the directory)
├── images-edu/0.png
├── images-combined/0.png
├── labels.csv            optional at first; created on the first saved call
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
| `brdu_image_path` | per view | Crop shown in the **BrdU correction gallery**. |
| `edu_image_path` | per view | Crop shown in the **EdU correction gallery**. |
| `combined_image_path` | per view | Crop shown in **Cell review**. |
| `image_path` | no | Crop used for any marker whose own path column is absent or empty. |

Each review view needs only its own image column: a barrage with only
`combined_image_path` supports **Cell review**, and one with only
`edu_image_path` supports the **EdU correction gallery**. A missing path is
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

## labels.csv

dinoapp stores calls in `labels.csv` in the barrage directory and creates the
file with the first saved call. To start from existing calls, provide it in
this format:

```csv
cell,brdu,edu
sample:1|10,1,0
sample:1|20,0,1
```

| Column | Content |
| --- | --- |
| `cell` | Identifier matching `index.csv`. Non-empty and unique. |
| `brdu` | `1` for BrdU-positive, `0` for BrdU-negative. |
| `edu` | `1` for EdU-positive, `0` for EdU-negative. |

The app refuses to open a labels file with missing columns, missing or duplicate
cells, or values other than `0` and `1`. Every edit rewrites the file sorted by
`cell`. Extra columns and rows for cells absent from `index.csv` are kept; new
rows leave the extra columns empty.

## Files the app writes

The barrage directory must be writable. Besides `labels.csv`, the app creates:

- `labels.csv.lock`, which serializes edits from concurrent sessions, and a
  short-lived `labels.csv.tmp` during each save;
- `cell-review/`, holding a backup of `labels.csv` taken before a session's
  first edit and a `changes-<session>.jsonl` journal of edits and undos.

## complete.json

Optional. The app does not read its contents. When the file changes, the app
treats the barrage as a new version and rebuilds its review queues, which is
useful when you regenerate crops in place.

## Check a barrage before reviewing

Opening a barrage validates `index.csv` and `labels.csv`. This check also opens
every crop that the index references:

```bash
python - /path/to/my-barrage <<'PY'
import sys
from pathlib import Path

from dinoapp.label_store import load_labels
from dinoapp.sources import BarrageSource

source = BarrageSource(Path(sys.argv[1]))
load_labels(source.root / "labels.csv")
columns = {"brdu": "brdu_image_path", "edu": "edu_image_path", "combined": "combined_image_path"}
for marker, column in columns.items():
    if column in source.rows or "image_path" in source.rows:
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
