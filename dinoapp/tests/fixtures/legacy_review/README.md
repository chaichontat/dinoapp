# Frozen review annotation artifacts

Produced once at commit `c1e45f4fda612181d96c98b0e713bfa4a2621f20` (branch
`generalize`) by an uncommitted generator run in the `multi` environment, and
read by `dinoapp/tests/test_characterization_review.py`. Do not regenerate them
with newer code: they exist to prove that files written by that commit keep
loading and resuming.

The generator built a three-cell direct-Zarr source (32 x 32 plane, cells
`tiny:r1|1`, `tiny:r1|2`, `tiny:r1|3`) and ran today's code on it:

- `barrage/` is the output of `dino-zarr barrage --candidate-cells 3 --n-cells 3
  --size 12` (`index.csv`, `run.json`, `complete.json`, `candidates.parquet`,
  `images-v3-12/`), with `source.json` removed because it records temporary
  absolute paths. The per-marker crops `images-brdu/` and `images-edu/` were
  rendered with `dinoapp.native.render_brdu_crop` and `render_edu_crop` from the
  same crops and the barrage `display_max`, and `index.csv` was rewritten with
  `brdu_image_path`, `edu_image_path`, and `combined_image_path` (equal to
  `image_path`), as `dinoapp/docs/data-format.md` describes for hand-made
  barrages.
- `barrage/labels.csv` and `barrage/cell-review/` come from one review session
  with session id `frozen01`. `update_label` first wrote `tiny:r1|1 = (0, 0)` and
  `tiny:r1|2 = (1, 0)`. `review_session.save_assignment` then assigned
  `tiny:r1|3 = (1, 1)` (action `assign`), toggled EdU on `tiny:r1|1`
  (`gallery_toggle_edu`) and BrdU on `tiny:r1|2` (`gallery_toggle_brdu`), and
  `review_session.undo_edit` undid the last edit. `barrage-frozen01.csv` is the
  backup taken before the first edit; `changes-frozen01.jsonl` is the journal.
  Its `path`, `source_id`, and `time_utc` fields hold the generator's temporary
  paths and run time.
- `barrage_crops.npz` holds the `CellCrop` arrays (`<rank>_labels`, `_mask`,
  `_neighbor_mask`, `_pi`, `_edu`, `_brdu`) that `dino.crop_rendering.read_crop`
  returned for each barrage row; rendering them reproduces `images-brdu/` and
  `images-edu/`.
- `labels_extras.csv` is a hand-written labels file with extra columns (`round`,
  `note`) and rows absent from the index. `labels_extras_edited.csv` is that file
  after `AnnotationFiles.assign` added `tiny:r1|1 = (1, 1)` and changed
  `tiny:r1|2` from `(1, 0)` to `(0, 1)`.
