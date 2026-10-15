# Legacy direct-Zarr barrage fixture

Produced once at commit `c1e45f4fda612181d96c98b0e713bfa4a2621f20` (branch `generalize`) by
`dino.zarr_barrage.main()` and `dinoapp.label_store.update_label`, with arguments
`--candidate-cells 6 --n-cells 4 --size 16 --cache-gb 0.001`, in the `seq` conda environment
(matplotlib 3.11.2, pyarrow 22.0.0). PNG and parquet bytes depend on those library versions, so the
tests compare image names and parquet frames, not their bytes. Do not regenerate these files
with newer code: `dino/tests/test_characterization_barrage.py` uses them to prove that newer
code still reads and resumes artifacts written by this commit.

The source manifest records absolute store roots under `/tmp/dino-wf0-legacy-barrage/stores`
(part of `source_sha256`). The test session copies `stores/` there before reading the source.

| Path | Content |
| --- | --- |
| `stores/pi.zarr`, `stores/edu.zarr`, `stores/brdu.zarr` | uint16 ZYX channels, shape (2, 40, 40), zarr v3 |
| `stores/dataset/roi/masked/labels.zarr` | uint32 ZYX labels, shape (2, 32, 32), six 4x4 cells on plane 1, offset (0, 4, 4) in the channels |
| `source/source.json` | `dino-zarr-source-v1` manifest, `source_sha256` 3d04ff64… |
| `source/cells.parquet` | six prepared cells (`dataset:roi|1` to `dataset:roi|6`) |
| `source/display.json` | `dino-zarr-display-v1` cache with `vmax` for `pi`, `edu`, `brdu` |
| `barrage/run.json`, `barrage/complete.json` | identical `dino-zarr-barrage-v3` rendering contract |
| `barrage/source.json` | copy of the source manifest written by the barrage |
| `barrage/candidates.parquet` | six measured candidates with `brdu_*`/`edu_*` statistics |
| `barrage/index.csv` | four selected cells, rank order `roi|4, roi|3, roi|2, roi|1` |
| `barrage/images-v3-16/*.png` | combined PI \| EdU \| BrdU crops (384 x 128) |
| `barrage/labels.csv` | calls written by `update_label`: `roi|2` BrdU+, `roi|4` EdU+ (lock file removed) |
