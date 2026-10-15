# Frozen legacy prepared source

Produced once at commit `c1e45f4fda612181d96c98b0e713bfa4a2621f20` by running today's
`dino.zarr_prepare.main()` (`dino-zarr prepare --labels … --pi … --edu … --brdu …
--offset-zyx 0 2 3 --workers 1`) on the inputs under `data/`. The generator ran at
`/tmp/wf0-source/legacy_source`, which is the path prefix recorded in every `root`
field of `prepared/source.json`. `dino/tests/test_characterization_source.py`
copies this directory and replaces that prefix (and the digest derived from it)
to load the artifact from its new location. Never regenerate these files; they
stand for artifacts already on disk.

- `data/demo-dataset/roi-a/masked/labels.zarr`: Zarr v3 root array, `uint32`,
  shape (4, 6, 6), chunks (2, 3, 3); label 7 at `[1:4, 1:5, 2:5]`, label 9 at
  `[0:2, 4:6, 0:2]`.
- `data/demo-dataset/roi-a/masked/volumes.npy`,
  `data/demo-dataset/roi-a/masked/props.parquet`: the masked
  volume references that unmeasured preparation validates against.
- `data/demo-dataset/fused/pi.zarr`, `data/demo-dataset/fused/edu.zarr`,
  `data/demo-dataset/fused/brdu.zarr`: Zarr v2 OME-Zarr groups with
  level-0 array `0`, `uint16`, shape (4, 10, 10), chunks (2, 5, 5); values are
  `arange(400).reshape(4, 10, 10)` plus 1000 (pi), 2000 (edu), or 3000 (brdu).
- `prepared/source.json`: the `dino-zarr-source-v1` manifest
  (`source_sha256` `cea57ee6fa467ec068d7e71668b4452a667514f62ed066a2e17585008ff635a6`).
- `prepared/complete.json`: the `dino-zarr-source-complete-v1` completion record.
- `prepared/cells.parquet`: the prepared cell table (2 cells).

The `prepared/partials/` chunk measurements that preparation also writes were
removed to keep the fixture small; a completed preparation does not read them.
