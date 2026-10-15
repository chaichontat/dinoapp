# legacy_model fixture

Frozen direct-Zarr training and inference artifacts written by the BrdU/EdU code at commit
`c1e45f4fda612181d96c98b0e713bfa4a2621f20` (branch `generalize`), with the Cell-DINO encoder
replaced by the `StubEncoder` defined in `dino/tests/test_characterization_train_infer.py` and
the transform replaced by identity. `dino/tests/test_characterization_train_infer.py` loads,
validates, and resumes from these files; do not regenerate them with newer code.

They were written in the `multi` conda environment (scikit-learn 1.9.0, torch 2.11.0, pyarrow
24.0.0, zarr 3.2.1, joblib 1.5.3, numpy 2.4.6), which is the environment the tests run in. The
trained cutoffs, prediction probabilities, and `cells.parquet` bytes depend on these versions.

Every path inside the artifacts is relative to this directory, which the tests copy to a
temporary location and use as the working directory. The only absolute path recorded is the
bundle's informational `training_barrage_dirs` entry.

| Path | Contents | Produced by |
| --- | --- | --- |
| `zarr/labels.zarr` | 1×12×48 uint32 masked labels, 16 cells of 2×2 pixels | zarr 3.2.1 |
| `zarr/{pi,edu,brdu}.zarr` | 1×14×52 uint16 fused channels, label offset (0, 1, 2) | zarr 3.2.1 |
| `source.json` | `dino-zarr-source-v1` manifest from `build_source_manifest(dataset="ds", roi="roi")`, with each `root` rewritten to its relative path and `source_sha256` recomputed by `manifest_digest` | today's code plus that rewrite |
| `cells.parquet` | prepared cell table (`cell, dataset, roi, label, plane_z, centroid_y, centroid_x`) in a shuffled order | hand-made |
| `index.csv`, `labels.csv` | barrage index and BrdU/EdU labels (`brdu = [0,0,1,1]*4`, `edu = [0,1,0,1]*4`) | hand-made |
| `encoder/` | stand-in DINOv2 checkout (`config.json`) and weights file whose hashes the bundle records | hand-made |
| `model/features.joblib` | `dino-zarr-features-v2` feature cache, 32 entries | `dino-zarr train` with default input channels |
| `model/model.joblib` | `dino-zarr-model-v6` bundle | same `dino-zarr train` run |
| `inference/shard-000/` | completed `dino-zarr-infer-v2` run with `--store-embeddings --partition-cells 8`: `contract.json`, `complete.json`, two prediction parquets, two embedding `.pt` files | `dino-zarr infer` |

Training arguments: `--barrage-dir . --out-dir model --dinov2-repo encoder --weights
encoder/weights.pth --crop-size 4 --resize-size 4 --crop-eval-size 4 --batch-size 4
--test-fraction 0.25 --cache-gb 0.001 --prefetch-workers 1 --crop-workers 1`.

Inference arguments: `--source-dir . --model-dir model --out-dir inference/shard-000
--store-embeddings --batch-size 4 --partition-cells 8 --cache-gb 0.001 --prefetch-workers 1
--crop-workers 1`.
