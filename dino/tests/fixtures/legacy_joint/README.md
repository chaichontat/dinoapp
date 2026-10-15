# legacy_joint fixture

Frozen joint-fit artifacts written by the BrdU/EdU `dino.joint_train.fit` at commit `80f2fc5` (branch
`wf4-c`, before the marker-contract changes to `joint_train.py`): one `dino-joint-dataset-v1` bundle and
the two `dino-knn-intensity-v1` bundles it deploys. `dino/tests/test_legacy_joint_bundles.py` loads them,
scores with them, and refits from the recorded inputs; do not regenerate them with newer code.

They were produced in the `multi` conda environment (scikit-learn 1.9.0, torch 2.11.0, pyarrow 24.0.0,
numpy 2.4.6), the environment the tests run in. Fitted cutoffs and probabilities depend on those versions.

`fit` records absolute paths, so `root/` is the verbatim tree under `/tmp/dino-wf4c-legacy-joint` that
the run used. The test session copies `root/` to that absolute location before reading the bundles.

| Path under `root/` | Contents | Produced by |
| --- | --- | --- |
| `config.json` | two datasets (`first`, `second`), seed 7, 12 validation cells, context features `log1p_cell_mean`, `log_cell_over_local_mean`, `log1p_dapi_mean` | hand-made |
| `inputs/<name>/tranche/labels.csv` | 24 cells (`a\|1` to `a\|24`, `b\|1` to `b\|24`), `brdu = i % 2`, `edu = i // 2 % 2` | hand-made |
| `inputs/<name>/embedding-model.joblib` | `dino-zarr-model-v6` bundle holding only the encoder keys and `source_sha256` | hand-made |
| `inputs/<name>/embeddings/shard-000/` | `complete.json` (`dino-zarr-infer-v2`, `stores_embeddings`), one prediction partition and one embedding partition (1024-wide bfloat16, seeded normal noise plus a label signal) | hand-made |
| `inputs/<name>/props.parquet`, `dapi.parquet`, `intensity.json` | seeded uniform regionprops intensities for 24 cells in a line, and the props SHA-256 | hand-made |
| `output/model.joblib` | `dino-joint-dataset-v1` bundle | `fit(config.json, output, workers=1)` |
| `output/deploy/<name>/model.joblib` | `dino-knn-intensity-v1` deployed bundles (embedding + 3 context columns) | same run |
| `output/labels.csv`, `output/deploy/<name>/labels.csv`, `output/labeled_predictions.parquet`, `output/metrics.json` | calls, joint predictions, metrics | same run |

The embedding-model bundles are hand-made (a real one is a full `dino-zarr-model-v6` training bundle);
the joint and deployed bundles are exactly what `fit` wrote.
