# Lab datasets and runs

These notes describe the lab's datasets and recorded runs on its analysis host.
The paths exist only there. The package READMEs describe the general workflows.

## Review targets

The lab's review configuration is
`/home/chaichontat/fishtools2/thesis-internal/config/dinoapp/markers.json`. It
defines the targets Prod2, Prod3, Prod2-2, and Prod3-2. Launch native review in the
`multi` environment:

```bash
DINO_REVIEW_CONFIG=/home/chaichontat/fishtools2/thesis-internal/config/dinoapp/markers.json \
DINO_REVIEW_SOURCE="Native fields" DINO_NATIVE_DATASET=Prod3 \
  conda run -n multi dinoapp --server.port 8502
```

Prod3 displays BrdU after its saved smooth BrdU/DAPI subtraction and periodic
intensity correction. DAPI uses the recorded 405 nm source; the 561 nm anatomical
image is a separate display channel.

Prod2-2 uses the reviewed ventricular surface, native segmentation, true DAPI,
corrected BrdU, and saved hierarchical probabilities. Its active classifier and
training review artifact are selected by `cell-review/current.json` under its
review directory.

Prod3-2 uses the recovered 405 nm DAPI fusion and postprocessed segmentation under
`/warm/prod3-2-405-local3d-ls-z128-core80`, with the short-window 514 nm EdU fusion
and baseline predictions under the target's `embedding_run`. Its spatial view uses
the reviewed ventricular surface. The transferred baseline was trained on Prod2
cells and has no training rows in the Prod3-2 review.

The Prod3-2 barrage at `/working/eduseg/prod3-2/dino-prod3-2/barrage-cortex-v1`
opens as a **Crop directory** whose `display.json` selects corrected crops.

## RHU2 review

```bash
conda run -n multi python -m dinoapp.rhu2_prepare \
  --review-root /working/chaichontat/rhu2-barrage-review \
  --release /working/chaichontat/ccf_release_rhu2_map2

DINO_REVIEW_SOURCE=Barrage \
DINO_REVIEW_BARRAGE_DIR=/working/chaichontat/rhu2-barrage-review \
  conda run -n multi dinoapp --server.port 8504 --server.headless true
```

## Smooth subtraction fits

Fits run in the `multi` environment. The Prod3 launcher
`/home/chaichontat/fishtools2/thesis-internal/thesis/output/prod3_brdu_dapi_subtraction/smooth_subtraction.py`
(`benchmark|run`) and `validate_smooth_subtraction.py` beside it call
`dino.spatial_subtraction_workflow` and `dino.spatial_subtraction_validation`
with their existing directory and prefix.

Prod2 fits use `--output-dir /home/chaichontat/fishtools2/thesis-internal/thesis/output/prod2_brdu_dapi_subtraction --prefix
prod2` and need tighter precision and a larger budget than the defaults:

```sh
python -m dino.spatial_subtraction_workflow run \
  --output-dir /home/chaichontat/fishtools2/thesis-internal/thesis/output/prod2_brdu_dapi_subtraction --prefix prod2 \
  --solver-absolute-tolerance 5e-5 --solver-relative-tolerance 1e-7 \
  --solver-max-iterations 180000 --solver-time-limit 600
```

At default precision, the Prod2 3,000-row benchmark was rejected by the 1e-5
relative-gap gate. At the tighter tolerances it passed with status `solved`,
4,075 iterations, 0.7195 s solver time, raw primal infeasibility 1.28786e-11,
relative objective gap 5.57510e-15 and mean-objective gap 2.27374e-13 raw
intensity units.

Prod2 outer fold 1, inner fold 2 at lambda 0.1 exhausted the default iteration
cap. Its 33,161-row training subset passed with a 180,000 iteration/600 s budget
at the same tighter tolerances: status `solved`, 68,125 iterations, 227.3825 s
solver time, 227.4970 s fit wall time, raw primal infeasibility 0.0496028893,
relative gap 6.531607e-8 and mean-objective gap 0.00000262611 raw intensity
units. Benchmark process peak RSS was 416,428 KiB. Prod3 fits use the default
settings.

## Joint classifier

The Prod2/Prod3 joint fit configuration is
`/working/eduseg/prod2-prod3-joint/config.json`. Cortex rescoring reproduces the
prediction contract of `/working/eduseg/20260808-prod2/full-run2/score-cortex.py`.
