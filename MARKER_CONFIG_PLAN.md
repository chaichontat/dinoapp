# Marker configuration: multi-agent implementation plan

Goal: replace the hardcoded BrdU/EdU/PI assumptions in `dino/` (thesis-dino) and `dinoapp/` with one shared marker configuration supporting N binary markers and a reference channel, while every existing artifact (prepared sources, barrages, `labels.csv`, feature caches, model bundles, prediction runs, RHU2 spatial snapshots) loads and resumes unchanged.

Settled by the author: both packages share one config; N binary markers; the default config reproduces today's names, with no migration of existing files.

Design review: the Astra consultation in `~/.cache/codex-consult/dinoinfer-marker-config/turn-1.md` changed the identity model and the phase order. Its corrections are folded in below.

## 1. Decisions (author, 2026-10-15)

| # | Decision | Resolution |
| --- | --- | --- |
| D1 | Config format | JSON. |
| D2 | Cell review with N markers | One +/− control per marker plus "Save and next", saving the complete assignment atomically; unsaved stays distinct from all-negative; per-marker saved-state filters replace the joint-identity list. |
| D3 | Combined crop panel order | A display setting in the JSON (`presentation.crop_panels`, default PI \| EdU \| BrdU), independent of the marker order that defines labels and models. |
| D4 | Frozen as BrdU/EdU-only | Legacy H5AD pipeline (`train.py`, `infer.py`, `barrage.py`, `rf_*`, `model_sweep.py`, `roc.py`, `integrate.py`, `runner.py`), `legacy_labeler.py`, `residual_brdu*.py`, `rhu2_*` training/embedding/predict modules, BrdU/DAPI spatial subtraction science. Their calls into shared helpers are still adapted. |
| D5 | Commits | Work happens on the `generalize` branch. The migration is committed first (worktrees contain only committed files), then one signed commit per wave so the next wave's worktrees start from it. |
| D6 | Full test suites | Both package suites run in full at each wave's integration gate. |
| D7 | Out-of-repo caller | Replace `--brdu-input-channel`/`--edu-input-channel` with `--input-channel TARGET=CH` and update `thesis-internal/scripts/dino_train_405.sh`. |

## 2. Target design (the contract every agent codes against)

### 2.1 Ownership

A new workspace member `markers/` (distribution `dino-markers`, module `dino_markers`, Python ≥3.10, no dependencies) owns the config schema, the packaged default, validation, the semantic contract, legacy normalization, name derivation, and call encoding. `dino` and `dinoapp` declare it as a dependency in their `pyproject.toml`; workspace membership alone is not enough. Operations receive an immutable config object as an argument. Do not replace `TARGETS` with mutable global state: one reviewer process opens several datasets.

### 2.2 Config file (`markers.json`)

```json
{
  "version": 1,
  "reference": {"key": "pi", "label": "PI"},
  "markers": [
    {"key": "brdu", "label": "BrdU", "color": "#ff00ff"},
    {"key": "edu",  "label": "EdU",  "color": "#00ff00"}
  ],
  "presentation": {
    "crop_panels": ["pi", "edu", "brdu"],
    "negative_color": "#404040",
    "uncalled_color": "#999999"
  }
}
```

Validation: keys unique, lowercase `[a-z0-9_]+`, no cell delimiter; at least one marker; the reference is not a marker; `crop_panels` is a permutation of reference and markers; colors parse. The marker count is capped by whatever call encoding the spec chooses (Astra: today's `uint8` arrays cannot hold a 2^N "no call" value at N ≥ 8; do not enumerate 2^N states anywhere).

### 2.3 Semantic contract and where it lives

The contract is `{"version": 1, "reference": "pi", "markers": ["brdu", "edu"]}`: ordered keys only. Labels, colors, and panel order are presentation and never invalidate features or models. Readers compare the ordered list exactly, so a same-length reorder is rejected.

| Artifact | Records the contract in | Legacy versions read as the BrdU/EdU contract |
| --- | --- | --- |
| Prepared source | prepare `complete.json` plus `markers.json` beside it; `source.json` unchanged so `source_sha256` is stable | prepare completions without the field |
| Barrage | `run.json`/`complete.json` plus `markers.json` in the barrage dir | `dino-zarr-barrage-v3`; hand-made barrages per `dinoapp/docs/data-format.md` without `markers.json` |
| Feature cache | cache metadata | `dino-zarr-features-v2` |
| Model bundle | bundle field, including the ordered targets used for joint feature assembly | `dino-zarr-model-v6`, `dino-knn-intensity-v1`, `dino-joint-dataset-v1` |
| Prediction run | contract.json, complete.json, partition provenance | `dino-zarr-infer-v2`, `dino-cortex-rescore-v1` |
| Periodic correction | existing `targets` list (already artifact-owned order) | current bundles |
| RHU2 spatial snapshot | `spatial.json` | `rhu2-spatial-review-v1` |
| Annotations | the owning artifact's contract; checked under the existing `labels.csv.lock` on every write | `labels.csv` in any legacy barrage |
| Change journal | a `markers` field on each new record | records without it are positional `[brdu, edu]` |

New artifact versions require the field; a missing field fails instead of defaulting. `--markers PATH` on a command that consumes existing artifacts asserts compatibility and never reinterprets them. Legacy records are normalized in memory, per recognized schema version, before the whole-dictionary equality checks in `zarr_train.py:475` (feature cache), `zarr_barrage.py:487` (barrage contract), and `zarr_infer.py:518` (resume); old bundles are never rewritten, because their file hashes identify them downstream.

### 2.4 Derived names

All persisted names derive from the key, so the default config reproduces today's files exactly: label column `k`, `p_k`, `pred_k`, `k_cutoff`, `k_prob` (legacy/RHU2), `k_image_path`, `k_model`, `k_feature_scaler`, `intensity_k_mean`, barrage stats `k_mean|min|max|median|std`, `display_max[k]`.

### 2.5 Behavior that must be generalized, not renamed

- Uncertainty is `1 / min_t |p_t − cutoff_t|` (infinite at an exact cutoff). `review_contracts.combined_uncertainty` becomes mapping-based over N targets; same formula and validation.
- Joint classifier features (`classifier_features.py:188`) concatenate embeddings then contexts in target order: take that order from the bundle and include it in `validate_feature_contract`.
- Periodic coefficients (`periodic_pixels.py`: `reshape(width, width, 2)`, `("brdu","edu").index`) use the bundle's `targets` order.
- `intensity_context.py:50` maps targets to columns by position; switch to name lookup.
- Joint stratification (`training_split.py`, `joint_train.py:236`, `rhu2_train.py:264`): stratify on the full joint label, fall back to each marker in order, keep dataset-level strata and class-support checks.
- Model inputs: N targets are independent of Cell-DINO's 4–5 input slots per target image. The default keeps `("brdu","edu")` for both classifiers; custom configs must give each target's input channels explicitly.
- `clean_labels.py:31` keeps only `cell,brdu,edu`: generalize to the configured columns.
- Spatial subtraction stays BrdU against true 405 nm DAPI (`spatial_pixels.py:42`). Dispatch reads target and reference from the saved identity; unsupported combinations are rejected, not substituted.
- Field and gallery colors: all-negative and uncalled use explicit presentation colors; positive states blend marker colors with defined clipping. With the default config the result must equal today's palette (`native.py:39-42`) and field composite (`native.py:577-585`).

### 2.6 CLI

- `dino-zarr prepare`: `--pi/--edu/--brdu` become a repeated `--channel KEY=PATH` validated against `--markers` (default: packaged config).
- `dino-zarr train`: `--brdu-input-channel/--edu-input-channel` become a repeated `--input-channel TARGET=CH` (D7).
- `dino-zarr barrage`: `--display-max` accepts the configured keys.

## 3. Execution mechanics

- Base: the `generalize` branch after D5. Each wave's implementers run with `isolation: 'worktree'` and own disjoint files (section 5). They do not commit; they return their worktree path, changed files, and test results.
- Imports inside a worktree: both conda envs import `dino` and `dinoapp` from the main checkout's editable installs. Every worktree agent runs tests as `PYTHONPATH=<wt>/markers/src:<wt>/dino:<wt>/dinoapp/src conda run -n seq pytest …` and first proves the override with `python -c "import dino, dinoapp, dino_markers; print(dino.__file__, dinoapp.__file__, dino_markers.__file__)"`. Use `seq` (Python 3.12, has fishtools); `multi` lacks fishtools.
- Per AGENTS.md: no here-docs with `conda run`; `ruff check --output-format=concise` on modified files; TDD for the logic changes.
- Encoder stubs: train and infer tests already patch the encoder (`dino/tests/test_zarr_train.py`, `dino/tests/test_zarr_infer.py`); the three-marker end-to-end test reuses that pattern.
- Integration (main session, between workflows): apply each worktree's diff to the branch in dependency order, resolve conflicts, run the wave's focused tests, then both full suites (D6), then a signed commit (D5). Report the wave result before launching the next.
- Workflow size: each workflow stays near the 10-agent guideline. Where a wave needs more, it is split into two workflows.

## 4. Workflows

### WF0. Baseline and characterization (≈6 agents, no worktrees)

Purpose: pin today's default behavior so every later wave can prove "unchanged".

- One agent runs both full suites on the committed migration and records pre-existing failures (these are excluded from later gates, not fixed).
- Four agents each add characterization tests in new files only (no edits to existing tests or source), one area each:
  1. Source and prepare: `source_sha256` and manifest for a small fixture; prepare `complete.json` keys.
  2. Barrage: run/complete contract, `display.json`, index columns, rendered combined-crop panel order.
  3. Train and infer: bundle keys, feature-cache metadata and entry keys, prediction columns and order (including `cortex_score`'s interleaved order), resume-contract equality.
  4. dinoapp: `labels.csv` bytes after an edit, journal record shape, call codes, palette, field composite, gallery and button identities, `combined_uncertainty` values including the exact-cutoff infinity.
- One reviewer confirms each test passes on current code and pins observable behavior rather than implementation details.

Output: `tests/test_characterization_*.py` per package. Gate: all pass on the base commit.

### WF1. Contract spec (≈3 agents) → checkpoint

- Two independent spec drafts of `dino_markers` (dataclasses, function signatures, legacy normalization table, error types, call encoding for N markers, color blending) and of every cross-package API change (`combined_uncertainty`, `render_crop` panels, `correct_channels`, `validate_feature_contract`, the generic `marker_crop` replacing `edu_cell_crop`/`brdu_cell_crop`, adapter prediction loading). One draft starts from compatibility, the other from the N-marker reviewer and pipeline.
- One judge merges them into `docs/marker-contract.md`, listing every legacy schema version and its normalization.
- Checkpoint (main session): Astra reviews the merged spec in the existing consultation thread; then the author approves it. The spec is frozen before WF2: later agents code against it and may not change it without a new checkpoint.

### WF2. `dino_markers` package (≈3 agents)

- One implementer, TDD: schema, loader, validation, packaged default, contract comparison, legacy normalization, name helpers, call encode/decode, color blending; `pyproject.toml` for the new member; dependency declarations in `dino/` and `dinoapp/`.
- Two reviewers with distinct lenses (legacy compatibility; N-marker correctness), each finding verified by an adversarial refuter before it counts; one fix pass if needed.

Gate: package tests, characterization tests unchanged, basic `dinoapp` import without `thesis-dino`.

### WF3. Usable path: prepare, barrage, annotation storage (≈8–10 agents)

Four implementers in worktrees (slices A, B, E, F in section 5), each followed in a pipeline by one reviewer and an optional fix agent. Order inside the workflow is free: the slices meet only through the frozen spec.

Gate: characterization tests unchanged; new tests for custom contracts at each artifact; reorder rejection at prepare→barrage→labels.

### WF4. Training and inference (≈6–8 agents)

Slices C and D, each implement → review (two lenses) → fix. C consumes A's completion-record contract and B's barrage contract from the integrated branch.

Gate: characterization tests unchanged (including feature-cache and resume equality on legacy fixtures); a three-marker train→infer run on stub encoders; same-size reorder rejected at train and infer; joint feature order bound to the bundle.

### WF5. Reviewer and frozen-path adaptations (≈8–10 agents)

Slices G (reviewer UI), H (dataset adapters and the shared prediction loader), I (frozen-path call sites and docs), each implement → review → fix. G and H depend on F's storage API and D's `combined_uncertainty`.

Gate: dinoapp characterization unchanged under the default config; Streamlit AppTest runs of cell review and both galleries with a three-marker barrage; stale-session writes rejected after a contract change; basic crop review without `thesis-dino`.

### WF6. Integration review and acceptance (≈8 agents)

- One agent writes the acceptance suite (section 6) against the integrated branch.
- Four reviewers, one lens each: legacy compatibility; N-marker correctness; preserved scientific contracts (spatial subtraction, DAPI features, RHU2); remaining hardcoding (a grep-driven sweep for `brdu|edu|pi` outside frozen modules, each hit classified).
- Each finding gets two adversarial refuters; survivors become fix tasks for a final round.
- One completeness critic: which artifact, command, or view was never exercised under a custom contract?

## 5. Slices and file ownership

| Slice | Wave | Owns | Main changes |
| --- | --- | --- | --- |
| A Source and prepare | WF3 | `dino/dino/zarr_source.py`, `zarr_prepare.py`, their tests | `required_channels` from config; `--channel KEY=PATH`; contract in prepare `complete.json`; `source.json` unchanged |
| B Barrage and rendering | WF3 | `zarr_barrage.py`, `crop_rendering.py`, their tests | stats and `display_max` from config; panels from `crop_panels`; contract in barrage records; `markers.json` written |
| E Pixel corrections | WF3 | `pixel_corrections.py`, `periodic_pixels.py`, `periodic.py`, `spatial_pixels.py`, their tests | periodic order from bundle `targets`; subtraction dispatch from saved identity; reject unsupported |
| F Annotation storage | WF3 | `dinoapp/src/dinoapp/label_store.py`, `sources.py`, `review_session.py`, `dinoapp/docs/data-format.md`, their tests | N-tuple labels; columns from contract; contract check under the lock; journal `markers` field; generic `marker_crop` on `ReviewSource` |
| C Training | WF4 | `zarr_train.py`, `zarr_model.py`, `classifier_features.py`, `training_split.py`, `intensity_context.py`, `joint_train.py`, `clean_labels.py`, their tests | loops over contract; `--input-channel`; joint feature order bound to bundle; N-marker stratification; name-based intensity columns |
| D Inference and provenance | WF4 | `zarr_infer.py`, `call_provenance.py`, `cortex_score.py`, `review_contracts.py`, `artifacts.py`, their tests | per-target arrays; N-ary `combined_uncertainty`; contract in outputs; legacy resume normalization |
| G Reviewer UI | WF5 | `dinoapp/src/dinoapp/app.py`, `native_app.py`, their tests | views from config; per-marker controls with "Save and next"; per-marker filters; labels from config |
| H Dataset adapters | WF5 | `native.py`, `prod3.py`, `prod3_2.py`, `prod2_2.py`, `rhu2.py`, `rhu2_app.py`, `rhu2_prepare.py`, `corrected_barrage.py`, `display_pixels.py`, their tests | one shared prediction loader keeping each adapter's checks; adapters declare raw-channel→marker mapping; palette and composite from config |
| I Frozen paths and docs | WF5 | legacy `train.py`, `infer.py`, `roc.py`, `rhu2_train.py`, `rhu2_embed.py`, `rhu2_predict.py`, `legacy_labeler.py` (call sites only); all READMEs | adapt calls to changed shared helpers under the BrdU/EdU contract; document the config and CLI |

A slice that needs a change in a file it does not own reports it instead of editing; the main session routes it at integration.

## 6. Acceptance criteria

1. Every WF0 characterization test passes unchanged at the end.
2. Existing artifacts on disk load read-only: a Prod3-2 barrage and `labels.csv`, a direct-Zarr model bundle, a completed prediction run (resume reports complete), and an RHU2 `spatial.json` (paths taken from the READMEs; read-only checks only).
3. A three-marker synthetic run completes prepare → barrage → label → train → infer → review on stub encoders.
4. A same-length reordered marker list is rejected at prepare, barrage, train, infer, and label write.
5. A new-format artifact missing its contract fails with an error naming the artifact.
6. A reviewer session holding a stale contract cannot write `labels.csv`.
7. `dinoapp` basic crop review works in an environment without `thesis-dino`.
8. Spatial subtraction with a non-BrdU target or non-DAPI reference is rejected.
9. The hardcoding sweep in WF6 reports no unclassified `brdu|edu|pi` literal outside the frozen modules and the packaged default.

## 7. Risks

- Spec drift between parallel slices: the frozen spec and the "report, do not edit" rule contain it; integration order follows section 5.
- Silent cache invalidation: caught by WF0's feature-cache and resume characterization tests.
- Tests passing against the wrong checkout: prevented by the PYTHONPATH proof step in every worktree agent.
- Large dinoapp UI change (D2): G is the biggest slice; if its review finds structural problems, split it into gallery and cell-review sub-slices in a follow-up workflow.
