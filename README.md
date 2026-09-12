# dinoapp

`dinoapp` is a standalone Streamlit application for reviewing BrdU and EdU
cell calls. It opens indexed barrage datasets, keeps review queues stable while
labels change, writes labels atomically with compare-and-set conflict checks,
and records backups and JSONL change journals.

## Install

Python 3.10 or newer is required on Linux or another POSIX system.

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e '.[dev]'
pytest
ruff check .
```

## Run

Point the app at a barrage directory and launch it:

```bash
export DINO_REVIEW_BARRAGE_DIR=/path/to/barrage
dinoapp --server.port 8502
```

You can also run `python -m dinoapp`. Any extra arguments are forwarded to
Streamlit.

The barrage directory must contain `index.csv` with unique `cell` and `rank`
columns. Crop files are resolved from `brdu_image_path`, `edu_image_path`, and
`combined_image_path`; a legacy `image_path` column is accepted as a fallback.
Labels are read from and written to `labels.csv` without dropping unrelated
rows or extra columns.

## Prod2 native source

Choose **Prod2 native** in the sidebar to load the existing native field and
training-review implementation lazily. Configure its external location with:

```bash
export DINO_NATIVE_REVIEW_DIR=/path/to/native/app
export DINO_REVIEW_LABEL_ROOT=/path/to/native/labels
```

The native directory must provide importable `cell_review.py` and
`cell_review_app.py` modules with the interfaces used by the adapter.

## License

[MIT](LICENSE)
