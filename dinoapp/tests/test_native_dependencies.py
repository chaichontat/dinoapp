import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "dinoapp.native",
        "dinoapp.prod3",
        "dinoapp.prod3_2",
        "dinoapp.prod2_2",
        "dinoapp.display_pixels",
        "dinoapp.corrected_barrage",
        "dinoapp.review_predictions",
    ],
)
def test_native_runtime_imports_without_compute_or_ui(module):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import importlib.abc
import sys
class RuntimeOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'dinov2', 'osqp', 'streamlit', 'squisher_lightsheet'}:
            raise ImportError(f'Unexpected native dependency: {fullname}')
sys.meta_path.insert(0, RuntimeOnly())
importlib.import_module(sys.argv[1])
""",
            module,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_plain_crop_workspace_runs_without_native_or_compute_dependencies():
    script = r"""
import importlib.abc
import os
from pathlib import Path
import sys
import tempfile


class PlainReviewOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'dino', 'torch', 'torchvision', 'dinov2', 'osqp'}:
            raise ImportError(f'Unexpected plain-review dependency: {fullname}')
        if fullname in {'dinoapp.native', 'dinoapp.native_app', 'dinoapp.rhu2',
                        'dinoapp.prod3', 'dinoapp.prod3_2', 'dinoapp.prod2_2'}:
            raise ImportError(f'Unexpected native adapter: {fullname}')


sys.meta_path.insert(0, PlainReviewOnly())
from PIL import Image
from streamlit.testing.v1 import AppTest
from dinoapp import app

with tempfile.TemporaryDirectory(prefix='dinoapp-plain-review-') as scratch:
    root = Path(scratch)
    Image.new('RGB', (8, 8)).save(root / 'crop.png')
    (root / 'index.csv').write_text('cell,rank,image_path\nplain|1,0,crop.png\n')
    os.environ['DINO_REVIEW_SOURCE'] = 'Barrage'
    os.environ['DINO_REVIEW_BARRAGE_DIR'] = str(root)
    review = AppTest.from_file(app.__file__).run(timeout=20)
    assert not review.exception, review.exception
    next(widget for widget in review.selectbox if widget.label == 'Review view').select('Cell review').run()
    assert not review.exception, review.exception
    next(widget for widget in review.button if widget.label == 'BrdU− / EdU−').click().run()
    assert not review.exception, review.exception
    assert 'plain|1,0,0' in (root / 'labels.csv').read_text()
    context, = review.session_state.review_contexts.values()
    assert len(context['history']) == 1
    print('Plain barrage startup, crop assignment, and shared history passed with DINO/native imports blocked.')
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
