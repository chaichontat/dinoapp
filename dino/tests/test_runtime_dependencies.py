"""Public reading and activation APIs do not require UI or compute packages."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", [
    "dino.artifacts",
    "dino.zarr_source",
    "dino.spatial_pixels",
    "dino.classifier_features",
    "dino.review_activation",
    "dino.review_contracts",
    "dino.crop_rendering",
    "dino.pixel_corrections",
    "dino.spatial_subtraction_validation",
])
def test_runtime_imports_without_compute_or_reviewer(module):
    result = subprocess.run(
        [sys.executable, "-c", """
import importlib
import importlib.abc
import sys

class RuntimeOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split('.')[0] in {'torch', 'torchvision', 'dinov2', 'osqp', 'streamlit', 'dinoapp'}
                or fullname == 'dino.spatial_subtraction_workflow'):
            raise ImportError(f'Unexpected runtime dependency: {fullname}')

sys.meta_path.insert(0, RuntimeOnly())
importlib.import_module(sys.argv[1])
""", module],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
