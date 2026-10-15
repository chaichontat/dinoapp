from __future__ import annotations

from pathlib import Path
from typing import Any


def require_dinov2_checkout(repo: Path) -> None:
    """Require installed transforms and local hub entrypoints to share a checkout."""
    import dinov2

    installed = Path(dinov2.__file__).resolve()
    expected = (repo / "dinov2" / "__init__.py").resolve()
    if installed != expected:
        raise RuntimeError(
            f"Installed dinov2 is {installed}, but the configured checkout is {repo.resolve()}. "
            f"Install that checkout with python -m pip install --no-deps -e {repo.resolve()}"
        )


def build_cell_transform(*, repo: Path, resize_size: int, crop_size: int) -> Any:
    require_dinov2_checkout(repo)
    from dinov2.data.cell_dino.transforms import make_classification_eval_cell_transform

    return make_classification_eval_cell_transform(resize_size=resize_size, crop_size=crop_size)
