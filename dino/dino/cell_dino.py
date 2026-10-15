from __future__ import annotations

import importlib.metadata
import json
from typing import Any

DINOV2_INSTALL = 'pip install --no-deps "dinov2 @ git+https://github.com/chaichontat/dinov2@<commit>"'


def installed_dinov2_source() -> str:
    """Name the installed dinov2 by the git URL and commit recorded in its PEP 610 direct_url.json."""
    try:
        distribution = importlib.metadata.distribution("dinov2")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"dinov2 is not installed; install it with {DINOV2_INSTALL}") from exc
    text = distribution.read_text("direct_url.json")
    record = json.loads(text) if text is not None else None
    vcs_info = record.get("vcs_info") if record is not None else None
    if vcs_info is None or vcs_info.get("vcs") != "git":
        raise RuntimeError(
            f"Installed dinov2 is not a git install (direct_url.json: {record}); reinstall it with {DINOV2_INSTALL}"
        )
    return f"git+{record['url']}@{vcs_info['commit_id']}"


def require_dinov2_source(recorded: str) -> None:
    installed = installed_dinov2_source()
    if installed != recorded:
        raise ValueError(f"Installed dinov2 is {installed}, but the model was built with {recorded}")


def build_cell_transform(*, resize_size: int, crop_size: int) -> Any:
    from dinov2.data.cell_dino.transforms import make_classification_eval_cell_transform

    return make_classification_eval_cell_transform(resize_size=resize_size, crop_size=crop_size)
