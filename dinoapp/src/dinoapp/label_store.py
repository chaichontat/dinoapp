"""Atomic per-cell edits shared by barrage annotation interfaces."""
from __future__ import annotations

import fcntl
from pathlib import Path
from typing import Literal

import pandas as pd

Label = tuple[int, int]  # BrdU, EdU


class LabelConflictError(ValueError):
    """The cell was edited after the caller last read it."""


def load_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["cell", "brdu", "edu"])
    frame = pd.read_csv(path, dtype={"cell": str})
    missing = {"cell", "brdu", "edu"} - set(frame.columns)
    if missing:
        raise ValueError(f"Labels CSV {path} is missing {sorted(missing)}")
    if frame["cell"].isna().any() or frame["cell"].duplicated().any():
        raise ValueError(f"Labels CSV {path} contains missing or duplicate cell identities")
    if not frame[["brdu", "edu"]].isin([0, 1]).all().all():
        raise ValueError(f"Labels CSV {path} must contain binary BrdU/EdU values")
    return frame.astype({"brdu": int, "edu": int})


def update_label(
    path: Path,
    cell: str,
    value: Label | None,
    *,
    expected: Label | None | Literal["any"] = "any",
) -> Label | None:
    """Read-merge-replace one cell while holding the shared file lock.

    Other cells and extra CSV columns survive each edit. An expected value makes
    undo conditional on no intervening edit to the same cell. None removes a
    newly added annotation when explicitly undone by the user.
    """
    if not cell or not isinstance(cell, str):
        raise ValueError("A nonempty cell identity is required")
    if value is not None and (len(value) != 2 or any(v not in (0, 1) for v in value)):
        raise ValueError("BrdU/EdU values must be a pair of binary labels")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        frame = load_labels(path)
        selected = frame["cell"] == cell
        previous = tuple(int(v) for v in frame.loc[selected, ["brdu", "edu"]].iloc[0]) if selected.any() else None
        if expected != "any" and previous != expected:
            raise LabelConflictError(f"{cell} changed on disk; reload before editing or undoing")
        if value is None:
            frame = frame.loc[~selected].copy()
        elif selected.any():
            frame.loc[selected, ["brdu", "edu"]] = value
        else:
            frame.loc[len(frame)] = {"cell": cell, "brdu": value[0], "edu": value[1]}
        frame = frame.sort_values("cell", kind="stable").reset_index(drop=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    return previous
