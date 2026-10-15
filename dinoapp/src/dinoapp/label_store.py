"""Atomic per-cell edits shared by barrage annotation interfaces."""
from __future__ import annotations

import fcntl
from pathlib import Path
from typing import Literal

import pandas as pd
from dino_markers import Label, MarkerContract, contract_of, require_same_contract, validate_label


class LabelConflictError(ValueError):
    """The cell was edited after the caller last read it."""


def load_labels(path: Path, *, contract: MarkerContract | None = None) -> pd.DataFrame:
    markers = list((contract_of(path.parent) if contract is None else contract).markers)
    if not path.exists():
        return pd.DataFrame(columns=["cell", *markers])
    frame = pd.read_csv(path, dtype={"cell": str})
    missing = {"cell", *markers} - set(frame.columns)
    if missing:
        raise ValueError(f"Labels CSV {path} is missing {sorted(missing)}")
    if frame["cell"].isna().any() or frame["cell"].duplicated().any():
        raise ValueError(f"Labels CSV {path} contains missing or duplicate cell identities")
    if not frame[markers].isin([0, 1]).all().all():
        raise ValueError(f"Labels CSV {path} must contain binary marker values")
    return frame.astype({marker: int for marker in markers})


def update_label(
    path: Path,
    cell: str,
    value: Label | None,
    *,
    contract: MarkerContract,
    expected: Label | None | Literal["any"] = "any",
) -> Label | None:
    """Read-merge-replace one cell while holding the shared file lock.

    Other cells and extra CSV columns survive each edit. An expected value makes
    undo conditional on no intervening edit to the same cell. None removes a
    newly added annotation when explicitly undone by the user. The directory's
    contract is checked under the lock, so a session opened under another
    contract cannot write.
    """
    if not cell or not isinstance(cell, str):
        raise ValueError("A nonempty cell identity is required")
    if value is not None:
        value = validate_label(contract, value)
    markers = list(contract.markers)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        require_same_contract(
            contract, contract_of(path.parent), expected_from="this review session", actual_from=str(path.parent)
        )
        frame = load_labels(path, contract=contract)
        selected = frame["cell"] == cell
        previous = tuple(int(v) for v in frame.loc[selected, markers].iloc[0]) if selected.any() else None
        if expected != "any" and previous != expected:
            raise LabelConflictError(f"{cell} changed on disk; reload before editing or undoing")
        if value is None:
            frame = frame.loc[~selected].copy()
        elif selected.any():
            frame.loc[selected, markers] = value
        else:
            frame.loc[len(frame)] = {"cell": cell, **dict(zip(markers, value))}
        frame = frame.sort_values("cell", kind="stable").reset_index(drop=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    return previous
