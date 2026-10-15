from __future__ import annotations

from collections.abc import Iterator
from math import ceil

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _partitions(
    rows: pd.DataFrame, *, keep: int, seed: int
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    for candidate in (
        rows["brdu"].astype(str) + "_" + rows["edu"].astype(str),
        rows["brdu"].astype(str),
        rows["edu"].astype(str),
    ):
        counts = candidate.value_counts()
        if counts.empty or counts.min() < 2 or min(keep, len(rows) - keep) < len(counts):
            continue
        train, test = train_test_split(
            np.arange(len(rows)), train_size=keep, random_state=seed, stratify=candidate.to_numpy()
        )
        yield (
            rows.iloc[np.sort(train)].reset_index(drop=True),
            rows.iloc[np.sort(test)].reset_index(drop=True),
        )


def split_training_rows(
    rows: pd.DataFrame, *, limit: int, test_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select a supported fixed-seed stratification with enough labels for CV and evaluation."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    cohorts = (
        (kept for kept, _ in _partitions(rows, keep=limit, seed=seed))
        if 0 < limit < len(rows)
        else iter((rows,))
    )
    for cohort in cohorts:
        keep = len(cohort) - ceil(len(cohort) * test_fraction)
        for training, validation in _partitions(cohort, keep=keep, seed=seed):
            if all(
                all(int(training[marker].eq(label).sum()) >= 2 for label in (0, 1))
                and set(validation[marker]) == {0, 1}
                for marker in ("brdu", "edu")
            ):
                return training, validation
    raise ValueError(
        "No supported joint/BrdU/EdU stratification satisfies the requested limit and test fraction: "
        "need both marker classes in validation and at least 2 rows per marker class in training."
    )
