from __future__ import annotations

from collections.abc import Iterator, Sequence
from math import ceil

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def joint_label(rows: pd.DataFrame, markers: Sequence[str]) -> pd.Series:
    """The markers' values joined in contract order, for example `1_0`."""
    label = rows[markers[0]].astype(str)
    for marker in markers[1:]:
        label = label + "_" + rows[marker].astype(str)
    return label


def _partitions(
    rows: pd.DataFrame, *, markers: Sequence[str], keep: int, seed: int
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    candidates = [joint_label(rows, markers)]
    if len(markers) > 1:
        candidates.extend(rows[marker].astype(str) for marker in markers)
    for candidate in candidates:
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
    rows: pd.DataFrame, *, markers: Sequence[str], limit: int, test_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select a supported fixed-seed stratification with enough labels for CV and evaluation."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    cohorts = (
        (kept for kept, _ in _partitions(rows, markers=markers, keep=limit, seed=seed))
        if 0 < limit < len(rows)
        else iter((rows,))
    )
    for cohort in cohorts:
        keep = len(cohort) - ceil(len(cohort) * test_fraction)
        for training, validation in _partitions(cohort, markers=markers, keep=keep, seed=seed):
            if all(
                all(int(training[marker].eq(label).sum()) >= 2 for label in (0, 1))
                and set(validation[marker]) == {0, 1}
                for marker in markers
            ):
                return training, validation
    raise ValueError(
        "No supported joint/per-marker stratification satisfies the requested limit and test fraction: "
        "need both classes of every marker in validation and at least 2 rows per marker class in training."
    )


def split_dataset_rows(
    rows: pd.DataFrame,
    *,
    markers: Sequence[str],
    dataset_index: np.ndarray,
    datasets: int,
    test_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sorted training and validation row indices, stratified by dataset and then by marker labels.

    The first stratification is the dataset with the joint label, with the first marker as the lowest
    bit. The fallbacks are the dataset with each single marker, in order. A stratification is used
    only when every stratum has two rows, the strata fit in both splits, and the result leaves every
    marker with both classes in validation, two rows of each class in training, and two rows of each
    class in every dataset.
    """
    dataset_index = np.asarray(dataset_index, dtype=np.int64)
    values = {marker: rows[marker].to_numpy(dtype=np.int64) for marker in markers}
    candidates = [
        dataset_index * 2 ** len(markers)
        + sum(values[marker] * 2**bit for bit, marker in enumerate(markers))
    ]
    if len(markers) > 1:
        candidates.extend(dataset_index * 2 + values[marker] for marker in markers)
    in_every_dataset = all(
        min(np.bincount(values[marker][dataset_index == dataset], minlength=2)) >= 2
        for marker in markers
        for dataset in range(datasets)
    )
    capacity = min(test_size, len(rows) - test_size)
    if in_every_dataset:
        for strata in candidates:
            counts = np.unique(strata, return_counts=True)[1]
            if counts.min() < 2 or len(counts) > capacity:
                continue
            train, validation = train_test_split(
                np.arange(len(rows)), test_size=test_size, random_state=seed, stratify=strata
            )
            if all(
                min(np.bincount(values[marker][train], minlength=2)) >= 2
                and len(np.unique(values[marker][validation])) == 2
                for marker in markers
            ):
                return np.sort(train), np.sort(validation)
    raise ValueError(
        f"No supported dataset/marker stratification for markers {list(markers)} across {datasets} datasets: "
        "need both classes of every marker in validation, and at least 2 rows per marker class in training "
        "and in every dataset."
    )
