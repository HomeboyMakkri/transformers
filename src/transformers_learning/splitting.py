"""Shared row partitions for baseline evaluation and Transformer fine-tuning."""

from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import train_test_split

from .datasets import LABEL_COLUMN, validate_sentiment_dataframe

OUTER_TEST_SIZE = 0.2
INNER_VALIDATION_SIZE = 0.2
SPLIT_RANDOM_STATE = 42


@dataclass(frozen=True)
class OuterSentimentSplit:
    """Positional row identities for the shared train/test boundary."""

    train_indices: npt.NDArray[np.int64]
    test_indices: npt.NDArray[np.int64]


@dataclass(frozen=True)
class SentimentRowSplit:
    """Disjoint positional row identities for Day 5 data partitions."""

    train_indices: npt.NDArray[np.int64]
    validation_indices: npt.NDArray[np.int64]
    test_indices: npt.NDArray[np.int64]


class SentimentSplitError(ValueError):
    """Raised when valid sentiment rows are too few for a stratified split."""


def split_outer_sentiment_indices(
    labels: npt.NDArray[np.int64],
) -> OuterSentimentSplit:
    """Create the shared stratified 80/20 split from positional row IDs.

    Both the frozen-feature baseline and fine-tuning index their own data with
    these IDs, which guarantees that their held-out examples are identical.
    """

    _validate_binary_labels(labels)
    row_indices = np.arange(labels.shape[0], dtype=np.int64)
    try:
        train_indices, test_indices = train_test_split(
            row_indices,
            test_size=OUTER_TEST_SIZE,
            random_state=SPLIT_RANDOM_STATE,
            stratify=labels,
        )
    except ValueError as error:
        raise SentimentSplitError(
            "Sentiment data is too small for the stratified outer split"
        ) from error

    return OuterSentimentSplit(
        train_indices=np.asarray(train_indices, dtype=np.int64),
        test_indices=np.asarray(test_indices, dtype=np.int64),
    )


def split_sentiment_row_indices(dataframe: pd.DataFrame) -> SentimentRowSplit:
    """Create Day 5 train/validation/test positional row partitions.

    The outer test identities come from the same helper used by Day 4. Only
    the outer training identities are split again for validation.
    """

    validated = validate_sentiment_dataframe(dataframe)
    labels = np.asarray(validated[LABEL_COLUMN].to_numpy(), dtype=np.int64)
    outer_split = split_outer_sentiment_indices(labels)

    outer_train_labels = labels[outer_split.train_indices]
    try:
        train_indices, validation_indices = train_test_split(
            outer_split.train_indices,
            test_size=INNER_VALIDATION_SIZE,
            random_state=SPLIT_RANDOM_STATE,
            stratify=outer_train_labels,
        )
    except ValueError as error:
        raise SentimentSplitError(
            "Sentiment data is too small for the stratified validation split"
        ) from error

    split = SentimentRowSplit(
        train_indices=np.asarray(train_indices, dtype=np.int64),
        validation_indices=np.asarray(validation_indices, dtype=np.int64),
        test_indices=outer_split.test_indices,
    )
    _validate_complete_partition(split, labels)
    return split


def _validate_binary_labels(labels: npt.NDArray[np.int64]) -> None:
    """Validate labels accepted by the shared split boundary."""

    if labels.ndim != 1:
        raise ValueError("Sentiment labels must have shape [samples]")
    if not np.array_equal(np.unique(labels), np.array([0, 1])):
        raise ValueError("Sentiment labels must contain both SST-2 classes 0 and 1")


def _validate_complete_partition(
    split: SentimentRowSplit,
    labels: npt.NDArray[np.int64],
) -> None:
    """Check coverage, separation, and class preservation after both splits."""

    partitions = (
        split.train_indices,
        split.validation_indices,
        split.test_indices,
    )
    combined = np.concatenate(partitions)
    expected = np.arange(labels.shape[0], dtype=np.int64)
    if combined.shape[0] != expected.shape[0] or not np.array_equal(
        np.sort(combined), expected
    ):
        raise SentimentSplitError(
            "Sentiment partitions must cover every source row exactly once"
        )
    for indices in partitions:
        partition_labels = cast(npt.NDArray[np.int64], labels[indices])
        if not np.array_equal(np.unique(partition_labels), np.array([0, 1])):
            raise SentimentSplitError(
                "Every sentiment partition must contain both SST-2 classes"
            )
