import numpy as np
import pandas as pd
import pytest

from transformers_learning.baseline import (
    FrozenEmbeddingDataset,
    split_frozen_embedding_dataset,
)
from transformers_learning.splitting import (
    SentimentSplitError,
    split_sentiment_row_indices,
)


def make_sentiment_dataframe(rows_per_class: int = 10) -> pd.DataFrame:
    """Build labelled rows whose text exposes the original row position."""

    labels = [0, 1] * rows_per_class
    return pd.DataFrame(
        {
            "text": [f"Review {row_id}" for row_id in range(len(labels))],
            "label": labels,
        }
    )


def test_split_sentiment_rows_is_reproducible_complete_and_disjoint() -> None:
    dataframe = make_sentiment_dataframe()

    first = split_sentiment_row_indices(dataframe)
    second = split_sentiment_row_indices(dataframe)

    assert first.train_indices.shape == (12,)
    assert first.validation_indices.shape == (4,)
    assert first.test_indices.shape == (4,)
    assert np.array_equal(first.train_indices, second.train_indices)
    assert np.array_equal(first.validation_indices, second.validation_indices)
    assert np.array_equal(first.test_indices, second.test_indices)

    train_ids = set(first.train_indices.tolist())
    validation_ids = set(first.validation_indices.tolist())
    test_ids = set(first.test_indices.tolist())
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert train_ids | validation_ids | test_ids == set(range(len(dataframe)))


def test_split_sentiment_rows_preserves_both_classes_in_every_partition() -> None:
    dataframe = make_sentiment_dataframe()
    split = split_sentiment_row_indices(dataframe)

    labels = dataframe["label"].to_numpy()
    assert set(labels[split.train_indices]) == {0, 1}
    assert set(labels[split.validation_indices]) == {0, 1}
    assert set(labels[split.test_indices]) == {0, 1}


def test_day4_and_day5_use_the_same_outer_test_row_identities() -> None:
    dataframe = make_sentiment_dataframe()
    row_ids = np.arange(len(dataframe), dtype=float)
    baseline_dataset = FrozenEmbeddingDataset(
        features=np.column_stack((row_ids, row_ids + 100.0)),
        labels=np.asarray(dataframe["label"].to_numpy(), dtype=np.int64),
    )

    day5_split = split_sentiment_row_indices(dataframe)
    day4_split = split_frozen_embedding_dataset(baseline_dataset)

    assert day4_split.X_test[:, 0].astype(np.int64).tolist() == (
        day5_split.test_indices.tolist()
    )
    assert day4_split.y_test.tolist() == dataframe.iloc[
        day5_split.test_indices
    ]["label"].tolist()


def test_split_sentiment_rows_validates_data_before_partitioning() -> None:
    dataframe = pd.DataFrame(
        {"text": ["Only positive", "Still positive"], "label": [1, 1]}
    )

    with pytest.raises(ValueError, match="both SST-2"):
        split_sentiment_row_indices(dataframe)


def test_split_sentiment_rows_rejects_too_few_rows_for_all_partitions() -> None:
    dataframe = pd.DataFrame(
        {"text": ["Bad", "Good", "Worse", "Better"], "label": [0, 1, 0, 1]}
    )

    with pytest.raises(SentimentSplitError, match="outer split"):
        split_sentiment_row_indices(dataframe)
