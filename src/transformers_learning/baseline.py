"""Frozen Transformer features for the Day 4 classification baseline."""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .datasets import LABEL_COLUMN, TEXT_COLUMN, validate_sentiment_dataframe
from .modeling import get_embeddings

TEST_SIZE = 0.2
RANDOM_STATE = 42


@dataclass(frozen=True)
class FrozenEmbeddingDataset:
    """Aligned frozen feature rows and binary sentiment labels."""

    features: npt.NDArray[np.floating[Any]]
    labels: npt.NDArray[np.int64]


@dataclass(frozen=True)
class FrozenEmbeddingSplit:
    """One reproducible train/test partition of frozen feature rows."""

    X_train: npt.NDArray[np.floating[Any]]
    X_test: npt.NDArray[np.floating[Any]]
    y_train: npt.NDArray[np.int64]
    y_test: npt.NDArray[np.int64]


def prepare_frozen_embedding_dataset(
    dataframe: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    batch_size: int = 32,
) -> FrozenEmbeddingDataset:
    """Extract one frozen first-token feature vector for each labelled text.

    The model is delegated to ``get_embeddings``, which sets evaluation mode and
    disables gradients. No classifier is fitted here, so this function does not
    inspect labels while creating feature values.
    """

    validated = validate_sentiment_dataframe(dataframe)
    texts = tuple(str(text) for text in validated[TEXT_COLUMN])
    labels = np.asarray(validated[LABEL_COLUMN].to_numpy(), dtype=np.int64)
    features = get_embeddings(texts, tokenizer, model, batch_size=batch_size)

    dataset = FrozenEmbeddingDataset(features=features, labels=labels)
    _validate_frozen_embedding_dataset(dataset)
    return dataset


def split_frozen_embedding_dataset(
    dataset: FrozenEmbeddingDataset,
) -> FrozenEmbeddingSplit:
    """Create the project's sole stratified 80/20 held-out split.

    Feature extraction is frozen and label-independent, but the classifier must
    later fit only on ``X_train, y_train``. ``X_test, y_test`` are reserved for
    the final Day 4 evaluation and must not guide model choices.
    """

    _validate_frozen_embedding_dataset(dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.features,
        dataset.labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=dataset.labels,
    )
    return FrozenEmbeddingSplit(
        X_train=cast(npt.NDArray[np.floating[Any]], X_train),
        X_test=cast(npt.NDArray[np.floating[Any]], X_test),
        y_train=cast(npt.NDArray[np.int64], y_train),
        y_test=cast(npt.NDArray[np.int64], y_test),
    )


def _validate_frozen_embedding_dataset(dataset: FrozenEmbeddingDataset) -> None:
    """Check public dataclass inputs before they reach scikit-learn."""

    if dataset.features.ndim != 2:
        raise ValueError("Frozen embeddings must have shape [samples, hidden]")
    if dataset.labels.ndim != 1:
        raise ValueError("Frozen labels must have shape [samples]")
    if dataset.features.shape[0] != dataset.labels.shape[0]:
        raise ValueError("Frozen embedding row count must match the label count")
    if dataset.features.shape[1] == 0:
        raise ValueError("Frozen embeddings must contain at least one feature")
    if not np.isfinite(dataset.features).all():
        raise ValueError("Frozen embeddings must contain only finite values")
    if not np.array_equal(np.unique(dataset.labels), np.array([0, 1])):
        raise ValueError("Frozen labels must contain both SST-2 classes 0 and 1")
