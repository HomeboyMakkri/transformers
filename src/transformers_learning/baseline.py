"""Frozen Transformer features for the Day 4 classification baseline."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .datasets import LABEL_COLUMN, TEXT_COLUMN, validate_sentiment_dataframe
from .modeling import get_embeddings


@dataclass(frozen=True)
class FrozenEmbeddingDataset:
    """Aligned frozen feature rows and binary sentiment labels."""

    features: npt.NDArray[np.floating[Any]]
    labels: npt.NDArray[np.int64]


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

    if features.ndim != 2:
        raise ValueError("Frozen embeddings must have shape [samples, hidden]")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("Frozen embedding row count must match the label count")
    if features.shape[1] == 0:
        raise ValueError("Frozen embeddings must contain at least one feature")
    if not np.isfinite(features).all():
        raise ValueError("Frozen embeddings must contain only finite values")

    return FrozenEmbeddingDataset(features=features, labels=labels)
