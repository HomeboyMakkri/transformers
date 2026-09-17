"""Reusable PyTorch inputs for Transformer sentiment fine-tuning."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict, cast

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .datasets import LABEL_COLUMN, TEXT_COLUMN, validate_sentiment_dataframe
from .tokenization import DEFAULT_MODEL_NAME

DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_LENGTH = 128
NUM_SENTIMENT_LABELS = 2


class SentimentDatasetItem(TypedDict):
    """One tokenized example before DataLoader batching."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


@dataclass(frozen=True)
class SentimentDataLoaders:
    """Train and validation loaders; the outer test split is excluded."""

    train: DataLoader[SentimentDatasetItem]
    validation: DataLoader[SentimentDatasetItem]


@dataclass(frozen=True)
class SequenceClassifierSetup:
    """A binary sequence classifier placed on its training device."""

    model: PreTrainedModel
    device: torch.device


class SentimentDataset(Dataset[SentimentDatasetItem]):
    """Lazily tokenize aligned binary sentiment examples."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must contain the same number of items")
        if isinstance(max_length, bool) or not isinstance(max_length, int):
            raise TypeError("max_length must be an integer")
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        validated = validate_sentiment_dataframe(
            pd.DataFrame({TEXT_COLUMN: list(texts), LABEL_COLUMN: list(labels)})
        )
        self.texts = tuple(str(text) for text in validated[TEXT_COLUMN])
        self.labels = tuple(int(label) for label in validated[LABEL_COLUMN])
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of aligned text/label examples."""

        return len(self.texts)

    def __getitem__(self, index: int) -> SentimentDatasetItem:
        """Tokenize one example and remove only its temporary batch axis."""

        encoding = self.tokenizer(
            self.texts[index],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = _extract_single_sequence_tensor(
            encoding.get("input_ids"), "input_ids", self.max_length
        )
        attention_mask = _extract_single_sequence_tensor(
            encoding.get("attention_mask"), "attention_mask", self.max_length
        )
        return SentimentDatasetItem(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=torch.tensor(self.labels[index], dtype=torch.long),
        )


def create_sentiment_dataloaders(
    train_dataset: SentimentDataset,
    validation_dataset: SentimentDataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> SentimentDataLoaders:
    """Build shuffled training and ordered validation loaders.

    The API deliberately has no outer-test argument, keeping held-out rows out
    of the Day 5 training and validation path.
    """

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    return SentimentDataLoaders(
        train=DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        validation=DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
        ),
    )


def select_training_device() -> torch.device:
    """Select CUDA when available and otherwise fall back to CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sequence_classifier(
    model_name: str = DEFAULT_MODEL_NAME,
    device: torch.device | None = None,
) -> SequenceClassifierSetup:
    """Load the binary sequence classifier and move it to one device."""

    selected_device = select_training_device() if device is None else device
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_SENTIMENT_LABELS,
    )
    model.to(selected_device)
    return SequenceClassifierSetup(model=model, device=selected_device)


def _extract_single_sequence_tensor(
    value: object,
    name: str,
    max_length: int,
) -> torch.Tensor:
    """Validate one tokenizer tensor shaped ``[1, max_length]``."""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Tokenizer {name} must be a PyTorch tensor")
    tensor = cast(torch.Tensor, value)
    if tensor.shape != (1, max_length):
        raise ValueError(f"Tokenizer {name} must have shape [1, max_length]")
    if tensor.dtype != torch.long:
        raise ValueError(f"Tokenizer {name} must use torch.long dtype")
    return tensor.squeeze(0)
