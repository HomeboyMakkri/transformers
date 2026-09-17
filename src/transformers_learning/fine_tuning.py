"""Reusable PyTorch inputs for Transformer sentiment fine-tuning."""

from collections.abc import Sequence
from typing import TypedDict, cast

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .datasets import LABEL_COLUMN, TEXT_COLUMN, validate_sentiment_dataframe

DEFAULT_MAX_LENGTH = 128


class SentimentDatasetItem(TypedDict):
    """One tokenized example before DataLoader batching."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


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
