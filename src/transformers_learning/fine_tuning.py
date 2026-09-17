"""Reusable PyTorch inputs for Transformer sentiment fine-tuning."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypedDict, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from .datasets import LABEL_COLUMN, TEXT_COLUMN, validate_sentiment_dataframe
from .tokenization import DEFAULT_MODEL_NAME

DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 2e-5
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


@dataclass(frozen=True)
class ValidationEvaluation:
    """Ordered validation labels, predictions, and aggregate metrics."""

    labels: npt.NDArray[np.int64]
    predictions: npt.NDArray[np.int64]
    accuracy: float
    macro_f1: float


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


def create_fine_tuning_optimizer(
    model: PreTrainedModel,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> torch.optim.AdamW:
    """Create AdamW over every sequence-classifier parameter."""

    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    return torch.optim.AdamW(model.parameters(), lr=learning_rate)


def train_epoch(
    model: PreTrainedModel,
    dataloader: DataLoader[SentimentDatasetItem],
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    """Fine-tune the classifier for one epoch and return mean batch loss."""

    model.train()
    total_loss = 0.0
    processed_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = cast(
            SequenceClassifierOutput,
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            ),
        )
        loss = outputs.loss
        if loss is None:
            raise RuntimeError("Sequence classifier output does not contain loss")
        if loss.ndim != 0:
            raise ValueError("Training loss must be a scalar tensor")

        loss_value = float(loss.detach().item())
        if not math.isfinite(loss_value) or loss_value < 0.0:
            raise ValueError("Training loss must be finite and non-negative")

        loss.backward()
        optimizer.step()
        total_loss += loss_value
        processed_batches += 1

    if processed_batches == 0:
        raise ValueError("Training dataloader must contain at least one batch")

    mean_loss = total_loss / processed_batches
    if not math.isfinite(mean_loss) or mean_loss < 0.0:
        raise ValueError("Mean training loss must be finite and non-negative")
    return mean_loss


def evaluate_sequence_classifier(
    model: PreTrainedModel,
    dataloader: DataLoader[SentimentDatasetItem],
    device: torch.device,
) -> ValidationEvaluation:
    """Evaluate ordered validation batches without gradients or updates."""

    model.eval()
    all_predictions: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = cast(
                SequenceClassifierOutput,
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ),
            )
            logits = outputs.logits
            if logits is None:
                raise RuntimeError("Sequence classifier output does not contain logits")
            _validate_validation_batch(logits, labels)
            predictions = torch.argmax(logits, dim=1)
            all_predictions.append(predictions.detach().cpu())
            all_labels.append(labels.detach().cpu())

    if not all_predictions:
        raise ValueError("Validation dataloader must contain at least one batch")

    predictions_array = np.asarray(
        torch.cat(all_predictions).numpy(), dtype=np.int64
    )
    labels_array = np.asarray(torch.cat(all_labels).numpy(), dtype=np.int64)
    if not np.all(np.isin(labels_array, [0, 1])):
        raise ValueError("Validation labels must use only SST-2 classes 0 and 1")

    accuracy = float(accuracy_score(labels_array, predictions_array))
    macro_f1 = float(
        f1_score(
            labels_array,
            predictions_array,
            labels=[0, 1],
            average="macro",
            zero_division=cast(Any, 0),
        )
    )
    if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
        raise ValueError("Validation accuracy must be finite and between 0 and 1")
    if not math.isfinite(macro_f1) or not 0.0 <= macro_f1 <= 1.0:
        raise ValueError("Validation macro F1 must be finite and between 0 and 1")

    return ValidationEvaluation(
        labels=labels_array,
        predictions=predictions_array,
        accuracy=accuracy,
        macro_f1=macro_f1,
    )


def _validate_validation_batch(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    """Validate the model-output boundary before collecting predictions."""

    if logits.ndim != 2 or logits.shape[1] != NUM_SENTIMENT_LABELS:
        raise ValueError("Validation logits must have shape [batch, 2]")
    if labels.ndim != 1:
        raise ValueError("Validation labels must have shape [batch]")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("Validation logits must align with validation labels")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("Validation logits must contain only finite values")


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
