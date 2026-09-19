"""Shared Day 6 data and artifact boundaries for model comparison."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .baseline import (
    prepare_frozen_embedding_dataset,
    train_logistic_regression_on_frozen_embeddings,
)
from .datasets import LABEL_COLUMN, TEXT_COLUMN, validate_sentiment_dataframe
from .fine_tuning import DEFAULT_FINE_TUNED_MODEL_DIRECTORY, NUM_SENTIMENT_LABELS
from .splitting import OuterSentimentSplit, split_outer_sentiment_indices

_MODEL_CONFIG_FILENAME = "config.json"
_TOKENIZER_CONFIG_FILENAME = "tokenizer_config.json"
_MODEL_WEIGHT_FILENAMES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


class FineTunedArtifactError(ValueError):
    """Raised when a local Day 5 artifact cannot be used for Day 6."""


@dataclass(frozen=True)
class ComparisonDataset:
    """The shared ordered outer partitions used by both Day 6 model paths."""

    outer_train: pd.DataFrame
    outer_test: pd.DataFrame
    split: OuterSentimentSplit


@dataclass(frozen=True)
class Day6ArtifactInputs:
    """Local inputs for Day 6; the frozen baseline is recreated, never loaded."""

    fine_tuned_model_directory: Path


@dataclass(frozen=True)
class FrozenBaselineSetup:
    """A freshly fitted baseline and the frozen Transformer used to create it."""

    classifier: LogisticRegression
    tokenizer: PreTrainedTokenizerBase
    encoder: PreTrainedModel


@dataclass(frozen=True)
class FineTunedInferenceSetup:
    """A local epoch-3 sequence classifier ready for no-gradient inference."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device


def prepare_comparison_dataset(dataframe: pd.DataFrame) -> ComparisonDataset:
    """Select the Day 4/5 outer partitions from validated source-row positions.

    Positional indices, rather than dataframe index labels, identify source rows.
    That makes the shared test order reproducible even when source index labels
    are non-consecutive or duplicated.
    """

    validated = validate_sentiment_dataframe(dataframe)
    labels = np.asarray(validated[LABEL_COLUMN].to_numpy(), dtype=np.int64)
    split = split_outer_sentiment_indices(labels)
    comparison = ComparisonDataset(
        outer_train=validated.iloc[split.train_indices].copy(deep=True),
        outer_test=validated.iloc[split.test_indices].copy(deep=True),
        split=split,
    )
    _validate_comparison_dataset(comparison, labels)
    return comparison


def get_day6_artifact_inputs(
    fine_tuned_model_directory: Path = DEFAULT_FINE_TUNED_MODEL_DIRECTORY,
) -> Day6ArtifactInputs:
    """Validate and name the only persisted Day 6 input artifact.

    Day 6 deliberately has no path for a serialized baseline classifier or a
    fitted text vectorizer: its frozen-embedding baseline is recreated from
    outer-training rows in D6-02.
    """

    validated_directory = validate_fine_tuned_model_artifact(
        fine_tuned_model_directory
    )
    return Day6ArtifactInputs(fine_tuned_model_directory=validated_directory)


def recreate_frozen_baseline(
    comparison: ComparisonDataset,
    tokenizer: PreTrainedTokenizerBase,
    encoder: PreTrainedModel,
    batch_size: int = 32,
) -> FrozenBaselineSetup:
    """Fit the fixed Day 4 baseline using only Day 6 outer-training rows."""

    training_dataset = prepare_frozen_embedding_dataset(
        comparison.outer_train,
        tokenizer,
        encoder,
        batch_size=batch_size,
    )
    classifier = train_logistic_regression_on_frozen_embeddings(training_dataset)
    return FrozenBaselineSetup(
        classifier=classifier,
        tokenizer=tokenizer,
        encoder=encoder,
    )


def select_inference_device() -> torch.device:
    """Select CUDA when available and otherwise use CPU for Day 6 inference."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_fine_tuned_inference(
    artifact_inputs: Day6ArtifactInputs,
    device: torch.device | None = None,
) -> FineTunedInferenceSetup:
    """Reload the saved local binary classifier without any optimization step."""

    directory = validate_fine_tuned_model_artifact(
        artifact_inputs.fine_tuned_model_directory
    )
    selected_device = select_inference_device() if device is None else device
    source = str(directory)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        local_files_only=True,
    )
    _validate_loaded_binary_head(model)
    model.to(selected_device)
    model.eval()
    return FineTunedInferenceSetup(
        model=model,
        tokenizer=tokenizer,
        device=selected_device,
    )


def validate_fine_tuned_model_artifact(directory: Path) -> Path:
    """Require a local, binary Day 5 model and tokenizer artifact directory."""

    if not directory.is_dir():
        raise FileNotFoundError(
            f"Fine-tuned model directory does not exist: {directory}"
        )

    config = _read_json_object(directory / _MODEL_CONFIG_FILENAME, "model config")
    _validate_binary_head(config)
    _read_json_object(directory / _TOKENIZER_CONFIG_FILENAME, "tokenizer config")

    if not any((directory / filename).is_file() for filename in _MODEL_WEIGHT_FILENAMES):
        raise FileNotFoundError(
            "Fine-tuned model artifact is missing model weights "
            f"in {directory}"
        )
    return directory


def _validate_comparison_dataset(
    comparison: ComparisonDataset,
    labels: npt.NDArray[np.int64],
) -> None:
    """Reject incomplete, overlapping, or reordered outer partitions."""

    train_indices = comparison.split.train_indices
    test_indices = comparison.split.test_indices
    expected_indices = np.arange(labels.shape[0], dtype=np.int64)
    selected_indices = np.concatenate((train_indices, test_indices))
    if selected_indices.shape[0] != expected_indices.shape[0] or not np.array_equal(
        np.sort(selected_indices), expected_indices
    ):
        raise ValueError("Comparison partitions must cover every source row once")

    expected_train_labels = labels[train_indices]
    expected_test_labels = labels[test_indices]
    actual_train_labels = np.asarray(
        comparison.outer_train[LABEL_COLUMN].to_numpy(), dtype=np.int64
    )
    actual_test_labels = np.asarray(
        comparison.outer_test[LABEL_COLUMN].to_numpy(), dtype=np.int64
    )
    if not np.array_equal(actual_train_labels, expected_train_labels):
        raise ValueError("Comparison outer-train rows must preserve split order")
    if not np.array_equal(actual_test_labels, expected_test_labels):
        raise ValueError("Comparison outer-test rows must preserve split order")
    if list(comparison.outer_train.columns) != [TEXT_COLUMN, LABEL_COLUMN] or list(
        comparison.outer_test.columns
    ) != [TEXT_COLUMN, LABEL_COLUMN]:
        raise ValueError("Comparison partitions must use the text,label contract")


def _read_json_object(path: Path, artifact_name: str) -> dict[str, Any]:
    """Read one required artifact metadata file as a JSON object."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Fine-tuned model artifact is missing {artifact_name}: {path}"
        )
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FineTunedArtifactError(
            f"Fine-tuned model {artifact_name} is not valid JSON: {path}"
        ) from error
    if not isinstance(loaded, dict):
        raise FineTunedArtifactError(
            f"Fine-tuned model {artifact_name} must be a JSON object: {path}"
        )
    return loaded


def _validate_binary_head(config: dict[str, Any]) -> None:
    """Confirm that local config metadata declares the SST-2 binary head."""

    num_labels = config.get("num_labels")
    id2label = config.get("id2label")
    if num_labels is not None:
        if isinstance(num_labels, bool) or not isinstance(num_labels, int):
            raise FineTunedArtifactError("Fine-tuned model num_labels must be an integer")
        if num_labels != NUM_SENTIMENT_LABELS:
            raise FineTunedArtifactError(
                "Fine-tuned model must have a binary SST-2 classification head"
            )
        return
    if not isinstance(id2label, dict) or set(id2label) != {"0", "1"}:
        raise FineTunedArtifactError(
            "Fine-tuned model config must declare binary labels 0 and 1"
        )


def _validate_loaded_binary_head(model: PreTrainedModel) -> None:
    """Reject an artifact whose loaded classifier head is not binary."""

    num_labels = getattr(model.config, "num_labels", None)
    if isinstance(num_labels, bool) or num_labels != NUM_SENTIMENT_LABELS:
        raise FineTunedArtifactError(
            "Loaded fine-tuned model must have a binary SST-2 classification head"
        )
