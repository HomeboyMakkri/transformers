"""Frozen Transformer features for the Day 4 classification baseline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .datasets import (
    LABEL_COLUMN,
    SST2_LABEL_MAP,
    SST2_METADATA,
    TEXT_COLUMN,
    validate_sentiment_dataframe,
)
from .modeling import ProgressCallback, get_embeddings
from .tokenization import DEFAULT_MODEL_NAME

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


@dataclass(frozen=True)
class BaselineEvaluation:
    """Held-out predictions and metrics for the frozen-embedding baseline."""

    predictions: npt.NDArray[np.int64]
    classification_report: str
    macro_f1: float


def prepare_frozen_embedding_dataset(
    dataframe: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    batch_size: int = 32,
    progress_callback: ProgressCallback | None = None,
) -> FrozenEmbeddingDataset:
    """Extract one frozen first-token feature vector for each labelled text.

    The model is delegated to ``get_embeddings``, which sets evaluation mode and
    disables gradients. No classifier is fitted here, so this function does not
    inspect labels while creating feature values.
    """

    validated = validate_sentiment_dataframe(dataframe)
    texts = tuple(str(text) for text in validated[TEXT_COLUMN])
    labels = np.asarray(validated[LABEL_COLUMN].to_numpy(), dtype=np.int64)
    features = get_embeddings(
        texts,
        tokenizer,
        model,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )

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


def train_logistic_regression(
    split: FrozenEmbeddingSplit,
) -> LogisticRegression:
    """Fit the Day 4 baseline on frozen training embeddings only.

    ``X_test`` and ``y_test`` intentionally do not appear in this function's
    training path. They remain untouched until the held-out evaluation step.
    """

    _validate_training_partition(split.X_train, split.y_train)
    classifier = LogisticRegression(max_iter=1000, n_jobs=-1)
    classifier.fit(split.X_train, split.y_train)
    return classifier


def evaluate_logistic_regression(
    classifier: LogisticRegression,
    split: FrozenEmbeddingSplit,
) -> BaselineEvaluation:
    """Evaluate a fitted baseline once on the reserved held-out partition."""

    _validate_test_partition(split.X_test, split.y_test)
    predictions = np.asarray(classifier.predict(split.X_test), dtype=np.int64)
    if predictions.ndim != 1 or predictions.shape[0] != split.y_test.shape[0]:
        raise ValueError("Baseline predictions must align with held-out labels")
    if not np.all(np.isin(predictions, [0, 1])):
        raise ValueError("Baseline predictions must use only SST-2 labels 0 and 1")

    report = cast(
        str,
        classification_report(
            split.y_test,
            predictions,
            labels=[0, 1],
            target_names=[SST2_LABEL_MAP[0], SST2_LABEL_MAP[1]],
            zero_division=cast(Any, 0),
        ),
    )
    macro_f1 = float(f1_score(split.y_test, predictions, average="macro"))
    if not np.isfinite(macro_f1) or not 0.0 <= macro_f1 <= 1.0:
        raise ValueError("Held-out macro F1 must be finite and between 0 and 1")
    return BaselineEvaluation(
        predictions=predictions,
        classification_report=report,
        macro_f1=macro_f1,
    )


def save_baseline_results(
    evaluation: BaselineEvaluation,
    path: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    dataset_identifier: str = SST2_METADATA.identifier,
) -> None:
    """Save macro F1 and minimal reproducibility context outside version control."""

    if not np.isfinite(evaluation.macro_f1) or not 0.0 <= evaluation.macro_f1 <= 1.0:
        raise ValueError("Held-out macro F1 must be finite and between 0 and 1")

    path.parent.mkdir(parents=True, exist_ok=True)
    label_mapping = ", ".join(
        f"{label}={name}" for label, name in SST2_LABEL_MAP.items()
    )
    path.write_text(
        "\n".join(
            (
                f"macro_f1: {evaluation.macro_f1:.6f}",
                f"model_name: {model_name}",
                f"dataset_identifier: {dataset_identifier}",
                f"label_mapping: {label_mapping}",
                f"test_size: {TEST_SIZE}",
                f"random_state: {RANDOM_STATE}",
                "",
            )
        ),
        encoding="utf-8",
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


def _validate_training_partition(
    features: npt.NDArray[np.floating[Any]],
    labels: npt.NDArray[np.int64],
) -> None:
    """Validate only the data that will reach the classifier's ``fit`` call."""

    if features.ndim != 2:
        raise ValueError("Training embeddings must have shape [samples, hidden]")
    if labels.ndim != 1:
        raise ValueError("Training labels must have shape [samples]")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("Training embedding row count must match the label count")
    if features.shape[1] == 0:
        raise ValueError("Training embeddings must contain at least one feature")
    if not np.isfinite(features).all():
        raise ValueError("Training embeddings must contain only finite values")
    if not np.array_equal(np.unique(labels), np.array([0, 1])):
        raise ValueError("Training labels must contain both SST-2 classes 0 and 1")


def _validate_test_partition(
    features: npt.NDArray[np.floating[Any]],
    labels: npt.NDArray[np.int64],
) -> None:
    """Validate held-out inputs without using them for training decisions."""

    if features.ndim != 2:
        raise ValueError("Held-out embeddings must have shape [samples, hidden]")
    if labels.ndim != 1:
        raise ValueError("Held-out labels must have shape [samples]")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("Held-out embedding row count must match the label count")
    if features.shape[1] == 0:
        raise ValueError("Held-out embeddings must contain at least one feature")
    if not np.isfinite(features).all():
        raise ValueError("Held-out embeddings must contain only finite values")
    if not np.array_equal(np.unique(labels), np.array([0, 1])):
        raise ValueError("Held-out labels must contain both SST-2 classes 0 and 1")
