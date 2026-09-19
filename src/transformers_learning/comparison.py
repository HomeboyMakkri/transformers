"""Shared Day 6 data and artifact boundaries for model comparison."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from .baseline import (
    prepare_frozen_embedding_dataset,
    train_logistic_regression_on_frozen_embeddings,
)
from .datasets import (
    LABEL_COLUMN,
    SST2_LABEL_MAP,
    TEXT_COLUMN,
    validate_sentiment_dataframe,
)
from .fine_tuning import DEFAULT_FINE_TUNED_MODEL_DIRECTORY, NUM_SENTIMENT_LABELS
from .modeling import get_embeddings
from .splitting import OuterSentimentSplit, split_outer_sentiment_indices
from .tokenization import tokenize_texts

_MODEL_CONFIG_FILENAME = "config.json"
_TOKENIZER_CONFIG_FILENAME = "tokenizer_config.json"
_MODEL_WEIGHT_FILENAMES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
DAY6_EXAMPLE_TEXTS = (
    "This movie was absolutely fantastic!",
    "Terrible, waste of my time.",
    "It was okay, nothing special.",
    "Best film I've seen this year!",
    "Boring and too long.",
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


@dataclass(frozen=True)
class SentimentPrediction:
    """One ordered binary prediction with class probabilities in label order."""

    text: str
    prediction: int
    probabilities: npt.NDArray[np.float64]


@dataclass(frozen=True)
class ExampleComparison:
    """Aligned predictions from both Day 6 paths for one illustrative text."""

    text: str
    fine_tuned: SentimentPrediction
    baseline: SentimentPrediction
    predictions_agree: bool


@dataclass(frozen=True)
class HeldOutModelEvaluation:
    """One model's ordered holdout predictions and aggregate binary metrics."""

    predictions: npt.NDArray[np.int64]
    classification_report: str
    class_support: dict[int, int]
    accuracy: float
    macro_f1: float


@dataclass(frozen=True)
class PairedHoldoutEvaluation:
    """Metrics for two models evaluated on the identical ordered holdout."""

    labels: npt.NDArray[np.int64]
    fine_tuned: HeldOutModelEvaluation
    baseline: HeldOutModelEvaluation
    accuracy_delta: float
    macro_f1_delta: float
    relative_macro_f1_delta: float | None


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


def predict_fine_tuned(
    texts: str | Sequence[str],
    setup: FineTunedInferenceSetup,
    batch_size: int = 32,
) -> tuple[SentimentPrediction, ...]:
    """Predict binary sentiment from one text or an ordered text sequence."""

    normalized_texts = _normalize_prediction_texts(texts)
    _validate_batch_size(batch_size)
    setup.model.eval()
    predictions: list[SentimentPrediction] = []

    for start in range(0, len(normalized_texts), batch_size):
        batch_texts = normalized_texts[start : start + batch_size]
        encoded = tokenize_texts(
            batch_texts,
            setup.tokenizer,
            max_length=128,
        ).to(setup.device)
        with torch.no_grad():
            outputs = cast(SequenceClassifierOutput, setup.model(**encoded))
        probabilities = _probabilities_from_logits(outputs.logits, len(batch_texts))
        labels = probabilities.argmax(dim=1)
        probability_rows = np.asarray(
            probabilities.detach().cpu().numpy(), dtype=np.float64
        )
        for text, label, row in zip(batch_texts, labels.tolist(), probability_rows):
            predictions.append(
                SentimentPrediction(
                    text=text,
                    prediction=int(label),
                    probabilities=row.copy(),
                )
            )
    return tuple(predictions)


def predict_baseline(
    texts: str | Sequence[str],
    setup: FrozenBaselineSetup,
    batch_size: int = 32,
) -> tuple[SentimentPrediction, ...]:
    """Predict binary sentiment through frozen embeddings and Logistic Regression."""

    normalized_texts = _normalize_prediction_texts(texts)
    _validate_batch_size(batch_size)
    features = np.asarray(
        get_embeddings(
            normalized_texts,
            setup.tokenizer,
            setup.encoder,
            batch_size=batch_size,
        )
    )
    _validate_prediction_features(features, len(normalized_texts))
    predictions = _validate_baseline_labels(
        setup.classifier.predict(features),
        len(normalized_texts),
    )
    probabilities = _baseline_probabilities_in_label_order(
        setup.classifier,
        features,
        len(normalized_texts),
    )
    return tuple(
        SentimentPrediction(
            text=text,
            prediction=int(prediction),
            probabilities=probability.copy(),
        )
        for text, prediction, probability in zip(
            normalized_texts,
            predictions,
            probabilities,
        )
    )


def compare_five_examples(
    fine_tuned_setup: FineTunedInferenceSetup,
    baseline_setup: FrozenBaselineSetup,
    batch_size: int = 32,
) -> tuple[ExampleComparison, ...]:
    """Run both prediction paths on the five fixed Day 6 example sentences."""

    return compare_example_predictions(
        DAY6_EXAMPLE_TEXTS,
        fine_tuned_setup,
        baseline_setup,
        batch_size=batch_size,
    )


def compare_example_predictions(
    texts: str | Sequence[str],
    fine_tuned_setup: FineTunedInferenceSetup,
    baseline_setup: FrozenBaselineSetup,
    batch_size: int = 32,
) -> tuple[ExampleComparison, ...]:
    """Pair two prediction paths without treating selected examples as metrics."""

    normalized_texts = _normalize_prediction_texts(texts)
    fine_tuned_predictions = predict_fine_tuned(
        normalized_texts,
        fine_tuned_setup,
        batch_size=batch_size,
    )
    baseline_predictions = predict_baseline(
        normalized_texts,
        baseline_setup,
        batch_size=batch_size,
    )
    if len(fine_tuned_predictions) != len(normalized_texts) or len(
        baseline_predictions
    ) != len(normalized_texts):
        raise ValueError("Example predictions must align with every input text")

    comparisons: list[ExampleComparison] = []
    for text, fine_tuned, baseline in zip(
        normalized_texts,
        fine_tuned_predictions,
        baseline_predictions,
    ):
        _validate_example_prediction(fine_tuned, text, "Fine-tuned")
        _validate_example_prediction(baseline, text, "Baseline")
        comparisons.append(
            ExampleComparison(
                text=text,
                fine_tuned=fine_tuned,
                baseline=baseline,
                predictions_agree=fine_tuned.prediction == baseline.prediction,
            )
        )
    return tuple(comparisons)


def build_example_comparison_table(
    comparisons: Sequence[ExampleComparison],
) -> pd.DataFrame:
    """Return a display-ready table of labels, probabilities, and agreement."""

    return pd.DataFrame(
        {
            "text": comparison.text,
            "fine_tuned_prediction": comparison.fine_tuned.prediction,
            "fine_tuned_probabilities": tuple(
                float(value) for value in comparison.fine_tuned.probabilities
            ),
            "baseline_prediction": comparison.baseline.prediction,
            "baseline_probabilities": tuple(
                float(value) for value in comparison.baseline.probabilities
            ),
            "predictions_agree": comparison.predictions_agree,
        }
        for comparison in comparisons
    )


def evaluate_paired_holdout(
    comparison: ComparisonDataset,
    fine_tuned_setup: FineTunedInferenceSetup,
    baseline_setup: FrozenBaselineSetup,
    batch_size: int = 32,
) -> PairedHoldoutEvaluation:
    """Evaluate both fixed model paths on the same ordered outer-test rows."""

    texts = tuple(str(text) for text in comparison.outer_test[TEXT_COLUMN])
    labels = np.asarray(comparison.outer_test[LABEL_COLUMN].to_numpy(), dtype=np.int64)
    _validate_holdout_labels(labels, len(texts))
    fine_tuned_predictions = predict_fine_tuned(
        texts,
        fine_tuned_setup,
        batch_size=batch_size,
    )
    baseline_predictions = predict_baseline(
        texts,
        baseline_setup,
        batch_size=batch_size,
    )
    fine_tuned_labels = _extract_aligned_prediction_labels(
        fine_tuned_predictions,
        texts,
        "Fine-tuned",
    )
    baseline_labels = _extract_aligned_prediction_labels(
        baseline_predictions,
        texts,
        "Baseline",
    )
    fine_tuned_evaluation = _evaluate_holdout_predictions(labels, fine_tuned_labels)
    baseline_evaluation = _evaluate_holdout_predictions(labels, baseline_labels)
    macro_f1_delta = fine_tuned_evaluation.macro_f1 - baseline_evaluation.macro_f1
    relative_macro_f1_delta = (
        None
        if baseline_evaluation.macro_f1 == 0.0
        else macro_f1_delta / baseline_evaluation.macro_f1
    )
    return PairedHoldoutEvaluation(
        labels=labels.copy(),
        fine_tuned=fine_tuned_evaluation,
        baseline=baseline_evaluation,
        accuracy_delta=fine_tuned_evaluation.accuracy - baseline_evaluation.accuracy,
        macro_f1_delta=macro_f1_delta,
        relative_macro_f1_delta=relative_macro_f1_delta,
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


def _normalize_prediction_texts(texts: str | Sequence[str]) -> tuple[str, ...]:
    """Accept a scalar text or a non-empty ordered sequence of valid texts."""

    normalized = (texts,) if isinstance(texts, str) else tuple(texts)
    if not normalized:
        raise ValueError("texts must contain at least one item")
    if not all(isinstance(text, str) and text.strip() for text in normalized):
        raise ValueError("texts must contain only non-empty strings")
    return normalized


def _validate_batch_size(batch_size: int) -> None:
    """Reject invalid public batch sizes before tokenization."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")


def _probabilities_from_logits(logits: object, batch_size: int) -> torch.Tensor:
    """Validate binary logits and convert them to finite probability rows."""

    if not isinstance(logits, torch.Tensor):
        raise TypeError("Fine-tuned logits must be a tensor")
    if logits.ndim != 2 or logits.shape != (batch_size, NUM_SENTIMENT_LABELS):
        raise ValueError("Fine-tuned logits must have shape [batch, 2]")
    if not torch.isfinite(logits).all():
        raise ValueError("Fine-tuned logits must contain only finite values")

    probabilities = torch.softmax(logits, dim=1)
    if not torch.isfinite(probabilities).all() or not torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(batch_size, device=probabilities.device, dtype=probabilities.dtype),
    ):
        raise ValueError("Fine-tuned probabilities must be finite and sum to one")
    return probabilities


def _validate_prediction_features(features: npt.NDArray[Any], count: int) -> None:
    """Require finite aligned frozen feature rows before classifier inference."""

    if features.ndim != 2 or features.shape[0] != count or features.shape[1] == 0:
        raise ValueError("Frozen embeddings must have shape [batch, hidden]")
    if not np.issubdtype(features.dtype, np.number) or not np.isfinite(features).all():
        raise ValueError("Frozen embeddings must contain only finite numeric values")


def _validate_example_prediction(
    prediction: SentimentPrediction,
    expected_text: str,
    model_name: str,
) -> None:
    """Check one predictor record before it is paired into a comparison table."""

    if prediction.text != expected_text:
        raise ValueError(f"{model_name} prediction text does not preserve input order")
    if prediction.prediction not in (0, 1):
        raise ValueError(f"{model_name} prediction must use SST-2 labels 0 and 1")
    probabilities = prediction.probabilities
    if probabilities.shape != (NUM_SENTIMENT_LABELS,):
        raise ValueError(f"{model_name} probabilities must have shape [2]")
    if not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1.0):
        raise ValueError(f"{model_name} probabilities must be finite and sum to one")


def _extract_aligned_prediction_labels(
    predictions: Sequence[SentimentPrediction],
    texts: Sequence[str],
    model_name: str,
) -> npt.NDArray[np.int64]:
    """Ensure prediction records have one valid row for every ordered text."""

    if len(predictions) != len(texts):
        raise ValueError(f"{model_name} predictions must align with held-out texts")
    labels: list[int] = []
    for expected_text, prediction in zip(texts, predictions):
        _validate_example_prediction(prediction, expected_text, model_name)
        labels.append(prediction.prediction)
    return np.asarray(labels, dtype=np.int64)


def _validate_holdout_labels(labels: npt.NDArray[np.int64], count: int) -> None:
    """Require one binary held-out label for every shared outer-test text."""

    if labels.ndim != 1 or labels.shape[0] != count:
        raise ValueError("Held-out labels must align with held-out texts")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Held-out labels must use only SST-2 labels 0 and 1")


def _evaluate_holdout_predictions(
    labels: npt.NDArray[np.int64],
    predictions: npt.NDArray[np.int64],
) -> HeldOutModelEvaluation:
    """Compute fixed-label held-out metrics without modifying either model."""

    if predictions.shape != labels.shape:
        raise ValueError("Held-out predictions must align with held-out labels")
    report = cast(
        str,
        classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=[SST2_LABEL_MAP[0], SST2_LABEL_MAP[1]],
            zero_division=cast(Any, 0),
        ),
    )
    accuracy = float(accuracy_score(labels, predictions))
    macro_f1 = float(
        f1_score(
            labels,
            predictions,
            labels=[0, 1],
            average="macro",
            zero_division=cast(Any, 0),
        )
    )
    class_support = {label: int(np.count_nonzero(labels == label)) for label in (0, 1)}
    return HeldOutModelEvaluation(
        predictions=predictions.copy(),
        classification_report=report,
        class_support=class_support,
        accuracy=accuracy,
        macro_f1=macro_f1,
    )


def _validate_baseline_labels(predictions: object, count: int) -> npt.NDArray[np.int64]:
    """Validate one binary classifier label per input text."""

    values = np.asarray(predictions)
    if values.ndim != 1 or values.shape[0] != count:
        raise ValueError("Baseline predictions must align with input texts")
    if not np.all(np.isin(values, [0, 1])):
        raise ValueError("Baseline predictions must use only SST-2 labels 0 and 1")
    return np.asarray(values, dtype=np.int64)


def _baseline_probabilities_in_label_order(
    classifier: LogisticRegression,
    features: npt.NDArray[Any],
    count: int,
) -> npt.NDArray[np.float64]:
    """Validate ``predict_proba`` and reorder columns to SST-2 labels ``[0, 1]``."""

    raw_probabilities = np.asarray(classifier.predict_proba(features), dtype=np.float64)
    if raw_probabilities.shape != (count, NUM_SENTIMENT_LABELS):
        raise ValueError("Baseline probabilities must have shape [batch, 2]")
    if not np.isfinite(raw_probabilities).all() or np.any(raw_probabilities < 0.0):
        raise ValueError("Baseline probabilities must be finite and non-negative")
    if not np.allclose(raw_probabilities.sum(axis=1), np.ones(count)):
        raise ValueError("Baseline probabilities must sum to one")

    classes = np.asarray(classifier.classes_)
    if classes.ndim != 1 or classes.shape[0] != NUM_SENTIMENT_LABELS:
        raise ValueError("Baseline classifier must expose two class labels")
    if set(classes.tolist()) != {0, 1}:
        raise ValueError("Baseline classifier classes must be SST-2 labels 0 and 1")
    label_columns = [int(np.flatnonzero(classes == label)[0]) for label in (0, 1)]
    return raw_probabilities[:, label_columns]
