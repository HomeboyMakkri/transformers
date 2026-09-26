"""Day 7 held-out inputs and ordered fine-tuned predictions."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from .comparison import (
    SentimentPrediction,
    get_day6_artifact_inputs,
    load_fine_tuned_inference,
    predict_fine_tuned,
    prepare_comparison_dataset,
)
from .datasets import LABEL_COLUMN, SST2_LABEL_MAP, SST2_METADATA, TEXT_COLUMN
from .fine_tuning import DEFAULT_FINE_TUNED_MODEL_DIRECTORY
from .modeling import ProgressCallback
from .splitting import OUTER_TEST_SIZE, SPLIT_RANDOM_STATE

DEFAULT_ERROR_ANALYSIS_PATH = Path("error_analysis.txt")


@dataclass(frozen=True)
class Day7Holdout:
    """The unchanged ordered outer-test rows selected from source positions."""

    source_positions: npt.NDArray[np.int64]
    texts: tuple[str, ...]
    labels: npt.NDArray[np.int64]


@dataclass(frozen=True)
class Day7HoldoutPredictions:
    """Aligned fine-tuned predictions and their local artifact context."""

    holdout: Day7Holdout
    predictions: tuple[SentimentPrediction, ...]
    fine_tuned_model_directory: Path


@dataclass(frozen=True)
class Day7ErrorTables:
    """Aligned rows and their exhaustive binary FP/FN error partition."""

    all_rows: pd.DataFrame
    errors: pd.DataFrame
    false_positives: pd.DataFrame
    false_negatives: pd.DataFrame
    fine_tuned_model_directory: Path


@dataclass(frozen=True)
class TextLengthSummary:
    """Descriptive character-length statistics for one row group."""

    count: int
    mean: float | None
    median: float | None
    minimum: int | None
    maximum: int | None


@dataclass(frozen=True)
class Day7ErrorSummary:
    """Measured aggregate counts and text lengths for the outer holdout."""

    total_rows: int
    correct_count: int
    error_count: int
    error_rate: float
    false_positive_count: int
    false_negative_count: int
    correct_text_lengths: TextLengthSummary
    error_text_lengths: TextLengthSummary


def prepare_day7_holdout(dataframe: pd.DataFrame) -> Day7Holdout:
    """Recreate the shared Day 4-6 outer holdout without exposing train rows."""

    comparison = prepare_comparison_dataset(dataframe)
    outer_test = comparison.outer_test
    texts = tuple(
        cast(str, value) for value in outer_test[TEXT_COLUMN].tolist()
    )
    labels = np.asarray(outer_test[LABEL_COLUMN].to_numpy(), dtype=np.int64)
    source_positions = np.asarray(
        comparison.split.test_indices,
        dtype=np.int64,
    ).copy()
    _validate_day7_holdout(source_positions, texts, labels)
    return Day7Holdout(
        source_positions=source_positions,
        texts=texts,
        labels=labels.copy(),
    )


def align_day7_holdout_predictions(
    holdout: Day7Holdout,
    predictions: Sequence[SentimentPrediction],
    fine_tuned_model_directory: Path,
) -> Day7HoldoutPredictions:
    """Validate one ordered prediction record for every outer-test text."""

    _validate_day7_holdout(
        holdout.source_positions,
        holdout.texts,
        holdout.labels,
    )
    aligned = tuple(predictions)
    if len(aligned) != len(holdout.texts):
        raise ValueError("Day 7 predictions must align with outer-test rows")
    for expected_text, prediction in zip(holdout.texts, aligned):
        if prediction.text != expected_text:
            raise ValueError(
                "Day 7 prediction texts must preserve outer-test order"
            )
    return Day7HoldoutPredictions(
        holdout=holdout,
        predictions=aligned,
        fine_tuned_model_directory=fine_tuned_model_directory,
    )


def run_day7_holdout_inference(
    dataframe: pd.DataFrame,
    fine_tuned_model_directory: Path = DEFAULT_FINE_TUNED_MODEL_DIRECTORY,
    batch_size: int = 32,
    device: torch.device | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Day7HoldoutPredictions:
    """Reload the local epoch-3 classifier and predict the shared outer holdout."""

    holdout = prepare_day7_holdout(dataframe)
    artifact_inputs = get_day6_artifact_inputs(fine_tuned_model_directory)
    setup = load_fine_tuned_inference(artifact_inputs, device=device)
    predictions = predict_fine_tuned(
        holdout.texts,
        setup,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )
    return align_day7_holdout_predictions(
        holdout,
        predictions,
        artifact_inputs.fine_tuned_model_directory,
    )


def build_day7_prediction_preview(
    result: Day7HoldoutPredictions,
    limit: int = 5,
) -> pd.DataFrame:
    """Build a small alignment preview; full error columns belong to D7-02."""

    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be an integer")
    if limit <= 0:
        raise ValueError("limit must be positive")
    aligned = align_day7_holdout_predictions(
        result.holdout,
        result.predictions,
        result.fine_tuned_model_directory,
    )
    row_count = min(limit, len(aligned.holdout.texts))
    return pd.DataFrame(
        {
            "source_position": aligned.holdout.source_positions[:row_count],
            "text": aligned.holdout.texts[:row_count],
            "true_label": aligned.holdout.labels[:row_count],
            "predicted_label": [
                prediction.prediction
                for prediction in aligned.predictions[:row_count]
            ],
        }
    )


def build_day7_error_tables(
    result: Day7HoldoutPredictions,
) -> Day7ErrorTables:
    """Build validated aligned rows and split every binary error into FP or FN."""

    aligned = align_day7_holdout_predictions(
        result.holdout,
        result.predictions,
        result.fine_tuned_model_directory,
    )
    rows: list[dict[str, object]] = []
    for source_position, text, true_label, prediction in zip(
        aligned.holdout.source_positions,
        aligned.holdout.texts,
        aligned.holdout.labels,
        aligned.predictions,
    ):
        predicted_label, probabilities = _validate_day7_prediction(prediction)
        rows.append(
            {
                "source_position": int(source_position),
                "text": text,
                "true_label": int(true_label),
                "predicted_label": predicted_label,
                "probabilities": (
                    float(probabilities[0]),
                    float(probabilities[1]),
                ),
                "predicted_confidence": float(probabilities[predicted_label]),
                "text_length": len(text),
            }
        )

    all_rows = pd.DataFrame.from_records(rows)
    error_mask = all_rows["true_label"] != all_rows["predicted_label"]
    false_positive_mask = (all_rows["true_label"] == 0) & (
        all_rows["predicted_label"] == 1
    )
    false_negative_mask = (all_rows["true_label"] == 1) & (
        all_rows["predicted_label"] == 0
    )
    errors = all_rows.loc[error_mask].reset_index(drop=True)
    false_positives = all_rows.loc[false_positive_mask].reset_index(drop=True)
    false_negatives = all_rows.loc[false_negative_mask].reset_index(drop=True)
    if len(errors) != len(false_positives) + len(false_negatives):
        raise ValueError("Every Day 7 binary error must be exactly one of FP or FN")
    return Day7ErrorTables(
        all_rows=all_rows,
        errors=errors,
        false_positives=false_positives,
        false_negatives=false_negatives,
        fine_tuned_model_directory=aligned.fine_tuned_model_directory,
    )


def summarize_day7_errors(tables: Day7ErrorTables) -> Day7ErrorSummary:
    """Summarize measured error counts and correct-versus-error text lengths."""

    total_rows = len(tables.all_rows)
    if total_rows == 0:
        raise ValueError("Day 7 error tables must contain holdout rows")
    error_count = len(tables.errors)
    false_positive_count = len(tables.false_positives)
    false_negative_count = len(tables.false_negatives)
    if error_count != false_positive_count + false_negative_count:
        raise ValueError("Day 7 errors must be exhaustively partitioned into FP and FN")
    correct_rows = tables.all_rows.loc[
        tables.all_rows["true_label"] == tables.all_rows["predicted_label"]
    ]
    if len(correct_rows) + error_count != total_rows:
        raise ValueError("Day 7 correct and error rows must cover the holdout")
    return Day7ErrorSummary(
        total_rows=total_rows,
        correct_count=len(correct_rows),
        error_count=error_count,
        error_rate=error_count / total_rows,
        false_positive_count=false_positive_count,
        false_negative_count=false_negative_count,
        correct_text_lengths=_summarize_text_lengths(correct_rows),
        error_text_lengths=_summarize_text_lengths(tables.errors),
    )


def build_day7_length_summary_table(summary: Day7ErrorSummary) -> pd.DataFrame:
    """Present correct-versus-error character lengths as a compact table."""

    return pd.DataFrame.from_records(
        (
            _length_summary_record("correct", summary.correct_text_lengths),
            _length_summary_record("error", summary.error_text_lengths),
        )
    )


def build_day7_error_report(
    tables: Day7ErrorTables,
    qualitative_observations: Sequence[str] = (),
    examples_per_type: int = 5,
    dataset_identifier: str = SST2_METADATA.identifier,
) -> str:
    """Build a deterministic report separating measurements from hypotheses."""

    _validate_report_inputs(
        qualitative_observations,
        examples_per_type,
        dataset_identifier,
    )
    summary = summarize_day7_errors(tables)
    label_mapping = ", ".join(
        f"{label}={name}" for label, name in SST2_LABEL_MAP.items()
    )
    lines = [
        "=== DAY 7 ERROR ANALYSIS ===",
        "",
        "=== RUN CONTEXT ===",
        f"dataset_identifier: {dataset_identifier}",
        f"label_mapping: {label_mapping}",
        f"outer_test_size: {OUTER_TEST_SIZE}",
        f"random_state: {SPLIT_RANDOM_STATE}",
        f"test_sample_count: {summary.total_rows}",
        f"fine_tuned_checkpoint: {tables.fine_tuned_model_directory}",
        "",
        "=== MEASURED FACTS ===",
        f"total_rows: {summary.total_rows}",
        f"correct_count: {summary.correct_count}",
        f"error_count: {summary.error_count}",
        f"error_rate: {summary.error_rate:.6f}",
        f"false_positive_count: {summary.false_positive_count}",
        f"false_negative_count: {summary.false_negative_count}",
        _format_length_summary("correct_text_length", summary.correct_text_lengths),
        _format_length_summary("error_text_length", summary.error_text_lengths),
        "",
    ]
    lines.extend(
        _format_error_examples(
            "FALSE POSITIVE EXAMPLES (true=negative, predicted=positive)",
            tables.false_positives,
            examples_per_type,
        )
    )
    lines.append("")
    lines.extend(
        _format_error_examples(
            "FALSE NEGATIVE EXAMPLES (true=positive, predicted=negative)",
            tables.false_negatives,
            examples_per_type,
        )
    )
    lines.extend(
        [
            "",
            "=== QUALITATIVE HYPOTHESES (HUMAN-WRITTEN) ===",
            (
                "These observations are hypotheses for inspection, not measured "
                "causes and not instructions for tuning on the holdout."
            ),
        ]
    )
    if qualitative_observations:
        lines.extend(f"- {observation.strip()}" for observation in qualitative_observations)
    else:
        lines.append("- No qualitative observations recorded.")
    return "\n".join(lines) + "\n"


def save_day7_error_report(
    tables: Day7ErrorTables,
    path: Path = DEFAULT_ERROR_ANALYSIS_PATH,
    qualitative_observations: Sequence[str] = (),
    examples_per_type: int = 5,
    dataset_identifier: str = SST2_METADATA.identifier,
) -> Path:
    """Write the deterministic UTF-8 Day 7 report to an ignored local path."""

    report = build_day7_error_report(
        tables,
        qualitative_observations=qualitative_observations,
        examples_per_type=examples_per_type,
        dataset_identifier=dataset_identifier,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return path


def _validate_day7_holdout(
    source_positions: npt.NDArray[np.int64],
    texts: Sequence[str],
    labels: npt.NDArray[np.int64],
) -> None:
    """Require aligned, non-empty source positions, texts, and binary labels."""

    count = len(texts)
    if count == 0:
        raise ValueError("Day 7 outer holdout must contain rows")
    if source_positions.ndim != 1 or source_positions.shape[0] != count:
        raise ValueError("Day 7 source positions must align with outer-test texts")
    if labels.ndim != 1 or labels.shape[0] != count:
        raise ValueError("Day 7 labels must align with outer-test texts")
    if not all(isinstance(text, str) and text.strip() for text in texts):
        raise ValueError("Day 7 outer-test texts must be non-empty strings")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Day 7 labels must use only SST-2 labels 0 and 1")


def _validate_day7_prediction(
    prediction: SentimentPrediction,
) -> tuple[int, npt.NDArray[np.float64]]:
    """Validate one binary label and its `[negative, positive]` probabilities."""

    predicted_label = prediction.prediction
    if isinstance(predicted_label, bool) or predicted_label not in (0, 1):
        raise ValueError("Day 7 predictions must use only SST-2 labels 0 and 1")
    probabilities = np.asarray(prediction.probabilities)
    if probabilities.shape != (2,):
        raise ValueError("Day 7 probabilities must have shape [2]")
    if not np.issubdtype(probabilities.dtype, np.number):
        raise TypeError("Day 7 probabilities must be numeric")
    numeric_probabilities = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(numeric_probabilities).all():
        raise ValueError("Day 7 probabilities must contain only finite values")
    if np.any(numeric_probabilities < 0.0) or np.any(
        numeric_probabilities > 1.0
    ):
        raise ValueError("Day 7 probabilities must be between zero and one")
    if not np.isclose(numeric_probabilities.sum(), 1.0):
        raise ValueError("Day 7 probabilities must sum to one")
    if int(numeric_probabilities.argmax()) != predicted_label:
        raise ValueError("Day 7 predicted labels must match probability argmax")
    return predicted_label, numeric_probabilities


def _summarize_text_lengths(dataframe: pd.DataFrame) -> TextLengthSummary:
    """Return explicit `None` statistics for an empty row group."""

    lengths = np.asarray(dataframe["text_length"].to_numpy(), dtype=np.int64)
    if lengths.size == 0:
        return TextLengthSummary(
            count=0,
            mean=None,
            median=None,
            minimum=None,
            maximum=None,
        )
    return TextLengthSummary(
        count=int(lengths.size),
        mean=float(lengths.mean()),
        median=float(np.median(lengths)),
        minimum=int(lengths.min()),
        maximum=int(lengths.max()),
    )


def _length_summary_record(
    group: str,
    summary: TextLengthSummary,
) -> dict[str, object]:
    """Convert one typed length summary into a display-table row."""

    return {
        "group": group,
        "count": summary.count,
        "mean": summary.mean,
        "median": summary.median,
        "minimum": summary.minimum,
        "maximum": summary.maximum,
    }


def _format_length_summary(name: str, summary: TextLengthSummary) -> str:
    """Format one text-length group without emitting NaN for empty groups."""

    if summary.count == 0:
        return f"{name}: count=0, mean=n/a, median=n/a, min=n/a, max=n/a"
    return (
        f"{name}: count={summary.count}, mean={summary.mean:.2f}, "
        f"median={summary.median:.2f}, min={summary.minimum}, "
        f"max={summary.maximum}"
    )


def _format_error_examples(
    heading: str,
    dataframe: pd.DataFrame,
    limit: int,
) -> list[str]:
    """Format examples in stable source-row order."""

    lines = [f"=== {heading} ==="]
    examples = dataframe.sort_values("source_position", kind="stable").head(limit)
    if examples.empty:
        lines.append("No examples.")
        return lines
    for number, (_, row) in enumerate(examples.iterrows(), start=1):
        negative_probability, positive_probability = row["probabilities"]
        lines.extend(
            [
                f"[{number}] source_position: {int(row['source_position'])}",
                f"text: {row['text']}",
                f"true_label: {int(row['true_label'])}",
                f"predicted_label: {int(row['predicted_label'])}",
                (
                    "probabilities: "
                    f"negative={float(negative_probability):.6f}, "
                    f"positive={float(positive_probability):.6f}"
                ),
                f"predicted_confidence: {float(row['predicted_confidence']):.6f}",
                f"text_length: {int(row['text_length'])}",
                "",
            ]
        )
    if lines[-1] == "":
        lines.pop()
    return lines


def _validate_report_inputs(
    qualitative_observations: Sequence[str],
    examples_per_type: int,
    dataset_identifier: str,
) -> None:
    """Reject ambiguous report context and placeholder observations."""

    if isinstance(examples_per_type, bool) or not isinstance(examples_per_type, int):
        raise TypeError("examples_per_type must be an integer")
    if examples_per_type <= 0:
        raise ValueError("examples_per_type must be positive")
    if not isinstance(dataset_identifier, str) or not dataset_identifier.strip():
        raise ValueError("dataset_identifier must be a non-empty string")
    if isinstance(qualitative_observations, str) or not all(
        isinstance(observation, str) and observation.strip()
        for observation in qualitative_observations
    ):
        raise ValueError(
            "qualitative_observations must contain only non-empty strings"
        )
