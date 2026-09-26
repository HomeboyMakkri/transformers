import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from transformers_learning import error_analysis
from transformers_learning.comparison import (
    Day6ArtifactInputs,
    FineTunedInferenceSetup,
    SentimentPrediction,
)
from transformers_learning.error_analysis import (
    Day7Holdout,
    Day7HoldoutPredictions,
    align_day7_holdout_predictions,
    build_day7_error_report,
    build_day7_error_tables,
    build_day7_length_summary_table,
    build_day7_prediction_preview,
    prepare_day7_holdout,
    run_day7_holdout_inference,
    save_day7_error_report,
    summarize_day7_errors,
)
from transformers_learning.splitting import split_outer_sentiment_indices


def make_sentiment_dataframe(rows_per_class: int = 10) -> pd.DataFrame:
    """Build rows whose text exposes source position despite duplicate indices."""

    labels = [0, 1] * rows_per_class
    return pd.DataFrame(
        {
            "text": [f"Review {position}" for position in range(len(labels))],
            "label": labels,
        },
        index=np.full(len(labels), 7, dtype=np.int64),
    )


def write_fine_tuned_artifact(directory: Path) -> None:
    """Create the minimum local binary artifact accepted before mocked loading."""

    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps({"num_labels": 2}),
        encoding="utf-8",
    )
    (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"test weights")


def make_predictions(texts: tuple[str, ...]) -> tuple[SentimentPrediction, ...]:
    """Return compact injected predictions in the received order."""

    return tuple(
        SentimentPrediction(
            text=text,
            prediction=index % 2,
            probabilities=np.array([0.75, 0.25], dtype=np.float64),
        )
        for index, text in enumerate(texts)
    )


def test_prepare_day7_holdout_preserves_source_positions_and_day6_order() -> None:
    dataframe = make_sentiment_dataframe()
    split = split_outer_sentiment_indices(
        dataframe["label"].to_numpy(dtype=np.int64)
    )

    holdout = prepare_day7_holdout(dataframe)

    assert holdout.source_positions.tolist() == split.test_indices.tolist()
    assert holdout.texts == tuple(
        dataframe.iloc[split.test_indices]["text"].tolist()
    )
    assert holdout.labels.tolist() == dataframe.iloc[split.test_indices][
        "label"
    ].tolist()
    assert not set(holdout.source_positions).intersection(split.train_indices)
    assert not hasattr(holdout, "outer_train")


def test_align_predictions_and_preview_keep_one_to_one_row_identity() -> None:
    holdout = prepare_day7_holdout(make_sentiment_dataframe())
    predictions = make_predictions(holdout.texts)

    result = align_day7_holdout_predictions(
        holdout,
        predictions,
        Path("fine_tuned_model"),
    )
    preview = build_day7_prediction_preview(result, limit=3)

    assert result.predictions == predictions
    assert preview.columns.tolist() == [
        "source_position",
        "text",
        "true_label",
        "predicted_label",
    ]
    assert preview["source_position"].tolist() == holdout.source_positions[:3].tolist()
    assert preview["text"].tolist() == list(holdout.texts[:3])
    assert preview["true_label"].tolist() == holdout.labels[:3].tolist()
    assert preview["predicted_label"].tolist() == [0, 1, 0]


@pytest.mark.parametrize("mode", ("missing", "reordered"))
def test_align_predictions_rejects_missing_or_reordered_texts(mode: str) -> None:
    holdout = prepare_day7_holdout(make_sentiment_dataframe())
    predictions = make_predictions(holdout.texts)
    if mode == "missing":
        invalid = predictions[:-1]
        message = "align with outer-test rows"
    else:
        invalid = (predictions[1], predictions[0], *predictions[2:])
        message = "preserve outer-test order"

    with pytest.raises(ValueError, match=message):
        align_day7_holdout_predictions(
            holdout,
            invalid,
            Path("fine_tuned_model"),
        )


def test_align_predictions_rejects_misaligned_injected_holdout() -> None:
    holdout = Day7Holdout(
        source_positions=np.array([3], dtype=np.int64),
        texts=("first", "second"),
        labels=np.array([0, 1], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="source positions"):
        align_day7_holdout_predictions(
            holdout,
            (),
            Path("fine_tuned_model"),
        )


def test_run_day7_holdout_inference_reuses_local_loader_and_predictor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataframe = make_sentiment_dataframe()
    model_directory = tmp_path / "fine_tuned_model"
    write_fine_tuned_artifact(model_directory)
    setup = FineTunedInferenceSetup(
        model=cast(PreTrainedModel, object()),
        tokenizer=cast(PreTrainedTokenizerBase, object()),
        device=error_analysis.torch.device("cpu"),
    )
    received: dict[str, object] = {}
    progress_callback = lambda completed, total: None

    def fake_load(
        artifact_inputs: Day6ArtifactInputs,
        device: error_analysis.torch.device | None = None,
    ) -> FineTunedInferenceSetup:
        received["load"] = (artifact_inputs, device)
        return setup

    def fake_predict(
        texts: tuple[str, ...],
        received_setup: FineTunedInferenceSetup,
        batch_size: int,
        progress_callback: object,
    ) -> tuple[SentimentPrediction, ...]:
        received["predict"] = (
            texts,
            received_setup,
            batch_size,
            progress_callback,
        )
        return make_predictions(texts)

    monkeypatch.setattr(error_analysis, "load_fine_tuned_inference", fake_load)
    monkeypatch.setattr(error_analysis, "predict_fine_tuned", fake_predict)

    result = run_day7_holdout_inference(
        dataframe,
        model_directory,
        batch_size=4,
        device=error_analysis.torch.device("cpu"),
        progress_callback=progress_callback,
    )

    assert received["load"] == (
        Day6ArtifactInputs(fine_tuned_model_directory=model_directory),
        error_analysis.torch.device("cpu"),
    )
    assert received["predict"] == (
        result.holdout.texts,
        setup,
        4,
        progress_callback,
    )
    assert result.fine_tuned_model_directory == model_directory


def make_error_analysis_result(
    labels: tuple[int, ...] = (0, 1, 0, 1),
    predictions: tuple[int, ...] = (1, 0, 0, 1),
) -> Day7HoldoutPredictions:
    """Build aligned injected rows with deterministic binary probabilities."""

    texts = ("bad", "great", "mixed", "bright")[: len(labels)]
    holdout = Day7Holdout(
        source_positions=np.arange(20, 20 + len(labels), dtype=np.int64),
        texts=texts,
        labels=np.asarray(labels, dtype=np.int64),
    )
    records = tuple(
        SentimentPrediction(
            text=text,
            prediction=prediction,
            probabilities=np.array(
                [0.8, 0.2] if prediction == 0 else [0.2, 0.8],
                dtype=np.float64,
            ),
        )
        for text, prediction in zip(texts, predictions)
    )
    return Day7HoldoutPredictions(
        holdout=holdout,
        predictions=records,
        fine_tuned_model_directory=Path("fine_tuned_model"),
    )


def test_build_error_tables_aligns_columns_confidence_and_fp_fn_partition() -> None:
    tables = build_day7_error_tables(make_error_analysis_result())

    assert tables.all_rows.columns.tolist() == [
        "source_position",
        "text",
        "true_label",
        "predicted_label",
        "probabilities",
        "predicted_confidence",
        "text_length",
    ]
    assert tables.all_rows["source_position"].tolist() == [20, 21, 22, 23]
    assert tables.all_rows["probabilities"].tolist() == [
        (0.2, 0.8),
        (0.8, 0.2),
        (0.8, 0.2),
        (0.2, 0.8),
    ]
    assert tables.all_rows["predicted_confidence"].tolist() == [0.8] * 4
    assert tables.all_rows["text_length"].tolist() == [3, 5, 5, 6]
    assert tables.errors["source_position"].tolist() == [20, 21]
    assert tables.false_positives["source_position"].tolist() == [20]
    assert tables.false_negatives["source_position"].tolist() == [21]
    assert len(tables.errors) == (
        len(tables.false_positives) + len(tables.false_negatives)
    )


def test_build_error_tables_accepts_all_correct_and_empty_error_groups() -> None:
    all_correct = build_day7_error_tables(
        make_error_analysis_result(predictions=(0, 1, 0, 1))
    )
    only_false_positive = build_day7_error_tables(
        make_error_analysis_result(
            labels=(0, 1),
            predictions=(1, 1),
        )
    )

    assert all_correct.errors.empty
    assert all_correct.false_positives.empty
    assert all_correct.false_negatives.empty
    assert only_false_positive.errors["source_position"].tolist() == [20]
    assert only_false_positive.false_positives["source_position"].tolist() == [20]
    assert only_false_positive.false_negatives.empty


@pytest.mark.parametrize(
    ("prediction", "probabilities", "message"),
    (
        (2, np.array([0.2, 0.8]), "labels 0 and 1"),
        (1, np.array([0.8]), "shape"),
        (1, np.array([float("nan"), 0.8]), "finite"),
        (1, np.array([0.2, 0.6]), "sum to one"),
        (1, np.array([-0.1, 1.1]), "between zero and one"),
        (0, np.array([0.2, 0.8]), "probability argmax"),
    ),
)
def test_build_error_tables_rejects_invalid_labels_and_probabilities(
    prediction: int,
    probabilities: np.ndarray,
    message: str,
) -> None:
    result = make_error_analysis_result(labels=(0,), predictions=(0,))
    invalid_record = SentimentPrediction(
        text=result.holdout.texts[0],
        prediction=prediction,
        probabilities=probabilities,
    )
    invalid_result = Day7HoldoutPredictions(
        holdout=result.holdout,
        predictions=(invalid_record,),
        fine_tuned_model_directory=result.fine_tuned_model_directory,
    )

    with pytest.raises((TypeError, ValueError), match=message):
        build_day7_error_tables(invalid_result)


def test_build_error_tables_rejects_reordered_or_missing_predictions() -> None:
    result = make_error_analysis_result()
    reordered = Day7HoldoutPredictions(
        holdout=result.holdout,
        predictions=(result.predictions[1], result.predictions[0], *result.predictions[2:]),
        fine_tuned_model_directory=result.fine_tuned_model_directory,
    )
    missing = Day7HoldoutPredictions(
        holdout=result.holdout,
        predictions=result.predictions[:-1],
        fine_tuned_model_directory=result.fine_tuned_model_directory,
    )

    with pytest.raises(ValueError, match="preserve outer-test order"):
        build_day7_error_tables(reordered)
    with pytest.raises(ValueError, match="align with outer-test rows"):
        build_day7_error_tables(missing)


def test_summarize_errors_reports_counts_rates_and_text_lengths() -> None:
    tables = build_day7_error_tables(make_error_analysis_result())

    summary = summarize_day7_errors(tables)
    length_table = build_day7_length_summary_table(summary)

    assert summary.total_rows == 4
    assert summary.correct_count == 2
    assert summary.error_count == 2
    assert summary.error_rate == pytest.approx(0.5)
    assert summary.false_positive_count == 1
    assert summary.false_negative_count == 1
    assert summary.correct_text_lengths.count == 2
    assert summary.correct_text_lengths.mean == pytest.approx(5.5)
    assert summary.correct_text_lengths.median == pytest.approx(5.5)
    assert summary.correct_text_lengths.minimum == 5
    assert summary.correct_text_lengths.maximum == 6
    assert summary.error_text_lengths.count == 2
    assert summary.error_text_lengths.mean == pytest.approx(4.0)
    assert summary.error_text_lengths.median == pytest.approx(4.0)
    assert summary.error_text_lengths.minimum == 3
    assert summary.error_text_lengths.maximum == 5
    assert length_table["group"].tolist() == ["correct", "error"]


def test_summarize_errors_handles_zero_errors_and_zero_correct_rows() -> None:
    no_errors = summarize_day7_errors(
        build_day7_error_tables(
            make_error_analysis_result(predictions=(0, 1, 0, 1))
        )
    )
    all_errors = summarize_day7_errors(
        build_day7_error_tables(
            make_error_analysis_result(predictions=(1, 0, 1, 0))
        )
    )

    assert no_errors.error_rate == pytest.approx(0.0)
    assert no_errors.error_text_lengths.count == 0
    assert no_errors.error_text_lengths.mean is None
    assert no_errors.error_text_lengths.minimum is None
    assert all_errors.error_rate == pytest.approx(1.0)
    assert all_errors.correct_text_lengths.count == 0
    assert all_errors.correct_text_lengths.median is None
    assert all_errors.correct_text_lengths.maximum is None


def test_error_report_is_deterministic_utf8_and_separates_hypotheses(
    tmp_path: Path,
) -> None:
    result = make_error_analysis_result()
    unicode_prediction = SentimentPrediction(
        text="плохо",
        prediction=1,
        probabilities=np.array([0.2, 0.8], dtype=np.float64),
    )
    unicode_holdout = Day7Holdout(
        source_positions=result.holdout.source_positions.copy(),
        texts=("плохо", *result.holdout.texts[1:]),
        labels=result.holdout.labels.copy(),
    )
    tables = build_day7_error_tables(
        Day7HoldoutPredictions(
            holdout=unicode_holdout,
            predictions=(unicode_prediction, *result.predictions[1:]),
            fine_tuned_model_directory=Path("fine_tuned_model"),
        )
    )

    first = build_day7_error_report(
        tables,
        qualitative_observations=("Negation may be worth inspecting.",),
        examples_per_type=2,
        dataset_identifier="test-sst2",
    )
    second = build_day7_error_report(
        tables,
        qualitative_observations=("Negation may be worth inspecting.",),
        examples_per_type=2,
        dataset_identifier="test-sst2",
    )
    report_path = save_day7_error_report(
        tables,
        tmp_path / "nested" / "error_analysis.txt",
        qualitative_observations=("Negation may be worth inspecting.",),
        examples_per_type=2,
        dataset_identifier="test-sst2",
    )

    assert first == second
    assert report_path.read_text(encoding="utf-8") == first
    assert "dataset_identifier: test-sst2" in first
    assert "label_mapping: 0=negative, 1=positive" in first
    assert "outer_test_size: 0.2" in first
    assert "random_state: 42" in first
    assert "test_sample_count: 4" in first
    assert "fine_tuned_checkpoint: fine_tuned_model" in first
    assert "error_rate: 0.500000" in first
    assert "плохо" in first
    assert "HUMAN-WRITTEN" in first
    assert "not measured causes" in first
    assert "- Negation may be worth inspecting." in first


def test_error_report_handles_empty_fp_and_fn_groups() -> None:
    tables = build_day7_error_tables(
        make_error_analysis_result(predictions=(0, 1, 0, 1))
    )

    report = build_day7_error_report(tables)

    assert report.count("No examples.") == 2
    assert "error_text_length: count=0, mean=n/a" in report
    assert "No qualitative observations recorded." in report
