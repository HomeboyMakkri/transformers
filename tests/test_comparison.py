import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from transformers_learning import comparison
from transformers_learning.baseline import FrozenEmbeddingDataset
from transformers_learning.comparison import (
    DAY6_EXAMPLE_TEXTS,
    Day6ArtifactInputs,
    FineTunedArtifactError,
    FineTunedInferenceSetup,
    FrozenBaselineSetup,
    SentimentPrediction,
    build_example_comparison_table,
    compare_example_predictions,
    compare_five_examples,
    get_day6_artifact_inputs,
    load_fine_tuned_inference,
    predict_baseline,
    predict_fine_tuned,
    prepare_comparison_dataset,
    recreate_frozen_baseline,
)
from transformers_learning.splitting import (
    split_outer_sentiment_indices,
    split_sentiment_row_indices,
)


def make_sentiment_dataframe(rows_per_class: int = 10) -> pd.DataFrame:
    """Build labelled rows whose text exposes their original position."""

    labels = [0, 1] * rows_per_class
    return pd.DataFrame(
        {
            "text": [f"Review {row_id}" for row_id in range(len(labels))],
            "label": labels,
        },
        index=np.arange(100, 100 + len(labels)),
    )


def write_fine_tuned_artifact(
    directory: Path,
    config: dict[str, object] | None = None,
) -> None:
    """Create the minimum local Day 5 artifact structure for contract tests."""

    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps({"num_labels": 2} if config is None else config),
        encoding="utf-8",
    )
    (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"test weights")


def test_prepare_comparison_dataset_matches_day5_outer_holdout_exactly() -> None:
    dataframe = make_sentiment_dataframe()

    comparison = prepare_comparison_dataset(dataframe)
    day5_split = split_sentiment_row_indices(dataframe)
    outer_split = split_outer_sentiment_indices(
        dataframe["label"].to_numpy(dtype=np.int64)
    )

    assert comparison.split.test_indices.tolist() == day5_split.test_indices.tolist()
    assert comparison.outer_train.equals(dataframe.iloc[outer_split.train_indices])
    assert comparison.outer_test.equals(dataframe.iloc[day5_split.test_indices])
    assert comparison.outer_test["text"].tolist() == dataframe.iloc[
        day5_split.test_indices
    ]["text"].tolist()
    assert comparison.outer_test["label"].tolist() == dataframe.iloc[
        day5_split.test_indices
    ]["label"].tolist()


def test_prepare_comparison_dataset_is_complete_disjoint_and_deterministic() -> None:
    dataframe = make_sentiment_dataframe()

    first = prepare_comparison_dataset(dataframe)
    second = prepare_comparison_dataset(dataframe)

    assert first.split.train_indices.tolist() == second.split.train_indices.tolist()
    assert first.split.test_indices.tolist() == second.split.test_indices.tolist()
    assert set(first.split.train_indices).isdisjoint(set(first.split.test_indices))
    assert set(first.split.train_indices) | set(first.split.test_indices) == set(
        range(len(dataframe))
    )


def test_get_day6_artifact_inputs_requires_reloadable_binary_artifact(
    tmp_path: Path,
) -> None:
    model_directory = tmp_path / "fine_tuned_model"
    write_fine_tuned_artifact(model_directory)

    inputs = get_day6_artifact_inputs(model_directory)

    assert inputs.fine_tuned_model_directory == model_directory
    assert not hasattr(inputs, "baseline_model_path")
    assert not hasattr(inputs, "vectorizer_path")


@pytest.mark.parametrize(
    ("setup", "error_type", "message"),
    (
        ("missing", FileNotFoundError, "does not exist"),
        ("missing-tokenizer", FileNotFoundError, "tokenizer config"),
        ("missing-weights", FileNotFoundError, "model weights"),
        ("three-labels", FineTunedArtifactError, "binary SST-2"),
        ("invalid-json", FineTunedArtifactError, "valid JSON"),
    ),
)
def test_get_day6_artifact_inputs_rejects_absent_or_incompatible_artifacts(
    tmp_path: Path,
    setup: str,
    error_type: type[Exception],
    message: str,
) -> None:
    model_directory = tmp_path / "fine_tuned_model"
    if setup == "missing-tokenizer":
        write_fine_tuned_artifact(model_directory)
        (model_directory / "tokenizer_config.json").unlink()
    elif setup == "missing-weights":
        write_fine_tuned_artifact(model_directory)
        (model_directory / "model.safetensors").unlink()
    elif setup == "three-labels":
        write_fine_tuned_artifact(model_directory, {"num_labels": 3})
    elif setup == "invalid-json":
        write_fine_tuned_artifact(model_directory)
        (model_directory / "config.json").write_text("not json", encoding="utf-8")

    with pytest.raises(error_type, match=message):
        get_day6_artifact_inputs(model_directory)


def test_recreate_frozen_baseline_receives_only_outer_training_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison_dataset = prepare_comparison_dataset(make_sentiment_dataframe())
    received: dict[str, object] = {}
    tokenizer = cast(PreTrainedTokenizerBase, object())
    encoder = cast(PreTrainedModel, object())
    classifier = cast(object, object())

    def fake_prepare(
        dataframe: pd.DataFrame,
        received_tokenizer: PreTrainedTokenizerBase,
        received_encoder: PreTrainedModel,
        batch_size: int,
    ) -> FrozenEmbeddingDataset:
        received["dataframe"] = dataframe
        received["tokenizer"] = received_tokenizer
        received["encoder"] = received_encoder
        received["batch_size"] = batch_size
        return FrozenEmbeddingDataset(
            features=np.array([[1.0], [2.0]], dtype=float),
            labels=np.array([0, 1], dtype=np.int64),
        )

    def fake_train(dataset: FrozenEmbeddingDataset) -> object:
        received["training_dataset"] = dataset
        return classifier

    monkeypatch.setattr(comparison, "prepare_frozen_embedding_dataset", fake_prepare)
    monkeypatch.setattr(
        comparison, "train_logistic_regression_on_frozen_embeddings", fake_train
    )

    setup = recreate_frozen_baseline(
        comparison_dataset,
        tokenizer,
        encoder,
        batch_size=7,
    )

    assert cast(pd.DataFrame, received["dataframe"]).equals(
        comparison_dataset.outer_train
    )
    assert received["tokenizer"] is tokenizer
    assert received["encoder"] is encoder
    assert received["batch_size"] == 7
    assert setup.classifier is classifier
    assert setup.tokenizer is tokenizer
    assert setup.encoder is encoder


def test_load_fine_tuned_inference_uses_local_binary_artifact_and_eval_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_directory = tmp_path / "fine_tuned_model"
    write_fine_tuned_artifact(model_directory)
    received: dict[str, object] = {}

    class FakeTokenizer:
        pass

    class FakeModel:
        def __init__(self) -> None:
            self.config = type("Config", (), {"num_labels": 2})()
            self.device: torch.device | None = None
            self.eval_called = False

        def to(self, device: torch.device) -> "FakeModel":
            self.device = device
            return self

        def eval(self) -> "FakeModel":
            self.eval_called = True
            return self

    fake_tokenizer = FakeTokenizer()
    fake_model = FakeModel()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(source: str, **options: object) -> FakeTokenizer:
            received["tokenizer_source"] = source
            received["tokenizer_options"] = options
            return fake_tokenizer

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(source: str, **options: object) -> FakeModel:
            received["model_source"] = source
            received["model_options"] = options
            return fake_model

    monkeypatch.setattr(comparison, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(comparison, "AutoModelForSequenceClassification", FakeAutoModel)

    setup = load_fine_tuned_inference(
        Day6ArtifactInputs(model_directory),
        device=torch.device("cpu"),
    )

    assert received == {
        "tokenizer_source": str(model_directory),
        "tokenizer_options": {"local_files_only": True},
        "model_source": str(model_directory),
        "model_options": {"local_files_only": True},
    }
    assert fake_model.device == torch.device("cpu")
    assert fake_model.eval_called is True
    assert setup.model is fake_model
    assert setup.tokenizer is fake_tokenizer
    assert setup.device == torch.device("cpu")


def test_predict_fine_tuned_batches_preserves_order_and_disables_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenization_batches: list[tuple[str, ...]] = []
    moved_devices: list[torch.device] = []
    model_input_shapes: list[tuple[int, ...]] = []
    gradient_states: list[bool] = []
    text_ids = {"one": 0, "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}

    class FakeEncoding(dict[str, torch.Tensor]):
        def to(self, device: torch.device) -> "FakeEncoding":
            moved_devices.append(device)
            return self

    def fake_tokenize(
        texts: tuple[str, ...],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> FakeEncoding:
        del tokenizer
        assert max_length == 128
        tokenization_batches.append(texts)
        input_ids = torch.tensor([[text_ids[text]] for text in texts])
        return FakeEncoding(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
        )

    class FakeModel:
        def __init__(self) -> None:
            self.eval_calls = 0

        def eval(self) -> "FakeModel":
            self.eval_calls += 1
            return self

        def __call__(self, **encoding: torch.Tensor) -> SimpleNamespace:
            input_ids = encoding["input_ids"]
            model_input_shapes.append(tuple(input_ids.shape))
            gradient_states.append(torch.is_grad_enabled())
            first_logit = input_ids[:, 0].to(dtype=torch.float32)
            return SimpleNamespace(logits=torch.stack((first_logit, -first_logit), dim=1))

    fake_model = FakeModel()
    setup = FineTunedInferenceSetup(
        model=cast(PreTrainedModel, fake_model),
        tokenizer=cast(PreTrainedTokenizerBase, object()),
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(comparison, "tokenize_texts", fake_tokenize)

    scalar_prediction = predict_fine_tuned("one", setup, batch_size=2)
    predictions = predict_fine_tuned(
        ["first", "second", "third", "fourth", "fifth"],
        setup,
        batch_size=2,
    )

    assert [record.text for record in scalar_prediction] == ["one"]
    assert [record.text for record in predictions] == [
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
    ]
    assert [record.prediction for record in predictions] == [0, 0, 0, 0, 0]
    assert tokenization_batches == [
        ("one",),
        ("first", "second"),
        ("third", "fourth"),
        ("fifth",),
    ]
    assert model_input_shapes == [(1, 1), (2, 1), (2, 1), (1, 1)]
    assert moved_devices == [torch.device("cpu")] * 4
    assert gradient_states == [False] * 4
    assert fake_model.eval_calls == 2
    for record in (*scalar_prediction, *predictions):
        assert record.probabilities.shape == (2,)
        assert np.isfinite(record.probabilities).all()
        assert np.isclose(record.probabilities.sum(), 1.0)


@pytest.mark.parametrize(
    ("texts", "logits", "message"),
    (
        ([], torch.zeros((1, 2)), "at least one"),
        (["Text"], torch.zeros((1, 3)), "shape"),
        (["Text"], torch.tensor([[float("nan"), 0.0]]), "finite"),
    ),
)
def test_predict_fine_tuned_rejects_empty_input_and_malformed_logits(
    monkeypatch: pytest.MonkeyPatch,
    texts: list[str],
    logits: torch.Tensor,
    message: str,
) -> None:
    class FakeEncoding(dict[str, torch.Tensor]):
        def to(self, device: torch.device) -> "FakeEncoding":
            return self

    class FakeModel:
        def eval(self) -> "FakeModel":
            return self

        def __call__(self, **encoding: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(logits=logits)

    setup = FineTunedInferenceSetup(
        model=cast(PreTrainedModel, FakeModel()),
        tokenizer=cast(PreTrainedTokenizerBase, object()),
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(
        comparison,
        "tokenize_texts",
        lambda *args, **kwargs: FakeEncoding(
            input_ids=torch.ones((1, 1), dtype=torch.long),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
        ),
    )

    with pytest.raises(ValueError, match=message):
        predict_fine_tuned(texts, setup)


def test_predict_baseline_preserves_order_and_reorders_probability_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, object] = {}
    tokenizer = cast(PreTrainedTokenizerBase, object())
    encoder = cast(PreTrainedModel, object())

    class FakeClassifier:
        classes_ = np.array([1, 0], dtype=np.int64)

        def predict(self, features: np.ndarray) -> np.ndarray:
            received["prediction_features"] = features
            return np.array([1, 0, 1], dtype=np.int64)

        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            received["probability_features"] = features
            return np.array(
                [[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]],
                dtype=np.float64,
            )

    features = np.array([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]])

    def fake_get_embeddings(
        texts: tuple[str, ...],
        received_tokenizer: PreTrainedTokenizerBase,
        received_encoder: PreTrainedModel,
        batch_size: int,
    ) -> np.ndarray:
        received["texts"] = texts
        received["tokenizer"] = received_tokenizer
        received["encoder"] = received_encoder
        received["batch_size"] = batch_size
        return features

    monkeypatch.setattr(comparison, "get_embeddings", fake_get_embeddings)
    setup = FrozenBaselineSetup(
        classifier=cast(LogisticRegression, FakeClassifier()),
        tokenizer=tokenizer,
        encoder=encoder,
    )

    predictions = predict_baseline(
        ["first", "second", "third"],
        setup,
        batch_size=2,
    )

    assert received["texts"] == ("first", "second", "third")
    assert received["tokenizer"] is tokenizer
    assert received["encoder"] is encoder
    assert received["batch_size"] == 2
    assert received["prediction_features"] is features
    assert received["probability_features"] is features
    assert [record.text for record in predictions] == ["first", "second", "third"]
    assert [record.prediction for record in predictions] == [1, 0, 1]
    assert [record.probabilities.tolist() for record in predictions] == [
        [0.8, 0.2],
        [0.3, 0.7],
        [0.6, 0.4],
    ]


@pytest.mark.parametrize(
    ("features", "probabilities", "message"),
    (
        (np.array([[1.0, 2.0]]), np.array([[0.5, 0.5]]), "shape"),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[float("nan"), 0.5], [0.5, 0.5]]),
            "finite",
        ),
    ),
)
def test_predict_baseline_rejects_invalid_features_and_probabilities(
    monkeypatch: pytest.MonkeyPatch,
    features: np.ndarray,
    probabilities: np.ndarray,
    message: str,
) -> None:
    class FakeClassifier:
        classes_ = np.array([0, 1], dtype=np.int64)

        def predict(self, embeddings: np.ndarray) -> np.ndarray:
            return np.zeros(embeddings.shape[0], dtype=np.int64)

        def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
            return probabilities

    monkeypatch.setattr(comparison, "get_embeddings", lambda *args, **kwargs: features)
    setup = FrozenBaselineSetup(
        classifier=cast(LogisticRegression, FakeClassifier()),
        tokenizer=cast(PreTrainedTokenizerBase, object()),
        encoder=cast(PreTrainedModel, object()),
    )

    with pytest.raises(ValueError, match=message):
        predict_baseline(["first", "second"], setup)


def test_compare_five_examples_builds_aligned_display_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, object] = {}
    fine_tuned_setup = cast(FineTunedInferenceSetup, object())
    baseline_setup = cast(FrozenBaselineSetup, object())

    def fake_fine_tuned(
        texts: tuple[str, ...],
        setup: FineTunedInferenceSetup,
        batch_size: int,
    ) -> tuple[SentimentPrediction, ...]:
        received["fine_tuned"] = (texts, setup, batch_size)
        return tuple(
            SentimentPrediction(
                text=text,
                prediction=index % 2,
                probabilities=np.array([0.8, 0.2]),
            )
            for index, text in enumerate(texts)
        )

    def fake_baseline(
        texts: tuple[str, ...],
        setup: FrozenBaselineSetup,
        batch_size: int,
    ) -> tuple[SentimentPrediction, ...]:
        received["baseline"] = (texts, setup, batch_size)
        return tuple(
            SentimentPrediction(
                text=text,
                prediction=0 if index != 1 else 1,
                probabilities=np.array([0.3, 0.7]),
            )
            for index, text in enumerate(texts)
        )

    monkeypatch.setattr(comparison, "predict_fine_tuned", fake_fine_tuned)
    monkeypatch.setattr(comparison, "predict_baseline", fake_baseline)

    comparisons = compare_five_examples(
        fine_tuned_setup,
        baseline_setup,
        batch_size=3,
    )
    table = build_example_comparison_table(comparisons)

    assert received == {
        "fine_tuned": (DAY6_EXAMPLE_TEXTS, fine_tuned_setup, 3),
        "baseline": (DAY6_EXAMPLE_TEXTS, baseline_setup, 3),
    }
    assert [comparison.text for comparison in comparisons] == list(DAY6_EXAMPLE_TEXTS)
    assert [comparison.predictions_agree for comparison in comparisons] == [True, True, True, False, True]
    assert table.columns.tolist() == [
        "text",
        "fine_tuned_prediction",
        "fine_tuned_probabilities",
        "baseline_prediction",
        "baseline_probabilities",
        "predictions_agree",
    ]
    assert table["text"].tolist() == list(DAY6_EXAMPLE_TEXTS)
    assert table["fine_tuned_probabilities"].tolist() == [(0.8, 0.2)] * 5
    assert table["baseline_probabilities"].tolist() == [(0.3, 0.7)] * 5


def test_compare_example_predictions_rejects_misaligned_model_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fine_tuned(*args: object, **kwargs: object) -> tuple[SentimentPrediction, ...]:
        return (
            SentimentPrediction(
                text="wrong text",
                prediction=0,
                probabilities=np.array([0.5, 0.5]),
            ),
        )

    def fake_baseline(*args: object, **kwargs: object) -> tuple[SentimentPrediction, ...]:
        return (
            SentimentPrediction(
                text="expected text",
                prediction=0,
                probabilities=np.array([0.5, 0.5]),
            ),
        )

    monkeypatch.setattr(comparison, "predict_fine_tuned", fake_fine_tuned)
    monkeypatch.setattr(comparison, "predict_baseline", fake_baseline)

    with pytest.raises(ValueError, match="preserve input order"):
        compare_example_predictions(
            "expected text",
            cast(FineTunedInferenceSetup, object()),
            cast(FrozenBaselineSetup, object()),
        )
