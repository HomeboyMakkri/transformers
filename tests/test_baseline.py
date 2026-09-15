from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from transformers_learning import baseline
from transformers_learning.baseline import (
    BaselineEvaluation,
    FrozenEmbeddingDataset,
    FrozenEmbeddingSplit,
    evaluate_logistic_regression,
    prepare_frozen_embedding_dataset,
    save_baseline_results,
    split_frozen_embedding_dataset,
    train_logistic_regression,
)
from transformers_learning.datasets import SentimentDatasetValidationError


def test_prepare_frozen_embedding_dataset_preserves_text_label_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, object] = {}

    def fake_get_embeddings(
        texts: tuple[str, ...],
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        batch_size: int = 32,
        progress_callback: object = None,
    ) -> npt.NDArray[np.floating[Any]]:
        received["texts"] = texts
        received["tokenizer"] = tokenizer
        received["model"] = model
        received["batch_size"] = batch_size
        received["progress_callback"] = progress_callback
        return np.array([[30.0, 3.0], [10.0, 1.0], [20.0, 2.0]])

    monkeypatch.setattr(baseline, "get_embeddings", fake_get_embeddings)
    tokenizer = cast(PreTrainedTokenizerBase, object())
    model = cast(PreTrainedModel, object())
    dataframe = pd.DataFrame(
        {"text": ["Third", "First", "Second"], "label": [1, 0, 1]}
    )

    prepared = prepare_frozen_embedding_dataset(
        dataframe, tokenizer, model, batch_size=2
    )

    assert received == {
        "texts": ("Third", "First", "Second"),
        "tokenizer": tokenizer,
        "model": model,
        "batch_size": 2,
        "progress_callback": None,
    }
    assert prepared.features.tolist() == [[30.0, 3.0], [10.0, 1.0], [20.0, 2.0]]
    assert prepared.labels.tolist() == [1, 0, 1]


def test_prepare_frozen_embedding_dataset_validates_data_before_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> npt.NDArray[np.float64]:
        raise AssertionError("Invalid data must not reach the embedding model")

    monkeypatch.setattr(baseline, "get_embeddings", fail_if_called)
    invalid_data = pd.DataFrame({"text": ["Only one class"], "label": [1]})

    with pytest.raises(SentimentDatasetValidationError, match="both SST-2"):
        prepare_frozen_embedding_dataset(
            invalid_data,
            cast(PreTrainedTokenizerBase, object()),
            cast(PreTrainedModel, object()),
        )


@pytest.mark.parametrize(
    ("embeddings", "message"),
    (
        (np.array([1.0, 2.0]), "shape"),
        (np.array([[1.0], [2.0], [3.0]]), "row count"),
        (np.empty((2, 0)), "at least one feature"),
        (np.array([[1.0], [np.nan]]), "finite"),
    ),
)
def test_prepare_frozen_embedding_dataset_rejects_invalid_model_output(
    monkeypatch: pytest.MonkeyPatch,
    embeddings: npt.NDArray[np.floating[Any]],
    message: str,
) -> None:
    monkeypatch.setattr(baseline, "get_embeddings", lambda *args, **kwargs: embeddings)
    dataframe = pd.DataFrame({"text": ["Good", "Bad"], "label": [1, 0]})

    with pytest.raises(ValueError, match=message):
        prepare_frozen_embedding_dataset(
            dataframe,
            cast(PreTrainedTokenizerBase, object()),
            cast(PreTrainedModel, object()),
        )


def test_split_frozen_embedding_dataset_is_stratified_and_reproducible() -> None:
    dataset = FrozenEmbeddingDataset(
        features=np.column_stack((np.arange(10, dtype=float), np.arange(10, dtype=float))),
        labels=np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64),
    )

    first_split = split_frozen_embedding_dataset(dataset)
    second_split = split_frozen_embedding_dataset(dataset)

    assert first_split.X_train.shape == (8, 2)
    assert first_split.X_test.shape == (2, 2)
    assert first_split.y_train.tolist().count(0) == 4
    assert first_split.y_train.tolist().count(1) == 4
    assert first_split.y_test.tolist().count(0) == 1
    assert first_split.y_test.tolist().count(1) == 1
    assert np.array_equal(first_split.X_train, second_split.X_train)
    assert np.array_equal(first_split.X_test, second_split.X_test)
    assert np.array_equal(first_split.y_train, second_split.y_train)
    assert np.array_equal(first_split.y_test, second_split.y_test)
    assert set(first_split.X_train[:, 0]) | set(first_split.X_test[:, 0]) == set(
        np.arange(10, dtype=float)
    )
    assert set(first_split.X_train[:, 0]).isdisjoint(set(first_split.X_test[:, 0]))


@pytest.mark.parametrize(
    ("features", "labels", "message"),
    (
        (np.array([1.0, 2.0]), np.array([0, 1], dtype=np.int64), "shape"),
        (np.array([[1.0], [2.0]]), np.array([[0], [1]], dtype=np.int64), "labels"),
        (np.array([[1.0], [2.0]]), np.array([0, 0], dtype=np.int64), "both SST-2"),
    ),
)
def test_split_frozen_embedding_dataset_rejects_invalid_public_dataset(
    features: npt.NDArray[np.floating[Any]],
    labels: npt.NDArray[np.int64],
    message: str,
) -> None:
    dataset = FrozenEmbeddingDataset(features=features, labels=labels)

    with pytest.raises(ValueError, match=message):
        split_frozen_embedding_dataset(dataset)


def test_train_logistic_regression_fits_only_the_training_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, object] = {}

    class FakeLogisticRegression:
        def __init__(self, *, max_iter: int, n_jobs: int) -> None:
            received["max_iter"] = max_iter
            received["n_jobs"] = n_jobs

        def fit(
            self,
            features: npt.NDArray[np.floating[Any]],
            labels: npt.NDArray[np.int64],
        ) -> "FakeLogisticRegression":
            received["features"] = features
            received["labels"] = labels
            return self

    monkeypatch.setattr(baseline, "LogisticRegression", FakeLogisticRegression)
    split = FrozenEmbeddingSplit(
        X_train=np.array([[1.0, 0.0], [0.0, 1.0], [1.5, 0.0], [0.0, 1.5]]),
        y_train=np.array([0, 1, 0, 1], dtype=np.int64),
        X_test=np.array([[999.0, 999.0], [888.0, 888.0]]),
        y_test=np.array([1, 0], dtype=np.int64),
    )

    classifier = train_logistic_regression(split)

    assert isinstance(classifier, FakeLogisticRegression)
    assert received == {
        "max_iter": 1000,
        "n_jobs": -1,
        "features": split.X_train,
        "labels": split.y_train,
    }


def test_train_logistic_regression_rejects_an_invalid_training_partition() -> None:
    split = FrozenEmbeddingSplit(
        X_train=np.array([[1.0], [2.0]]),
        y_train=np.array([0, 0], dtype=np.int64),
        X_test=np.array([[3.0], [4.0]]),
        y_test=np.array([0, 1], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="both SST-2"):
        train_logistic_regression(split)


def test_evaluate_logistic_regression_uses_only_held_out_features() -> None:
    received: dict[str, object] = {}

    class FakeClassifier:
        def predict(
            self, features: npt.NDArray[np.floating[Any]]
        ) -> npt.NDArray[np.int64]:
            received["features"] = features
            return np.array([0, 1], dtype=np.int64)

    split = FrozenEmbeddingSplit(
        X_train=np.array([[10.0], [20.0], [30.0], [40.0]]),
        y_train=np.array([0, 1, 0, 1], dtype=np.int64),
        X_test=np.array([[101.0], [202.0]]),
        y_test=np.array([0, 1], dtype=np.int64),
    )

    evaluation = evaluate_logistic_regression(
        cast(LogisticRegression, FakeClassifier()), split
    )

    assert received == {"features": split.X_test}
    assert evaluation.predictions.tolist() == [0, 1]
    assert evaluation.macro_f1 == 1.0
    assert "negative" in evaluation.classification_report
    assert "positive" in evaluation.classification_report


@pytest.mark.parametrize(
    ("predictions", "message"),
    (
        (np.array([0], dtype=np.int64), "align"),
        (np.array([0, 2], dtype=np.int64), "only SST-2"),
    ),
)
def test_evaluate_logistic_regression_rejects_invalid_predictions(
    predictions: npt.NDArray[np.int64],
    message: str,
) -> None:
    class FakeClassifier:
        def predict(
            self, features: npt.NDArray[np.floating[Any]]
        ) -> npt.NDArray[np.int64]:
            return predictions

    split = FrozenEmbeddingSplit(
        X_train=np.array([[1.0], [2.0]]),
        y_train=np.array([0, 1], dtype=np.int64),
        X_test=np.array([[3.0], [4.0]]),
        y_test=np.array([0, 1], dtype=np.int64),
    )

    with pytest.raises(ValueError, match=message):
        evaluate_logistic_regression(cast(LogisticRegression, FakeClassifier()), split)


def test_save_baseline_results_writes_only_metric_and_run_context(
    tmp_path: Path,
) -> None:
    evaluation = BaselineEvaluation(
        predictions=np.array([0, 1], dtype=np.int64),
        classification_report="not persisted",
        macro_f1=0.75,
    )
    result_path = tmp_path / "nested" / "baseline_results.txt"

    save_baseline_results(evaluation, result_path, model_name="test-checkpoint")

    assert result_path.read_text(encoding="utf-8") == (
        "macro_f1: 0.750000\n"
        "model_name: test-checkpoint\n"
        "dataset_identifier: stanfordnlp/sst2\n"
        "label_mapping: 0=negative, 1=positive\n"
        "test_size: 0.2\n"
        "random_state: 42\n"
    )
