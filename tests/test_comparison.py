import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from transformers_learning import comparison
from transformers_learning.baseline import FrozenEmbeddingDataset
from transformers_learning.comparison import (
    Day6ArtifactInputs,
    FineTunedArtifactError,
    get_day6_artifact_inputs,
    load_fine_tuned_inference,
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
