import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transformers_learning.comparison import (
    FineTunedArtifactError,
    get_day6_artifact_inputs,
    prepare_comparison_dataset,
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
