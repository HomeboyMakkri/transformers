import pandas as pd
import pytest

from transformers_learning.datasets import (
    LABEL_COLUMN,
    SST2_METADATA,
    TEXT_COLUMN,
    SentimentDatasetValidationError,
    adapt_sst2_split,
    get_class_counts,
    validate_sentiment_dataframe,
)


def test_sst2_metadata_documents_the_selected_dataset() -> None:
    assert SST2_METADATA.identifier == "stanfordnlp/sst2"
    assert SST2_METADATA.label_map == {0: "negative", 1: "positive"}
    assert SST2_METADATA.split_row_counts == {
        "train": 67_349,
        "validation": 872,
        "test": 1_821,
    }
    assert SST2_METADATA.labelled_split_class_counts == {
        "train": {0: 29_780, 1: 37_569},
        "validation": {0: 428, 1: 444},
    }
    assert "unknown" in SST2_METADATA.license_note


def test_adapt_sst2_split_renames_sentence_and_discards_source_index() -> None:
    source = pd.DataFrame(
        {"idx": [11, 12], "sentence": ["Very good.", "Very bad."], "label": [1, 0]}
    )

    adapted = adapt_sst2_split(source)

    assert adapted.to_dict("list") == {
        TEXT_COLUMN: ["Very good.", "Very bad."],
        LABEL_COLUMN: [1, 0],
    }
    assert get_class_counts(adapted) == {0: 1, 1: 1}


@pytest.mark.parametrize(
    ("dataframe", "message"),
    (
        (pd.DataFrame({"text": ["Good"], "label": [1]}), "both SST-2"),
        (pd.DataFrame({"text": ["", "Bad"], "label": [1, 0]}), "non-empty"),
        (pd.DataFrame({"text": ["Good", "Bad"], "label": [2, 0]}), "0 and 1"),
        (pd.DataFrame({"text": ["Good", "Bad"]}), "required columns: label"),
    ),
)
def test_validate_sentiment_dataframe_rejects_invalid_values(
    dataframe: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(SentimentDatasetValidationError, match=message):
        validate_sentiment_dataframe(dataframe)


def test_validate_sentiment_dataframe_returns_a_defensive_copy() -> None:
    source = pd.DataFrame({"text": ["Good", "Bad"], "label": [1, 0]})

    validated = validate_sentiment_dataframe(source)
    source.loc[0, "text"] = "Changed after validation"

    assert validated.to_dict("list") == {
        TEXT_COLUMN: ["Good", "Bad"],
        LABEL_COLUMN: [1, 0],
    }


def test_adapt_sst2_split_rejects_the_unlabelled_test_shape() -> None:
    source = pd.DataFrame({"sentence": ["A review without a public label"]})

    with pytest.raises(SentimentDatasetValidationError, match="label"):
        adapt_sst2_split(source)
