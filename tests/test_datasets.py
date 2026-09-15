from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from zipfile import ZipFile

import pandas as pd
import pytest
from typing_extensions import Self

from transformers_learning import datasets
from transformers_learning.datasets import (
    LABEL_COLUMN,
    SST2_METADATA,
    TEXT_COLUMN,
    SentimentDatasetValidationError,
    SST2DownloadError,
    adapt_sst2_split,
    ensure_sst2_train_data,
    get_class_counts,
    validate_sentiment_dataframe,
)


class FakeDownloadResponse(BytesIO):
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def make_sst2_archive() -> bytes:
    archive_buffer = BytesIO()
    with ZipFile(archive_buffer, "w") as archive:
        archive.writestr("SST-2/train.tsv", "sentence\tlabel\nGood\t1\nBad\t0\n")
    return archive_buffer.getvalue()


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


def test_ensure_sst2_train_data_reuses_an_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_path = tmp_path / "SST-2" / "train.tsv"
    train_path.parent.mkdir()
    train_path.write_text("existing data", encoding="utf-8")

    def fail_if_called(*args: object, **kwargs: object) -> FakeDownloadResponse:
        raise AssertionError("Existing SST-2 data must not trigger a download")

    monkeypatch.setattr(datasets, "urlopen", fail_if_called)

    assert ensure_sst2_train_data(train_path) == train_path
    assert train_path.read_text(encoding="utf-8") == "existing data"


def test_ensure_sst2_train_data_downloads_only_the_training_tsv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, object] = {}

    def fake_urlopen(url: str, *, timeout: int) -> FakeDownloadResponse:
        received["url"] = url
        received["timeout"] = timeout
        return FakeDownloadResponse(make_sst2_archive())

    monkeypatch.setattr(datasets, "urlopen", fake_urlopen)
    train_path = tmp_path / "data" / "SST-2" / "train.tsv"

    assert ensure_sst2_train_data(train_path) == train_path
    assert received == {
        "url": "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip",
        "timeout": 30,
    }
    assert train_path.read_text(encoding="utf-8") == "sentence\tlabel\nGood\t1\nBad\t0\n"
    assert not (train_path.parent / "SST-2.zip").exists()


def test_ensure_sst2_train_data_reports_download_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_download(url: str, *, timeout: int) -> FakeDownloadResponse:
        raise URLError("network unavailable")

    monkeypatch.setattr(datasets, "urlopen", fail_download)

    with pytest.raises(SST2DownloadError, match="Could not download"):
        ensure_sst2_train_data(tmp_path / "SST-2" / "train.tsv")
