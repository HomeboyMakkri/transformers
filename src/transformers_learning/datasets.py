"""Dataset metadata and validation for the Day 4 sentiment baseline."""

from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from numbers import Integral
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import pandas as pd

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
SST2_SOURCE_TEXT_COLUMN = "sentence"
SST2_LABEL_MAP: Mapping[int, str] = {0: "negative", 1: "positive"}
SST2_ARCHIVE_URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
SST2_ARCHIVE_TRAIN_PATH = "SST-2/train.tsv"


@dataclass(frozen=True)
class SentimentDatasetMetadata:
    """Document the selected dataset without bundling its files into the repo."""

    identifier: str
    source_url: str
    label_map: Mapping[int, str]
    split_row_counts: Mapping[str, int]
    labelled_split_class_counts: Mapping[str, Mapping[int, int]]
    license_note: str


SST2_METADATA = SentimentDatasetMetadata(
    identifier="stanfordnlp/sst2",
    source_url="https://huggingface.co/datasets/stanfordnlp/sst2",
    label_map=SST2_LABEL_MAP,
    split_row_counts={"train": 67_349, "validation": 872, "test": 1_821},
    labelled_split_class_counts={
        "train": {0: 29_780, 1: 37_569},
        "validation": {0: 428, 1: 444},
    },
    license_note=(
        "The published dataset card lists the license as unknown. Keep a local "
        "copy only for this educational project and do not redistribute it."
    ),
)


class SentimentDatasetValidationError(ValueError):
    """Raised when a table cannot be used as binary sentiment data."""


class SST2DownloadError(RuntimeError):
    """Raised when the official SST-2 archive cannot provide ``train.tsv``."""


def ensure_sst2_train_data(path: Path) -> Path:
    """Return local SST-2 training data, downloading it once when absent.

    The archive is fetched from the URL referenced by the SST-2 dataset card.
    Only its labelled training TSV is stored under the caller-selected ignored
    data directory; the archive itself is never kept in the project.
    """

    if path.is_file():
        return path
    if path.exists():
        raise SST2DownloadError(f"SST-2 training path is not a file: {path}")

    try:
        with urlopen(SST2_ARCHIVE_URL, timeout=30) as response:
            archive_bytes = response.read()
        with ZipFile(BytesIO(archive_bytes)) as archive, archive.open(
            SST2_ARCHIVE_TRAIN_PATH
        ) as source:
            train_data = source.read()
    except (BadZipFile, KeyError, OSError, URLError) as error:
        raise SST2DownloadError("Could not download SST-2 training data") from error

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tsv.tmp")
    temporary_path.write_bytes(train_data)
    temporary_path.replace(path)
    return path


def adapt_sst2_split(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert one labelled SST-2 split from ``sentence,label`` to ``text,label``.

    The official test split has no public labels, so it must not be passed to
    this baseline's supervised training path.
    """

    required_columns = {SST2_SOURCE_TEXT_COLUMN, LABEL_COLUMN}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        formatted_columns = ", ".join(sorted(missing_columns))
        raise SentimentDatasetValidationError(
            f"SST-2 data is missing required columns: {formatted_columns}"
        )

    sentiment_dataframe = dataframe.loc[
        :, [SST2_SOURCE_TEXT_COLUMN, LABEL_COLUMN]
    ].rename(columns={SST2_SOURCE_TEXT_COLUMN: TEXT_COLUMN})
    return validate_sentiment_dataframe(sentiment_dataframe)


def validate_sentiment_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a validated defensive copy with the ``text,label`` contract.

    The baseline needs non-empty English text and both SST-2 classes. Requiring
    both classes here makes a later stratified split meaningful instead of
    failing deep inside scikit-learn.
    """

    required_columns = {TEXT_COLUMN, LABEL_COLUMN}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        formatted_columns = ", ".join(sorted(missing_columns))
        raise SentimentDatasetValidationError(
            f"Sentiment data is missing required columns: {formatted_columns}"
        )

    validated = dataframe.loc[:, [TEXT_COLUMN, LABEL_COLUMN]].copy(deep=True)
    if validated.empty:
        raise SentimentDatasetValidationError("Sentiment data must contain rows")
    if validated[TEXT_COLUMN].isna().any():
        raise SentimentDatasetValidationError("Sentiment text must not be missing")
    if not validated[TEXT_COLUMN].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all():
        raise SentimentDatasetValidationError(
            "Sentiment text must contain non-empty strings"
        )
    if validated[LABEL_COLUMN].isna().any():
        raise SentimentDatasetValidationError("Sentiment labels must not be missing")
    if not validated[LABEL_COLUMN].isin(SST2_LABEL_MAP).all():
        raise SentimentDatasetValidationError(
            "Sentiment labels must be the SST-2 values 0 and 1"
        )

    class_counts = get_class_counts(validated)
    if any(count == 0 for count in class_counts.values()):
        raise SentimentDatasetValidationError(
            "Sentiment data must contain both SST-2 label classes"
        )
    return validated


def get_class_counts(dataframe: pd.DataFrame) -> dict[int, int]:
    """Return counts for both binary labels, including an absent class as zero."""

    label_counts = dataframe[LABEL_COLUMN].value_counts()
    class_counts: dict[int, int] = {}
    for label in SST2_LABEL_MAP:
        count = label_counts.get(label, 0)
        if not isinstance(count, Integral):
            raise TypeError("Label counts must be integers")
        class_counts[label] = int(count)
    return class_counts
