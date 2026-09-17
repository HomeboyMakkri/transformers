from typing import cast

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import BatchEncoding, PreTrainedTokenizerBase

from transformers_learning.datasets import SentimentDatasetValidationError
from transformers_learning.fine_tuning import SentimentDataset


class FakeSentimentTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, text: str, **options: object) -> BatchEncoding:
        self.calls.append((text, options))
        token_value = 11 if text == "Good movie" else 22
        return BatchEncoding(
            {
                "input_ids": torch.tensor([[1, token_value, 2, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 0]]),
            }
        )


def make_fake_tokenizer() -> tuple[PreTrainedTokenizerBase, FakeSentimentTokenizer]:
    fake = FakeSentimentTokenizer()
    return cast(PreTrainedTokenizerBase, fake), fake


def test_sentiment_dataset_tokenizes_lazily_and_returns_one_aligned_item() -> None:
    tokenizer, fake = make_fake_tokenizer()
    dataset = SentimentDataset(
        ["Good movie", "Bad movie"], [1, 0], tokenizer, max_length=4
    )

    assert len(dataset) == 2
    assert fake.calls == []

    item = dataset[1]

    assert fake.calls == [
        (
            "Bad movie",
            {
                "padding": "max_length",
                "truncation": True,
                "max_length": 4,
                "return_tensors": "pt",
            },
        )
    ]
    assert item["input_ids"].tolist() == [1, 22, 2, 0]
    assert item["attention_mask"].tolist() == [1, 1, 1, 0]
    assert item["input_ids"].shape == (4,)
    assert item["attention_mask"].shape == (4,)
    assert item["labels"].shape == ()
    assert item["input_ids"].dtype == torch.long
    assert item["attention_mask"].dtype == torch.long
    assert item["labels"].dtype == torch.long
    assert item["labels"].item() == 0


def test_sentiment_dataset_collates_to_expected_batch_shapes() -> None:
    tokenizer, _ = make_fake_tokenizer()
    dataset = SentimentDataset(
        ["Good movie", "Bad movie"], [1, 0], tokenizer, max_length=4
    )

    batch = next(iter(DataLoader(dataset, batch_size=2)))

    assert batch["input_ids"].shape == (2, 4)
    assert batch["attention_mask"].shape == (2, 4)
    assert batch["labels"].shape == (2,)
    assert batch["labels"].tolist() == [1, 0]


def test_sentiment_dataset_rejects_misaligned_inputs() -> None:
    tokenizer, _ = make_fake_tokenizer()

    with pytest.raises(ValueError, match="same number"):
        SentimentDataset(["Good", "Bad"], [1], tokenizer)


@pytest.mark.parametrize(
    ("texts", "labels", "message"),
    (
        ([], [], "contain rows"),
        (["Good", ""], [1, 0], "non-empty"),
        (["Good", "Bad"], [1, 2], "0 and 1"),
        (["Good", "Also good"], [1, 1], "both SST-2"),
    ),
)
def test_sentiment_dataset_rejects_invalid_examples(
    texts: list[str],
    labels: list[int],
    message: str,
) -> None:
    tokenizer, _ = make_fake_tokenizer()

    with pytest.raises(SentimentDatasetValidationError, match=message):
        SentimentDataset(texts, labels, tokenizer)


@pytest.mark.parametrize("max_length", (0, -1))
def test_sentiment_dataset_rejects_non_positive_max_length(max_length: int) -> None:
    tokenizer, _ = make_fake_tokenizer()

    with pytest.raises(ValueError, match="positive"):
        SentimentDataset(["Good", "Bad"], [1, 0], tokenizer, max_length=max_length)


def test_sentiment_dataset_rejects_malformed_tokenizer_shape() -> None:
    class WrongShapeTokenizer:
        def __call__(self, text: str, **options: object) -> BatchEncoding:
            return BatchEncoding(
                {
                    "input_ids": torch.tensor([[1, 2, 0]]),
                    "attention_mask": torch.tensor([[1, 1, 0]]),
                }
            )

    tokenizer = cast(PreTrainedTokenizerBase, WrongShapeTokenizer())
    dataset = SentimentDataset(["Good", "Bad"], [1, 0], tokenizer, max_length=4)

    with pytest.raises(ValueError, match=r"input_ids.*\[1, max_length\]"):
        dataset[0]
