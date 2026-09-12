from typing import cast

import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase

from transformers_learning.tokenization import (
    TokenizerInfo,
    explain_tokenization,
    inspect_tokenizer,
    tokenize_single_text,
    tokenize_texts,
)


class FakeTokenizer:
    vocab_size = 5
    model_max_length = 8
    cls_token = "[CLS]"
    cls_token_id = 1
    sep_token = "[SEP]"
    sep_token_id = 2
    pad_token = "[PAD]"
    pad_token_id = 0
    unk_token = "[UNK]"
    unk_token_id = 3

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
        assert kwargs == {"add_special_tokens": True}
        assert text == "A movie"
        return {"input_ids": [1, 4, 2], "attention_mask": [1, 1, 1]}

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        assert ids == [1, 4, 2]
        return ["[CLS]", "movie", "[SEP]"]

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        assert token_ids == [1, 4, 2]
        return "[CLS] movie [SEP]"


def make_fake_tokenizer() -> PreTrainedTokenizerBase:
    """Expose the small test double through the production dependency type."""

    return cast(PreTrainedTokenizerBase, FakeTokenizer())


def test_inspect_tokenizer_collects_day1_properties() -> None:
    info = inspect_tokenizer(make_fake_tokenizer())

    assert info == TokenizerInfo(
        vocabulary_size=5,
        model_max_length=8,
        special_tokens={
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
        },
        special_token_ids={
            "cls_token": 1,
            "sep_token": 2,
            "pad_token": 0,
            "unk_token": 3,
        },
    )


def test_tokenize_single_text_exposes_the_encode_decode_path() -> None:
    walkthrough = tokenize_single_text("A movie", make_fake_tokenizer())

    assert walkthrough.text == "A movie"
    assert walkthrough.tokens == ("[CLS]", "movie", "[SEP]")
    assert walkthrough.input_ids == (1, 4, 2)
    assert walkthrough.attention_mask == (1, 1, 1)
    assert walkthrough.decoded_text == "[CLS] movie [SEP]"


class FakeBatchTokenizer:
    received_texts: list[str] | None = None
    received_options: dict[str, object] | None = None

    def __call__(
        self,
        texts: list[str],
        **options: object,
    ) -> BatchEncoding:
        self.received_texts = texts
        self.received_options = options
        return BatchEncoding(
            {
                "input_ids": torch.tensor([[1, 4, 2, 0], [1, 4, 4, 2]]),
                "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
            }
        )


def make_fake_batch_tokenizer() -> tuple[PreTrainedTokenizerBase, FakeBatchTokenizer]:
    fake = FakeBatchTokenizer()
    return cast(PreTrainedTokenizerBase, fake), fake


def test_tokenize_texts_returns_aligned_padded_tensors() -> None:
    tokenizer, fake = make_fake_batch_tokenizer()

    encoded = tokenize_texts(("Short", "A longer text"), tokenizer, max_length=4)

    assert fake.received_texts == ["Short", "A longer text"]
    assert fake.received_options == {
        "padding": True,
        "truncation": True,
        "max_length": 4,
        "return_tensors": "pt",
    }
    assert encoded["input_ids"].shape == (2, 4)
    assert encoded["attention_mask"].shape == encoded["input_ids"].shape
    assert encoded["attention_mask"][0].tolist() == [1, 1, 1, 0]


def test_tokenize_texts_rejects_an_empty_batch() -> None:
    tokenizer, _ = make_fake_batch_tokenizer()

    try:
        tokenize_texts([], tokenizer)
    except ValueError as error:
        assert str(error) == "texts must contain at least one item"
    else:
        raise AssertionError("An empty text batch must be rejected")


def test_explain_tokenization_describes_ids_mask_and_decoding() -> None:
    explanation = explain_tokenization("A movie", make_fake_tokenizer())

    assert "Tokens: ['[CLS]', 'movie', '[SEP]']" in explanation
    assert "Input IDs: [1, 4, 2]" in explanation
    assert "Attention mask: [1, 1, 1]" in explanation
    assert "Token count: 3" in explanation
    assert "Decoded text: [CLS] movie [SEP]" in explanation
