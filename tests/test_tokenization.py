from typing import cast

from transformers import PreTrainedTokenizerBase

from transformers_learning.tokenization import (
    TokenizerInfo,
    inspect_tokenizer,
    tokenize_single_text,
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
