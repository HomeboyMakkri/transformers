"""Tokenizer loading and inspection utilities for Day 1."""

from dataclasses import dataclass

from transformers import AutoTokenizer, PreTrainedTokenizerBase

DEFAULT_MODEL_NAME = "distilbert-base-uncased"


@dataclass(frozen=True)
class TokenizerInfo:
    """The tokenizer properties inspected in the Day 1 walkthrough."""

    vocabulary_size: int
    model_max_length: int
    special_tokens: dict[str, str]
    special_token_ids: dict[str, int]


@dataclass(frozen=True)
class TokenizationWalkthrough:
    """Observable stages of tokenizing and decoding one text."""

    text: str
    tokens: tuple[str, ...]
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    decoded_text: str


def load_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
) -> PreTrainedTokenizerBase:
    """Load the tokenizer shared by the Day 1 and Day 2 exercises."""

    return AutoTokenizer.from_pretrained(model_name)


def inspect_tokenizer(tokenizer: PreTrainedTokenizerBase) -> TokenizerInfo:
    """Collect vocabulary, context-window, and special-token metadata."""

    token_names = ("cls_token", "sep_token", "pad_token", "unk_token")
    special_tokens: dict[str, str] = {}
    special_token_ids: dict[str, int] = {}

    for name in token_names:
        token = getattr(tokenizer, name)
        token_id = getattr(tokenizer, f"{name}_id")
        if isinstance(token, str) and isinstance(token_id, int):
            special_tokens[name] = token
            special_token_ids[name] = token_id

    return TokenizerInfo(
        vocabulary_size=tokenizer.vocab_size,
        model_max_length=tokenizer.model_max_length,
        special_tokens=special_tokens,
        special_token_ids=special_token_ids,
    )


def tokenize_single_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
) -> TokenizationWalkthrough:
    """Show the encode/decode path for one text, including special tokens."""

    encoded = tokenizer(text, add_special_tokens=True)
    input_ids_list = _require_int_list(encoded["input_ids"], "input_ids")
    attention_mask_list = _require_int_list(
        encoded["attention_mask"],
        "attention_mask",
    )

    converted_tokens = tokenizer.convert_ids_to_tokens(input_ids_list)
    if isinstance(converted_tokens, str):
        tokens = (converted_tokens,)
    else:
        tokens = tuple(converted_tokens)

    decoded_text = tokenizer.decode(input_ids_list)
    if not isinstance(decoded_text, str):
        raise TypeError("Tokenizer returned multiple decoded texts for one input")

    return TokenizationWalkthrough(
        text=text,
        tokens=tokens,
        input_ids=tuple(input_ids_list),
        attention_mask=tuple(attention_mask_list),
        decoded_text=decoded_text,
    )


def _require_int_list(value: object, field_name: str) -> list[int]:
    """Validate a single-text tokenizer field at the third-party boundary."""

    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise TypeError(f"Tokenizer field {field_name!r} must be a list of integers")
    return value
