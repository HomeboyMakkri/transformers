import math
from typing import cast

import pytest
import torch
from transformers import (
    AutoModel,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.modeling_outputs import BaseModelOutput

from transformers_learning.modeling import (
    extract_first_token_representation,
    get_embeddings,
    load_model,
    similarity,
)


class FakeModel:
    def __init__(self) -> None:
        self.training = True

    def eval(self) -> "FakeModel":
        self.training = False
        return self


def test_load_model_uses_requested_checkpoint_and_enables_eval_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = FakeModel()
    received_calls: list[tuple[str, bool]] = []

    def fake_from_pretrained(
        model_name: str,
        *,
        output_attentions: bool,
    ) -> PreTrainedModel:
        received_calls.append((model_name, output_attentions))
        return cast(PreTrainedModel, fake_model)

    monkeypatch.setattr(AutoModel, "from_pretrained", fake_from_pretrained)

    model = load_model("example-checkpoint")

    assert received_calls == [("example-checkpoint", False)]
    assert model is fake_model
    assert model.training is False


def test_load_model_requests_attention_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = FakeModel()
    received_output_attentions: list[bool] = []

    def fake_from_pretrained(
        model_name: str,
        *,
        output_attentions: bool,
    ) -> PreTrainedModel:
        assert model_name == "example-checkpoint"
        received_output_attentions.append(output_attentions)
        return cast(PreTrainedModel, fake_model)

    monkeypatch.setattr(AutoModel, "from_pretrained", fake_from_pretrained)

    model = load_model("example-checkpoint", output_attentions=True)

    assert received_output_attentions == [True]
    assert model is fake_model
    assert model.training is False


def test_extract_first_token_representation_preserves_batch_and_hidden_axes() -> None:
    hidden_states = torch.arange(24).reshape(2, 3, 4)

    first_token = extract_first_token_representation(hidden_states)

    assert first_token.shape == (2, 4)
    assert torch.equal(first_token, hidden_states[:, 0, :])


@pytest.mark.parametrize(
    "hidden_states",
    (
        torch.zeros(2, 4),
        torch.zeros(2, 0, 4),
    ),
)
def test_extract_first_token_representation_rejects_invalid_shapes(
    hidden_states: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        extract_first_token_representation(hidden_states)


class FakeEmbeddingTokenizer:
    def __init__(self) -> None:
        self.token_ids = {
            "Text C": 30,
            "Text A": 10,
            "Text B": 20,
            "Text D": 40,
            "Text E": 50,
        }
        self.received_batches: list[list[str]] = []

    def __call__(
        self,
        texts: list[str],
        **options: object,
    ) -> BatchEncoding:
        assert options == {
            "padding": True,
            "truncation": True,
            "max_length": 128,
            "return_tensors": "pt",
        }
        self.received_batches.append(texts)
        first_ids = [self.token_ids[text] for text in texts]
        return BatchEncoding(
            {
                "input_ids": torch.tensor([[token_id, 1] for token_id in first_ids]),
                "attention_mask": torch.ones(len(texts), 2, dtype=torch.int64),
            }
        )


class FakeEmbeddingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.grad_modes: list[bool] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> BaseModelOutput:
        assert input_ids.device == self.anchor.device
        assert attention_mask.device == self.anchor.device
        self.grad_modes.append(torch.is_grad_enabled())
        hidden_states = input_ids.unsqueeze(-1).repeat(1, 1, 3).float()
        return BaseModelOutput(
            last_hidden_state=cast(torch.FloatTensor, hidden_states)
        )


def make_embedding_dependencies() -> tuple[
    PreTrainedTokenizerBase,
    FakeEmbeddingTokenizer,
    PreTrainedModel,
    FakeEmbeddingModel,
]:
    tokenizer = FakeEmbeddingTokenizer()
    model = FakeEmbeddingModel()
    return (
        cast(PreTrainedTokenizerBase, tokenizer),
        tokenizer,
        cast(PreTrainedModel, model),
        model,
    )


def test_get_embeddings_preserves_order_and_processes_partial_final_batch() -> None:
    tokenizer, fake_tokenizer, model, fake_model = make_embedding_dependencies()
    texts = ("Text C", "Text A", "Text B", "Text D", "Text E")

    embeddings = get_embeddings(texts, tokenizer, model, batch_size=2)

    assert fake_tokenizer.received_batches == [
        ["Text C", "Text A"],
        ["Text B", "Text D"],
        ["Text E"],
    ]
    assert embeddings.shape == (5, 3)
    assert embeddings[:, 0].tolist() == [30.0, 10.0, 20.0, 40.0, 50.0]
    assert fake_model.grad_modes == [False, False, False]
    assert model.training is False


@pytest.mark.parametrize(
    ("texts", "batch_size", "message"),
    (
        ([], 2, "texts must contain at least one item"),
        (["Text A"], 0, "batch_size must be at least 1"),
    ),
)
def test_get_embeddings_rejects_invalid_input(
    texts: list[str],
    batch_size: int,
    message: str,
) -> None:
    tokenizer, _, model, _ = make_embedding_dependencies()

    with pytest.raises(ValueError, match=message):
        get_embeddings(texts, tokenizer, model, batch_size=batch_size)


def test_similarity_returns_finite_bounded_float_for_two_texts() -> None:
    tokenizer, fake_tokenizer, model, _ = make_embedding_dependencies()

    score = similarity("Text A", "Text B", tokenizer, model)

    assert fake_tokenizer.received_batches == [["Text A", "Text B"]]
    assert isinstance(score, float)
    assert math.isfinite(score)
    assert -1.0 <= score <= 1.0
