from pathlib import Path
from typing import cast

import pytest
import torch
from transformers import PreTrainedTokenizerBase

from transformers_learning.attention import (
    extract_attention_matrix,
    visualize_attention,
)


class FakeTokenizer:
    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [f"token-{token_id}" for token_id in token_ids]


def test_extract_attention_matrix_selects_first_item_and_head() -> None:
    attentions = (torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2),)

    matrix = extract_attention_matrix(attentions, layer=0, head=2)

    assert matrix.shape == (2, 2)
    assert torch.equal(matrix, attentions[0][0, 2])


@pytest.mark.parametrize(
    ("layer", "head"),
    ((1, 0), (0, 3)),
)
def test_extract_attention_matrix_rejects_invalid_indices(
    layer: int,
    head: int,
) -> None:
    attentions = (torch.zeros(1, 3, 2, 2),)

    with pytest.raises(IndexError):
        extract_attention_matrix(attentions, layer=layer, head=head)


def test_extract_attention_matrix_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError):
        extract_attention_matrix((torch.zeros(1, 2, 2),))


@pytest.mark.parametrize(
    "attention",
    (
        torch.zeros(0, 2, 3, 3),
        torch.zeros(1, 2, 3, 4),
    ),
)
def test_extract_attention_matrix_rejects_invalid_dimensions(
    attention: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        extract_attention_matrix((attention,))


def test_visualize_attention_saves_token_labelled_heatmap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "attention.png"
    tokenizer = FakeTokenizer()
    input_ids = torch.tensor([[101, 2023, 102]])
    attentions = (torch.full((1, 2, 3, 3), 1 / 3),)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    figure = visualize_attention(
        input_ids,
        attentions,
        cast(PreTrainedTokenizerBase, tokenizer),
        layer=0,
        head=1,
        output_path=output_path,
    )

    assert output_path.is_file()
    assert figure.axes[0].get_xlabel() == "Keys (tokens being attended to)"
    assert figure.axes[0].get_ylabel() == "Queries (tokens looking for context)"
