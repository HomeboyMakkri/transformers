from typing import cast

import pytest
import torch
from transformers import AutoModel, PreTrainedModel

from transformers_learning.modeling import (
    extract_first_token_representation,
    load_model,
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
    requested_names: list[str] = []

    def fake_from_pretrained(model_name: str) -> PreTrainedModel:
        requested_names.append(model_name)
        return cast(PreTrainedModel, fake_model)

    monkeypatch.setattr(AutoModel, "from_pretrained", fake_from_pretrained)

    model = load_model("example-checkpoint")

    assert requested_names == ["example-checkpoint"]
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
