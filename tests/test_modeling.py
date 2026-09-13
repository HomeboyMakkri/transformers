from typing import cast

import pytest
from transformers import AutoModel, PreTrainedModel

from transformers_learning.modeling import load_model


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
