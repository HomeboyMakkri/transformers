"""Model loading and hidden-state utilities for Day 2."""

import torch
from transformers import AutoModel, PreTrainedModel

from .tokenization import DEFAULT_MODEL_NAME


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
) -> PreTrainedModel:
    """Load the model matching the tokenizer and prepare it for inference."""

    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return model


def extract_first_token_representation(
    last_hidden_state: torch.Tensor,
) -> torch.Tensor:
    """Extract one contextual first-token vector per batch item."""

    if last_hidden_state.ndim != 3:
        raise ValueError(
            "last_hidden_state must have shape [batch, sequence, hidden]"
        )
    if last_hidden_state.shape[1] == 0:
        raise ValueError("last_hidden_state must contain at least one token")

    return last_hidden_state[:, 0, :]
