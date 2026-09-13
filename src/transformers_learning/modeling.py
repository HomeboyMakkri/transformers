"""Base-model loading utilities for Day 2."""

from transformers import AutoModel, PreTrainedModel

from .tokenization import DEFAULT_MODEL_NAME


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
) -> PreTrainedModel:
    """Load the model matching the tokenizer and prepare it for inference."""

    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return model
