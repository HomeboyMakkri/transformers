"""Model loading and inference utilities for the learning project."""

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import BaseModelOutput

from .tokenization import DEFAULT_MODEL_NAME, tokenize_texts

ProgressCallback = Callable[[int, int], None]


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    output_attentions: bool = False,
) -> PreTrainedModel:
    """Load a matching model and prepare it for inference."""

    model = AutoModel.from_pretrained(
        model_name,
        output_attentions=output_attentions,
    )
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


def get_embeddings(
    texts: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    batch_size: int = 32,
    progress_callback: ProgressCallback | None = None,
) -> npt.NDArray[np.floating[Any]]:
    """Return first-token representations in input order using small batches.

    When supplied, ``progress_callback`` receives completed and total text
    counts after every completed batch.
    """

    if not texts:
        raise ValueError("texts must contain at least one item")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    model.eval()
    device = next(model.parameters()).device
    all_embeddings: list[npt.NDArray[np.floating[Any]]] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        tokens = tokenize_texts(batch_texts, tokenizer, max_length=128)
        tokens = tokens.to(device)

        with torch.no_grad():
            outputs = cast(BaseModelOutput, model(**tokens))

        if outputs.last_hidden_state is None:
            raise RuntimeError("model output does not contain last_hidden_state")

        first_token = extract_first_token_representation(outputs.last_hidden_state)
        compact_first_token = first_token.detach().contiguous().cpu()
        all_embeddings.append(compact_first_token.numpy())
        if progress_callback is not None:
            progress_callback(start + len(batch_texts), len(texts))

    return np.vstack(all_embeddings)


def similarity(
    text1: str,
    text2: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
) -> float:
    """Return bounded cosine similarity between two first-token representations."""

    embeddings = get_embeddings((text1, text2), tokenizer, model)
    score = cosine_similarity(embeddings[0:1], embeddings[1:2])[0, 0]
    return float(np.clip(score, -1.0, 1.0))
