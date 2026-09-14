"""Attention inspection and visualization utilities for Day 3."""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib.figure import Figure
from transformers import PreTrainedTokenizerBase


def extract_attention_matrix(
    attentions: Sequence[torch.Tensor],
    layer: int = 0,
    head: int = 0,
) -> torch.Tensor:
    """Return one ``[sequence, sequence]`` attention matrix."""

    if not attentions:
        raise ValueError("attentions must contain at least one layer")
    if not 0 <= layer < len(attentions):
        raise IndexError(f"layer must be in [0, {len(attentions) - 1}]")

    layer_attention = attentions[layer]
    if layer_attention.ndim != 4:
        raise ValueError(
            "attention must have shape [batch, heads, sequence, sequence]"
        )
    if layer_attention.shape[0] == 0:
        raise ValueError("attention batch must contain at least one item")
    if layer_attention.shape[2] != layer_attention.shape[3]:
        raise ValueError("attention query and key dimensions must be equal")
    if not 0 <= head < layer_attention.shape[1]:
        raise IndexError(f"head must be in [0, {layer_attention.shape[1] - 1}]")

    return layer_attention[0, head]


def visualize_attention(
    input_ids: torch.Tensor,
    attentions: Sequence[torch.Tensor],
    tokenizer: PreTrainedTokenizerBase,
    layer: int = 0,
    head: int = 0,
    output_path: str | Path | None = None,
) -> Figure:
    """Render the first batch item's selected attention matrix."""

    attention_matrix = extract_attention_matrix(attentions, layer, head)
    if input_ids.ndim != 2 or input_ids.shape[0] == 0:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if input_ids.shape[1] != attention_matrix.shape[0]:
        raise ValueError("input_ids and attention must have the same sequence length")

    token_ids = input_ids[0].detach().cpu().tolist()
    token_list = cast(list[str], tokenizer.convert_ids_to_tokens(token_ids))

    figure, axes = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        attention_matrix.detach().cpu().numpy(),
        xticklabels=token_list,
        yticklabels=token_list,
        cmap="viridis",
        cbar=True,
        vmin=0.0,
        vmax=1.0,
        ax=axes,
    )
    axes.set_title(f"Attention - Layer {layer}, Head {head}")
    axes.set_xlabel("Keys (tokens being attended to)")
    axes.set_ylabel("Queries (tokens looking for context)")
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path)
    plt.show()
    plt.close(figure)
    return figure
