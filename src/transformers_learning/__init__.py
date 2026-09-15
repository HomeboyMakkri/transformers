"""Reusable learning utilities for the transformers project."""

from .attention import extract_attention_matrix, visualize_attention
from .baseline import (
    FrozenEmbeddingDataset,
    FrozenEmbeddingSplit,
    prepare_frozen_embedding_dataset,
    split_frozen_embedding_dataset,
    train_logistic_regression,
)
from .datasets import (
    LABEL_COLUMN,
    SST2_LABEL_MAP,
    SST2_METADATA,
    TEXT_COLUMN,
    SentimentDatasetMetadata,
    SentimentDatasetValidationError,
    adapt_sst2_split,
    get_class_counts,
    validate_sentiment_dataframe,
)
from .modeling import (
    extract_first_token_representation,
    get_embeddings,
    load_model,
    similarity,
)
from .tokenization import (
    DEFAULT_MODEL_NAME,
    TokenizationWalkthrough,
    TokenizerInfo,
    explain_tokenization,
    inspect_tokenizer,
    load_tokenizer,
    tokenize_single_text,
    tokenize_texts,
)

__all__ = [
    "DEFAULT_MODEL_NAME",
    "LABEL_COLUMN",
    "SST2_LABEL_MAP",
    "SST2_METADATA",
    "TEXT_COLUMN",
    "FrozenEmbeddingDataset",
    "FrozenEmbeddingSplit",
    "SentimentDatasetMetadata",
    "SentimentDatasetValidationError",
    "TokenizationWalkthrough",
    "TokenizerInfo",
    "adapt_sst2_split",
    "explain_tokenization",
    "extract_attention_matrix",
    "extract_first_token_representation",
    "get_class_counts",
    "get_embeddings",
    "inspect_tokenizer",
    "load_model",
    "load_tokenizer",
    "prepare_frozen_embedding_dataset",
    "similarity",
    "split_frozen_embedding_dataset",
    "tokenize_single_text",
    "tokenize_texts",
    "train_logistic_regression",
    "validate_sentiment_dataframe",
    "visualize_attention",
]
